# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Colliders push apart bodies that are in contact."""

import abc
import functools
from typing import Any, Callable, List, Optional, Tuple

from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1 import pytree
from brax.v1.experimental.tracing import customize
from brax.v1.physics import bodies
from brax.v1.physics import config_pb2
from brax.v1.physics import geometry
from brax.v1.physics.base import P, Q, QP, vec_to_arr


class Cull(abc.ABC):
  """Selects collidable pair candidates for collision detection."""

  @abc.abstractmethod
  def get(self) -> Tuple[geometry.Collidable, geometry.Collidable]:
    """Returns collidable pair candidates for collision detection."""

  def update(self, qp: QP):
    """Updates candidate statistics given qp world state."""


@pytree.register
class Pairs(Cull):
  """Naive strategy: returns pairs provided manually."""

  def __init__(self, col_a: geometry.Collidable, col_b: geometry.Collidable):
    self.col_a = col_a
    self.col_b = col_b

  def get(self) -> Tuple[geometry.Collidable, geometry.Collidable]:
    return self.col_a, self.col_b


@pytree.register
class NearNeighbors(Cull):
  """Returns top K nearest neighbor collidables for collision detection."""

  __pytree_ignore__ = ('cutoff',)

  def __init__(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               mask: Tuple[jp.ndarray, jp.ndarray], cutoff: int):

    dist_off = jp.zeros(col_a.body.idx.shape + col_b.body.idx.shape)
    # TODO: consider accounting for bounds/radius of a collidable
    dist_mask = dist_off + float('inf')
    dist_off += jp.index_update(dist_mask, mask, 0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    self.dist_off = dist_off
    self.cutoff = cutoff
    self.candidate_a, self.candidate_b = col_a, col_b
    self.col_a, self.col_b = self.candidate_a, self.candidate_b

  def update(self, qp: QP):
    if self.candidate_a is self.candidate_b:
      pos = self.candidate_a.position(qp)
      dist = jp.vmap(lambda pt: jp.norm(pos - pt, axis=-1))(pos)
    else:
      pos_a = self.candidate_a.position(qp)
      pos_b = self.candidate_b.position(qp)
      dist = jp.vmap(lambda pt: jp.norm(pos_b - pt, axis=-1))(pos_a)
    # add in offset and convert to closeness
    sim = -(dist + self.dist_off)
    # TODO: return a status if more valid candidates than cutoff
    _, idx = jp.top_k(sim.ravel(), self.cutoff)
    self.col_a = jp.take(self.candidate_a, idx // sim.shape[-1])
    self.col_b = jp.take(self.candidate_b, idx % sim.shape[-1])

  def get(self) -> Tuple[geometry.Collidable, geometry.Collidable]:
    return self.col_a, self.col_b


class Collider(abc.ABC):
  """Calculates impulses given contacts from a contact function."""

  __pytree_ignore__ = ('contact_fn', 'cull', 'baumgarte_erp', 'collide_scale',
                       'velocity_threshold')

  def __init__(self, contact_fn: Callable[[Any, Any, QP, QP], geometry.Contact],
               cull: Cull, config: config_pb2.Config):
    """Creates a PairwiseCollider that exhaustively checks for contacts.

    Args:
      contact_fn: a function that returns contacts given collidables and qp
      cull: a cull strategy
      config: for specifying global system config
    """
    self.contact_fn = contact_fn
    self.cull = cull
    self.baumgarte_erp = config.baumgarte_erp * config.substeps / config.dt
    self.h = config.dt / config.substeps
    self.substeps = config.substeps
    self.collide_scale = config.solver_scale_collide
    # updates only applied if velocity differences exceed this threshold
    self.velocity_threshold = jp.norm(vec_to_arr(config.gravity)) * self.h * 4.0

  def apply(self, qp: QP) -> Tuple[P, geometry.Contact]:
    """Returns impulse from any potential contacts between collidables.

    Args:
      qp: Coordinate/velocity frame of the bodies.

    Returns:
      dP: Impulse to apply to the bodies in the collision.
    """
    col_a, col_b = self.cull.get()
    qp_a = jp.take(qp, col_a.body.idx)
    qp_b = jp.take(qp, col_b.body.idx)
    contact = jp.vmap(self.contact_fn)(col_a, col_b, qp_a, qp_b)
    pre_contact = contact
    dp_a, dp_b = jp.vmap(self._contact)(col_a, col_b, qp_a, qp_b, contact)

    rep_a = dp_a.vel.shape[1]
    rep_b = dp_b.vel.shape[1] if dp_b else None
    dp_a, dp_b = jp.tree_map(jp.concatenate, (dp_a, dp_b))

    if dp_b is None:
      dp_vel, dp_ang, body_idx = dp_a.vel, dp_a.ang, jp.repeat(
          col_a.body.idx, rep_a)
    else:
      body_idx = jp.concatenate(
          (jp.repeat(col_a.body.idx, rep_a), jp.repeat(col_b.body.idx, rep_b)))
      dp_vel = jp.concatenate((dp_a.vel, dp_b.vel))
      dp_ang = jp.concatenate((dp_a.ang, dp_b.ang))
    contact = jp.where(jp.any(dp_vel, axis=-1), 1.0, 0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    contact = jp.segment_sum(contact, body_idx, qp.pos.shape[0])
    dp_vel = jp.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    contact = jp.reshape(1e-8 + contact, (dp_vel.shape[0], 1))
    dp_vel = dp_vel / contact
    dp_ang = dp_ang / contact
    return P(vel=dp_vel, ang=dp_ang), pre_contact

  def velocity_apply(self, qp: QP, dlambda: jp.ndarray, qp_prev: QP,
                     contact) -> P:
    """Returns impulse from any potential contacts between collidables.

    Args:
      qp: Coordinate/velocity frame of the bodies.
      dlambda: Normal forces from position based collision pass
      qp_prev: State data before the collision pass
      contact: Contact data from the last collision pass

    Returns:
      dP: Impulse to apply to the bodies in the collision.
    """
    col_a, col_b = self.cull.get()
    qp_a, qp_a_prev = jp.take((qp, qp_prev), col_a.body.idx)
    qp_b, qp_b_prev = jp.take((qp, qp_prev), col_b.body.idx)
    dp_a, dp_b = jp.vmap(self._velocity_contact)(col_a, col_b, qp_a, qp_b,  # pytype: disable=attribute-error
                                                 contact, dlambda, qp_a_prev,
                                                 qp_b_prev)

    rep_a = dp_a.vel.shape[1]
    rep_b = dp_b.vel.shape[1] if dp_b else None
    dp_a, dp_b = jp.tree_map(jp.concatenate, (dp_a, dp_b))

    if dp_b is None:
      dp_vel, dp_ang, body_idx = dp_a.vel, dp_a.ang, jp.repeat(
          col_a.body.idx, rep_a)
    else:
      body_idx = jp.concatenate(
          (jp.repeat(col_a.body.idx, rep_a), jp.repeat(col_b.body.idx, rep_b)))
      dp_vel = jp.concatenate((dp_a.vel, dp_b.vel))
      dp_ang = jp.concatenate((dp_a.ang, dp_b.ang))
    contact = jp.where(jp.any(dp_vel, axis=-1), 1.0, 0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    contact = jp.segment_sum(contact, body_idx, qp.pos.shape[0])
    dp_vel = jp.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    contact = jp.reshape(1e-6 + contact, (dp_vel.shape[0], 1))
    dp_vel = dp_vel / contact
    dp_ang = dp_ang / contact
    return P(vel=dp_vel, ang=dp_ang)

  def position_apply(self, qp: QP,
                     qp_prev: QP) -> Tuple[Q, jp.ndarray, geometry.Contact]:
    """Returns a position based update that resolves a collisions for QP.

    Args:
      qp: Most recent state data for the system.
      qp_prev: State data before the most recent kinetic update.

    Returns:
      dQ: Changes in position and quaternion to enforce collision.
    """
    col_a, col_b = self.cull.get()
    qp_a, qp_a_prev = jp.take((qp, qp_prev), col_a.body.idx)
    qp_b, qp_b_prev = jp.take((qp, qp_prev), col_b.body.idx)

    contact = jp.vmap(self.contact_fn)(col_a, col_b, qp_a, qp_b)
    pre_contact = contact
    dq_a, dq_b, dlambda = jp.vmap(self._position_contact)(col_a, col_b, qp_a,
                                                          qp_b, qp_a_prev,
                                                          qp_b_prev, contact)

    rep_a = dq_a.pos.shape[1]
    rep_b = dq_b.pos.shape[1] if dq_b else None
    dq_a, dq_b = jp.tree_map(jp.concatenate, (dq_a, dq_b))

    if dq_b is None:
      dq_pos, dq_rot, body_idx = dq_a.pos, dq_a.rot, jp.repeat(
          col_a.body.idx, rep_a)
    else:
      body_idx = jp.concatenate(
          (jp.repeat(col_a.body.idx, rep_b), jp.repeat(col_b.body.idx, rep_b)))
      dq_pos = jp.concatenate((dq_a.pos, dq_b.pos))
      dq_rot = jp.concatenate((dq_a.rot, dq_b.rot))
    contact = jp.where(jp.any(dq_pos, axis=-1), 1.0, 0.0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    contact = jp.segment_sum(contact, body_idx, qp.pos.shape[0])
    dq_pos = jp.segment_sum(dq_pos, body_idx, qp.pos.shape[0])
    dq_rot = jp.segment_sum(dq_rot, body_idx, qp.rot.shape[0])

    # equally distribute impulse over possible contacts
    contact = jp.reshape(1e-6 + contact, (dq_pos.shape[0], 1))
    dq_pos = dq_pos / contact
    dq_rot = dq_rot / contact
    return Q(pos=dq_pos, rot=dq_rot), dlambda, pre_contact

  @abc.abstractmethod
  def _contact(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               qp_a: QP, qp_b: QP,
               contact: geometry.Contact) -> Tuple[P, Optional[P]]:
    pass

  @abc.abstractmethod
  def _position_contact(
      self, col_a: geometry.Collidable, col_b: geometry.Collidable, qp_a: QP,
      qp_b: QP, qp_a_old: QP, qp_b_old: QP,
      contact: geometry.Contact) -> Tuple[Q, Optional[Q], jp.ndarray]:
    pass

  @abc.abstractmethod
  def _velocity_contact(self, col_a: geometry.Collidable,
                        col_b: geometry.Collidable, qp_a: QP, qp_b: QP,
                        contact: geometry.Contact, dlambda: jp.ndarray,
                        qp_a_old: QP, qp_b_old: QP) -> Tuple[P, Optional[P]]:
    pass


@pytree.register
class OneWayCollider(Collider):
  """Calculates one-way impulses, where the second collidable is static."""

  def _contact(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               qp_a: QP, qp_b: QP,
               contact: geometry.Contact) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact):
      # there are a few ways to combine material properties during contact.
      # multiplying is a reasonable default.  in the future we may allow others
      elasticity = col_a.elasticity * col_b.elasticity
      friction = col_a.friction * col_b.friction
      rel_pos = contact.pos - qp_a.pos
      baumgarte_vel = self.baumgarte_erp * contact.penetration
      normal_vel = jp.dot(contact.normal, contact.vel)
      temp1 = col_a.body.inertia * jp.cross(rel_pos, contact.normal)
      ang = jp.dot(contact.normal, jp.cross(temp1, rel_pos))
      impulse = (-1. * (1. + elasticity) * normal_vel + baumgarte_vel) / (
          (1. / col_a.body.mass) + ang)
      dp_n = col_a.body.impulse(qp_a, impulse * contact.normal, contact.pos)

      # apply drag due to friction acting parallel to the surface contact
      vel_d = contact.vel - normal_vel * contact.normal
      impulse_d = jp.safe_norm(vel_d) / ((1. / (col_a.body.mass)) + ang)
      # drag magnitude cannot exceed max friction
      impulse_d = jp.minimum(impulse_d, friction * impulse)
      dir_d = vel_d / (1e-6 + jp.safe_norm(vel_d))
      dp_d = col_a.body.impulse(qp_a, -impulse_d * dir_d, contact.pos)
      # apply collision if penetrating, approaching, and oriented correctly
      apply_n = jp.where(  # pytype: disable=wrong-arg-types  # jax-ndarray
          (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1.,
          0.)
      # apply drag if moving laterally above threshold
      apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray

      dp_a = dp_n * apply_n + dp_d * apply_d
      return dp_a, None

    return _v_contact(contact)

  def _position_contact(
      self, col_a: geometry.Collidable, col_b: geometry.Collidable, qp_a: QP,
      qp_b: QP, qp_a_old: QP, qp_b_old: QP,
      contact: geometry.Contact) -> Tuple[Q, Optional[Q], jp.ndarray]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact):

      friction = col_a.friction * col_b.friction

      pos_p = contact.pos
      pos_c = contact.pos + contact.normal * contact.penetration
      dx = pos_p - pos_c
      pos_p = pos_p - qp_a.pos
      pos_c = pos_c - qp_b.pos

      n = contact.normal
      c = jp.dot(dx, n)

      # only spherical inertias for now
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      dlambda = -c / (w1 + 1e-6)

      coll_mask = jp.where(c < 0, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray
      p = dlambda * n * coll_mask

      dq_p_pos = p / col_a.body.mass
      dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                        qp_a.rot)

      dq_p = Q(
          pos=self.collide_scale * dq_p_pos, rot=self.collide_scale * dq_p_rot)

      # static friction

      q1inv = math.quat_inv(qp_a.rot)
      r1 = math.rotate(contact.pos - qp_a.pos, q1inv)

      p1bar = qp_a_old.pos + math.rotate(r1, qp_a_old.rot)
      p1 = contact.pos

      deltap = p1 - p1bar
      deltap_t = deltap - jp.dot(deltap, n) * n

      dx = deltap_t

      c = jp.safe_norm(dx)
      n = dx / (c + 1e-6)

      # using spherical inertia
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      dlambdat = -c / (w1 + 0.)
      static_mask = jp.where(  # pytype: disable=wrong-arg-types  # jax-ndarray
          jp.abs(dlambdat) < jp.abs(friction * dlambda), 1., 0.)
      p = dlambdat * n * static_mask * coll_mask

      dq_p_pos = p / col_a.body.mass
      dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                        qp_a.rot)

      dq_p = Q(
          pos=dq_p.pos + self.collide_scale * dq_p_pos,
          rot=dq_p.rot + self.collide_scale * dq_p_rot)

      return dq_p, None, dlambda * coll_mask

    return _v_contact(contact)

  def _velocity_contact(self, col_a: geometry.Collidable,
                        col_b: geometry.Collidable, qp_a: QP, qp_b: QP,
                        contact: geometry.Contact, dlambda: jp.ndarray,
                        qp_a_old: QP, qp_b_old: QP) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact, dlambda):
      # there are a few ways to combine material properties during contact.
      # multiplying is a reasonable default.  in the future we may allow others

      # dynamic friction calculation
      elasticity = col_a.elasticity * col_b.elasticity
      friction = col_a.friction * col_b.friction

      n = contact.normal
      rel_vel = qp_a.vel + jp.cross(qp_a.ang, contact.pos - qp_a.pos)
      v_n = jp.dot(rel_vel, n)
      v_t = rel_vel - n * v_n
      v_t_norm = jp.safe_norm(v_t)
      v_t_dir = v_t / (1e-6 + v_t_norm)

      # factor of 2 from integrator doing 1 collision pass every 2 steps
      dvel = -v_t_dir * jp.amin(
          jp.array([friction * jp.abs(dlambda) / (2. * self.h), v_t_norm]))

      angw = jp.cross((contact.pos - qp_a.pos), v_t_dir)
      w = (1. / col_a.body.mass) + jp.dot(angw, angw)

      p_dyn = dvel / (w + 1e-6)

      # restitution calculation

      rel_vel_old = qp_a_old.vel + jp.cross(qp_a_old.ang,
                                            contact.pos - qp_a_old.pos)
      v_n_old = jp.dot(rel_vel_old, n)

      dv_rest = n * (-v_n - jp.amin(jp.array([elasticity * v_n_old, 0.])))

      pos_p = contact.pos

      dx = dv_rest
      pos_p = pos_p - qp_a.pos

      c = jp.safe_norm(dx)
      n = dx / (c + 1e-6)

      # only spherical inertia effects
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      dlambda_rest = c / (w1 + 1e-6)
      static_mask = jp.where(contact.penetration > 0, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray
      sinking = jp.where(v_n_old <= -self.velocity_threshold, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray

      p = (dlambda_rest * n * sinking + p_dyn) * static_mask

      dp_p = P(
          vel=p / col_a.body.mass,
          ang=jp.cross(col_a.body.inertia * (contact.pos - qp_a.pos), p))

      return dp_p, None

    return _v_contact(contact, dlambda)


@pytree.register
class TwoWayCollider(Collider):
  """Calculates two-way impulses on collidable pairs."""

  def _contact(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               qp_a: QP, qp_b: QP,
               contact: geometry.Contact) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact):
      # there are a few ways to combine material properties during contact.
      # multiplying is a reasonable default.  in the future we may allow others
      elasticity = col_a.elasticity * col_b.elasticity
      friction = col_a.friction * col_b.friction
      rel_pos_a = contact.pos - qp_a.pos
      rel_pos_b = contact.pos - qp_b.pos
      baumgarte_vel = self.baumgarte_erp * contact.penetration
      normal_vel = jp.dot(contact.normal, contact.vel)
      temp1 = col_a.body.inertia * jp.cross(rel_pos_a, contact.normal)
      temp2 = col_b.body.inertia * jp.cross(rel_pos_b, contact.normal)
      ang = jp.dot(contact.normal,
                   jp.cross(temp1, rel_pos_a) + jp.cross(temp2, rel_pos_b))
      impulse = (-1. * (1. + elasticity) * normal_vel + baumgarte_vel) / (
          (1. / col_a.body.mass) + (1. / col_b.body.mass) + ang)
      dp_n_a = col_a.body.impulse(qp_a, impulse * contact.normal, contact.pos)
      dp_n_b = col_b.body.impulse(qp_b, -impulse * contact.normal, contact.pos)

      # apply drag due to friction acting parallel to the surface contact
      vel_d = contact.vel - normal_vel * contact.normal
      impulse_d = jp.safe_norm(vel_d) / ((1. / col_a.body.mass) +
                                         (1. / col_b.body.mass) + ang)
      # drag magnitude cannot exceed max friction
      impulse_d = jp.minimum(impulse_d, friction * impulse)
      dir_d = vel_d / (1e-6 + jp.safe_norm(vel_d))
      dp_d_a = col_a.body.impulse(qp_a, -impulse_d * dir_d, contact.pos)
      dp_d_b = col_b.body.impulse(qp_b, impulse_d * dir_d, contact.pos)
      # apply collision if penetrating, approaching, and oriented correctly
      apply_n = jp.where(  # pytype: disable=wrong-arg-types  # jax-ndarray
          (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1.,
          0.)
      # apply drag if moving laterally above threshold
      apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray

      dp_a = dp_n_a * apply_n + dp_d_a * apply_d
      dp_b = dp_n_b * apply_n + dp_d_b * apply_d
      return dp_a, dp_b

    return _v_contact(contact)

  def _position_contact(
      self, col_a: geometry.Collidable, col_b: geometry.Collidable, qp_a: QP,
      qp_b: QP, qp_a_old: QP, qp_b_old: QP,
      contact: geometry.Contact) -> Tuple[Q, Optional[Q], jp.ndarray]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact):
      pos_p = contact.pos - contact.normal * contact.penetration / 2.
      pos_c = contact.pos + contact.normal * contact.penetration / 2.
      pos_p = pos_p - qp_a.pos
      pos_c = pos_c - qp_b.pos

      n = contact.normal
      c = -contact.penetration

      # only spherical inertia effects
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      cr2 = jp.cross(pos_c, n)
      w2 = (1. / col_b.body.mass) + jp.dot(cr2, col_b.body.inertia * cr2)

      dlambda = -c / (w1 + w2 + 1e-6)
      coll_mask = jp.where(c < 0, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray
      p = dlambda * n * coll_mask

      dq_p_pos = p / col_a.body.mass
      dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                        qp_a.rot)

      dq_c_pos = -p / col_b.body.mass
      dq_c_rot = -.5 * math.vec_quat_mul(
          col_b.body.inertia * jp.cross(pos_c, p), qp_b.rot)

      dq_p = Q(
          pos=self.collide_scale * dq_p_pos, rot=self.collide_scale * dq_p_rot)
      dq_c = Q(
          pos=self.collide_scale * dq_c_pos, rot=self.collide_scale * dq_c_rot)

      # static friction stuff

      q1inv = math.quat_inv(qp_a.rot)
      r1 = math.rotate(contact.pos - qp_a.pos, q1inv)

      q2inv = math.quat_inv(qp_b.rot)
      r2 = math.rotate(contact.pos - qp_b.pos, q2inv)

      p1bar = qp_a_old.pos + math.rotate(r1, qp_a_old.rot)
      p2bar = qp_b_old.pos + math.rotate(r2, qp_b_old.rot)
      p0 = contact.pos

      deltap = (p0 - p1bar) - (p0 - p2bar)
      deltap_t = deltap - jp.dot(deltap, n) * n

      pos_p = contact.pos - qp_a.pos
      pos_c = contact.pos - qp_b.pos

      c = jp.safe_norm(deltap_t)
      n = deltap_t / (c + 1e-6)

      # ignoring inertial effects for now
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      cr2 = jp.cross(pos_c, n)
      w2 = (1. / col_b.body.mass) + jp.dot(cr2, col_b.body.inertia * cr2)

      dlambdat = -c / (w1 + w2)
      static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray
      p = dlambdat * n * static_mask * coll_mask

      dq_p_pos = p / col_a.body.mass
      dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                        qp_a.rot)

      dq_c_pos = -p / col_b.body.mass
      dq_c_rot = .5 * math.vec_quat_mul(
          col_b.body.inertia * jp.cross(pos_c, -p), qp_b.rot)

      dq_p += Q(
          pos=self.collide_scale * dq_p_pos, rot=self.collide_scale * dq_p_rot)
      dq_c += Q(
          pos=self.collide_scale * dq_c_pos, rot=self.collide_scale * dq_c_rot)

      return dq_p, dq_c, dlambda  # pytype: disable=bad-return-type

    return _v_contact(contact)  # pytype: disable=bad-return-type

  def _velocity_contact(self, col_a: geometry.Collidable,
                        col_b: geometry.Collidable, qp_a: QP, qp_b: QP,
                        contact: geometry.Contact, dlambda: jp.ndarray,
                        qp_a_old: QP, qp_b_old: QP) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""

    @jp.vmap
    def _v_contact(contact, dlambda):
      # there are a few ways to combine material properties during contact.
      # multiplying is a reasonable default.  in the future we may allow others

      # dynamic friction calculation
      friction = col_a.friction * col_b.friction
      elasticity = col_a.elasticity * col_b.elasticity
      n = contact.normal
      rel_vel = qp_a.vel + jp.cross(qp_a.ang, contact.pos - qp_a.pos) - (
          qp_b.vel + jp.cross(qp_b.ang, contact.pos - qp_b.pos))
      v_n = jp.dot(rel_vel, n)
      v_t = rel_vel - n * v_n
      v_t_norm = jp.safe_norm(v_t)
      v_t_dir = v_t / (1e-6 + v_t_norm)

      # factor of 2 from integrator doing 1 collision pass every 2 steps
      dvel = -v_t_dir * jp.amin(
          jp.array([friction * jp.abs(dlambda) / (2. * self.h), v_t_norm]))

      angw_1 = jp.cross((contact.pos - qp_a.pos), v_t_dir)
      angw_2 = jp.cross((contact.pos - qp_b.pos), v_t_dir)
      w1 = (1. / col_a.body.mass) + jp.dot(angw_1, col_a.body.inertia * angw_1)
      w2 = (1. / col_b.body.mass) + jp.dot(angw_2, col_b.body.inertia * angw_2)

      p_dyn = dvel / (w1 + w2 + 1e-6)

      # restitution calculation

      rel_vel_old = (
          qp_a_old.vel + jp.cross(qp_a_old.ang, contact.pos - qp_a_old.pos)) - (
              qp_b_old.vel + jp.cross(qp_b_old.ang, contact.pos - qp_b_old.pos))
      v_n_old = jp.dot(rel_vel_old, n)

      dv_rest = n * (-v_n - jp.amin(jp.array([elasticity * v_n_old, 0.])))

      pos_p = contact.pos
      pos_c = contact.pos + contact.normal * contact.penetration
      dx = dv_rest
      pos_p = pos_p - qp_a.pos
      pos_c = pos_c - qp_b.pos

      c = jp.safe_norm(dx)
      n = dx / (c + 1e-6)

      # ignoring inertial effects for now
      cr1 = jp.cross(pos_p, n)
      w1 = (1. / col_a.body.mass) + jp.dot(cr1, col_a.body.inertia * cr1)

      cr2 = jp.cross(pos_c, n)
      w2 = (1. / col_b.body.mass) + jp.dot(cr2, col_b.body.inertia * cr2)

      dlambda_rest = c / (w1 + w2 + 1e-6)
      static_mask = jp.where(contact.penetration > 0, 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray
      sinking = jp.where(v_n_old <= 0., 1., 0.)  # pytype: disable=wrong-arg-types  # jax-ndarray

      p = (dlambda_rest * n * sinking + p_dyn) * static_mask

      dp_p = P(
          vel=p / col_a.body.mass,
          ang=jp.cross(col_a.body.inertia * (contact.pos - qp_a.pos), p))

      dp_c = P(
          vel=-p / col_b.body.mass,
          ang=jp.cross(col_b.body.inertia * (contact.pos - qp_b.pos), -p))

      return dp_p, dp_c

    return _v_contact(contact, dlambda)


def _endpoints(end: jp.ndarray, qp: QP, offset: jp.ndarray):
  pos = qp.pos + math.rotate(offset, qp.rot)
  end = math.rotate(end, qp.rot)
  return pos + end, pos - end


def box_plane(box: geometry.Box, _: geometry.Plane, qp_a: QP,
              qp_b: QP) -> geometry.Contact:
  """Returns vectorized contacts between a box and a plane."""

  @jp.vmap
  def point_plane(corner):
    pos, vel = qp_a.to_world(corner)
    normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
    penetration = jp.dot(qp_b.pos - pos, normal)
    return pos, vel, normal, penetration

  pos, vel, normal, penetration = point_plane(box.corner)

  return geometry.Contact(pos, vel, normal, penetration)


def mesh_plane(mesh: geometry.Mesh, _: geometry.Plane, qp_a: QP,
               qp_b: QP) -> geometry.Contact:
  """Similar to box-plane collision, but uses vertices instead of corners."""

  @jp.vmap
  def point_plane(vertices):
    pos, vel = qp_a.to_world(vertices)
    normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
    penetration = jp.dot(qp_b.pos - pos, normal)
    return pos, vel, normal, penetration

  pos, vel, normal, penetration = point_plane(mesh.vertices)

  return geometry.Contact(pos, vel, normal, penetration)


def box_heightmap(box: geometry.Box, hm: geometry.HeightMap, qp_a: QP,
                  qp_b: QP) -> geometry.Contact:
  """Returns contact between a box corner and a height map."""
  # TODO: this only checks box corners against height map surfaces,
  # and is missing box planes against height map points.
  # TODO: collisions are not well defined outside of the height
  # map coordinates.
  @jp.vmap
  def corner_heightmap(corner):
    box_pos, vel = qp_a.to_world(corner)
    pos = math.inv_rotate(box_pos - qp_b.pos, qp_b.rot)
    uv_pos = pos[:2] / hm.cell_size

    # Find the square mesh that enclose the candidate point, which we split into
    # two triangles.
    uv_idx = jp.floor(uv_pos).astype(int)
    delta_uv = uv_pos - uv_idx
    lower_triangle = jp.sum(delta_uv) < 1
    mu = jp.where(lower_triangle, -1, 1)  # pytype: disable=wrong-arg-types  # jax-ndarray

    # Compute the triangle vertices (u, v) that enclose the candidate point.
    triangle_u = uv_idx[0] + jp.where(lower_triangle, jp.array([0, 1, 0]),
                                      jp.array([1, 0, 1]))
    triangle_v = uv_idx[1] + jp.where(lower_triangle, jp.array([0, 0, 1]),
                                      jp.array([1, 1, 0]))

    # Get the heights for each triangle vertice. Height data is stored in row
    # major order where the row index is the x-position and the column index is
    # the y-position. The heightmap origin (u, v) is at the top-left corner.
    h = hm.height[triangle_u, -triangle_v]

    raw_normal = jp.array(
        [mu * (h[1] - h[0]), mu * (h[2] - h[0]), hm.cell_size])
    normal = raw_normal / jp.safe_norm(raw_normal)
    p0 = jp.array(
        [triangle_u[0] * hm.cell_size, triangle_v[0] * hm.cell_size, h[0]])
    penetration = jp.dot(p0 - pos, normal)
    # Rotate back to the world frame.
    normal = math.rotate(normal, qp_b.rot)
    return box_pos, vel, normal, penetration

  pos, vel, normal, penetration = corner_heightmap(box.corner)
  return geometry.Contact(pos, vel, normal, penetration)


def capsule_plane(cap: geometry.CapsuleEnd, _: geometry.Plane, qp_a: QP,
                  qp_b: QP) -> geometry.Contact:
  """Returns contact between a capsule and a plane."""

  @jp.vmap
  def sphere_plane(end):
    cap_end_world = qp_a.pos + math.rotate(end, qp_a.rot)
    normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
    pos = cap_end_world - normal * cap.radius
    vel = qp_a.vel + jp.cross(qp_a.ang, pos - qp_a.pos)
    penetration = jp.dot(qp_b.pos - pos, normal)
    return pos, vel, normal, penetration

  pos, vel, normal, penetration = sphere_plane(cap.end)

  return geometry.Contact(pos, vel, normal, penetration)


def capsule_clippedplane(cap: geometry.CapsuleEnd, plane: geometry.ClippedPlane,
                         qp_a: QP, qp_b: QP) -> geometry.Contact:
  """Returns contact between a capsule and a clipped plane."""

  @jp.vmap
  def sphere_clippedplane(end):
    cap_end_world = qp_a.pos + math.rotate(end, qp_a.rot)
    normal = math.rotate(plane.normal, qp_b.rot)

    # orient the normal s.t. it points at the CoM of the capsule
    normal_dir = jp.where(qp_a.pos.dot(normal) > 0., 1, -1)  # pytype: disable=wrong-arg-types  # jax-ndarray
    normal = normal * normal_dir

    pos = cap_end_world - normal * cap.radius
    vel = qp_a.vel + jp.cross(qp_a.ang, pos - qp_a.pos)
    plane_pt = math.rotate(plane.pos, qp_b.rot) + qp_b.pos
    penetration = jp.dot(plane_pt - pos, normal)

    # Clip against side planes.
    norm_x = math.rotate(plane.x, qp_b.rot)
    norm_y = math.rotate(plane.y, qp_b.rot)
    side_plane_pt = jp.array([
        plane_pt + norm_x * plane.halfsize_x,
        plane_pt - norm_x * plane.halfsize_x,
        plane_pt + norm_y * plane.halfsize_y,
        plane_pt - norm_y * plane.halfsize_y])
    yn, xn = jp.cross(normal * normal_dir, norm_x), -jp.cross(
        normal * normal_dir, norm_y)
    side_plane_norm = jp.array([xn, -xn, yn, -yn])
    in_front_of_side_plane = jp.vmap(geometry.point_in_front_of_plane,
                                     include=[True, True, False])(
                                         side_plane_pt, side_plane_norm, pos)
    penetration = jp.where(jp.any(in_front_of_side_plane),  # pytype: disable=wrong-arg-types  # jax-ndarray
                           -jp.ones_like(penetration),
                           penetration)

    return pos, vel, normal, penetration

  pos, vel, normal, penetration = sphere_clippedplane(cap.end)

  return geometry.Contact(pos, vel, normal, penetration)


def capsule_capsule(cap_a: geometry.Capsule, cap_b: geometry.Capsule, qp_a: QP,
                    qp_b: QP) -> geometry.Contact:
  """Returns contact between two capsules."""
  a0, a1 = _endpoints(cap_a.end, qp_a, cap_a.pos)
  b0, b1 = _endpoints(cap_b.end, qp_b, cap_b.pos)
  a_best, b_best = geometry.closest_segment_to_segment_points(a0, a1, b0, b1)

  penetration_vec = a_best - b_best
  dist = jp.safe_norm(penetration_vec)
  normal = penetration_vec / (1e-6 + dist)
  penetration = cap_a.radius + cap_b.radius - dist
  pos = (a_best + b_best) / 2
  vel = qp_a.world_velocity(pos) - qp_b.world_velocity(pos)
  out = geometry.Contact(pos, vel, normal, penetration)
  return jp.tree_map(jp.expand_dims, out)


def capsule_mesh(cap: geometry.Capsule, mesh: geometry.BaseMesh, qp_a: QP,
                 qp_b: QP) -> geometry.Contact:
  """Returns the contacts for capsule-mesh collision."""

  @jp.vmap
  def capsule_face(faces, face_normals):
    # Determine the capsule line.
    a, b = _endpoints(cap.end, qp_a, cap.pos)
    triangle_normal = math.rotate(face_normals, qp_b.rot)

    pt = qp_b.pos + jp.vmap(math.rotate, include=[True, False])(faces, qp_b.rot)
    p0, p1, p2 = pt[..., 0, :], pt[..., 1, :], pt[..., 2, :]

    segment_p, triangle_p = geometry.closest_segment_triangle_points(
        a, b, p0, p1, p2, triangle_normal)

    penetration_vec = segment_p - triangle_p
    dist = jp.safe_norm(penetration_vec)
    normal = penetration_vec / (1e-6 + dist)
    penetration = cap.radius - dist

    pos = triangle_p
    vel = qp_a.world_velocity(pos) - qp_b.world_velocity(pos)
    return pos, vel, normal, penetration

  pos, vel, normal, penetration = capsule_face(mesh.faces, mesh.face_normals)
  return geometry.Contact(pos, vel, normal, penetration)


def hull_hull(mesh_a: geometry.BaseMesh, mesh_b: geometry.BaseMesh, qp_a: QP,
              qp_b: QP) -> geometry.Contact:
  """Gets hull-hull contacts."""

  @jp.vmap
  def get_faces(faces_a, faces_b, normals_a, normals_b):
    faces_a = qp_a.pos + jp.vmap(
        math.rotate, include=[True, False])(faces_a, qp_a.rot)
    faces_b = qp_b.pos + jp.vmap(
        math.rotate, include=[True, False])(faces_b, qp_b.rot)
    normals_a = math.rotate(normals_a, qp_a.rot)
    normals_b = math.rotate(normals_b, qp_b.rot)
    return faces_a, faces_b, normals_a, normals_b

  @jp.vmap
  def get_verts(vertices_a, vertices_b):
    vertices_a = qp_a.pos + math.rotate(vertices_a, qp_a.rot)
    vertices_b = qp_b.pos + math.rotate(vertices_b, qp_b.rot)
    return vertices_a, vertices_b

  faces_a, faces_b, normals_a, normals_b = get_faces(mesh_a.faces, mesh_b.faces,
                                                     mesh_a.face_normals,
                                                     mesh_b.face_normals)
  vertices_a, vertices_b = get_verts(mesh_a.vertices, mesh_b.vertices)

  # Create a potential face and edge contact using SAT.
  edge_contact, face_contact = geometry.sat_hull_hull(faces_a, faces_b,
                                                      vertices_a, vertices_b,
                                                      normals_a, normals_b)

  # Pick a face or edge as the final contact.
  contact = jp.cond(edge_contact.penetration[0] > 0, lambda *x: edge_contact,
                    lambda *x: face_contact)

  get_vel = lambda p: qp_a.world_velocity(p) - qp_b.world_velocity(p)
  contact.vel = jp.vmap(get_vel)(contact.pos)

  return contact


def get(config: config_pb2.Config, body: bodies.Body) -> List[Collider]:
  """Creates all colliders given a config."""

  mesh_geoms = {mg.name: mg for mg in config.mesh_geometries}
  collider_pairs = {
      ('box', 'plane'): (geometry.Box, geometry.Plane, box_plane),
      ('box', 'heightMap'): (geometry.Box, geometry.HeightMap, box_heightmap),
      ('capsule', 'box'):
          (geometry.Capsule, geometry.TriangulatedBox, capsule_mesh),
      ('capsule', 'plane'):
          (geometry.CapsuleEnd, geometry.Plane, capsule_plane),
      ('capsule', 'capsule'):
          (geometry.Capsule, geometry.Capsule, capsule_capsule),
      ('capsule', 'mesh'):
          (geometry.Capsule,
           functools.partial(geometry.Mesh,
                             mesh_geoms=mesh_geoms), capsule_mesh),
      ('capsule', 'clipped_plane'): (geometry.CapsuleEnd, geometry.ClippedPlane,
                                     capsule_clippedplane),
      ('mesh', 'plane'):
          (functools.partial(geometry.PointMesh, mesh_geoms=mesh_geoms),
           geometry.Plane, mesh_plane),
      ('box', 'box'): (geometry.HullBox, geometry.HullBox, hull_hull),
  }
  supported_near_neighbors = {('capsule', 'capsule')}
  unique_meshes = {}
  cols = []
  for b in config.bodies:
    for c_idx, c in enumerate(b.colliders):
      if c.no_contact:
        continue
      if c.WhichOneof('type') == 'sphere':
        c = c if isinstance(c, config_pb2.Collider) else c.msg
        nc = config_pb2.Collider()
        nc.CopyFrom(c)
        nc.capsule.radius = c.sphere.radius
        nc.capsule.length = 2 * c.sphere.radius
        nc.capsule.end = 1
        c = nc
      if c.WhichOneof('type') == 'mesh':
        unique_meshes[c.mesh.name] = 1
      cols.append((c, b, c_idx))

  include = {(ci.first, ci.second) for ci in config.collide_include}
  parents = ((j.parent, j.child) for j in config.joints)
  ret = []

  for (type_a, type_b), (cls_a, cls_b, contact_fn) in collider_pairs.items():
    replicas = unique_meshes.keys() if 'mesh' in (type_a, type_b) else [None]
    for mesh_name in replicas:
      cols_a, cols_b = [], []
      for (cols_i, type_i) in [(cols_a, type_a), (cols_b, type_b)]:
        for c, b, c_idx in cols:
          if c.WhichOneof('type') == type_i:
            if type_i == 'mesh' and c.mesh.name != mesh_name:
              continue  # only add colliders with the same mesh
            cols_i.append((c, b, c_idx))

      cols_a = [(c, b, c_idx) for c, b, c_idx in cols_a if not b.frozen.all]

      cols_ab = []

      pair_count = {}
      for ca, ba, ca_idx in cols_a:
        for cb, bb, cb_idx in cols_b:
          included = (ba.name, bb.name) in include or (bb.name,
                                                       ba.name) in include
          if (ba.name, ca_idx, bb.name,
              cb_idx) in pair_count or (bb.name, cb_idx, ba.name,
                                        ca_idx) in pair_count:
            continue  # don't double count collisions
          if ba.name == bb.name:
            continue  # no self-collision
          if ba.frozen.all and bb.frozen.all:
            continue  # two immovable bodies cannot collide
          if (ba.name, bb.name
             ) in parents or (bb.name, ba.name) in parents and not included:
            continue  # ignore colliders for joint parents and children
          if ca.no_contact or cb.no_contact:
            continue  # ignore colliders that are purely visual

          if not include or included:
            # two cases:
            # 1. includes in config. then, only include if set in config
            # 2. includes not set in config, then include all collisions
            cols_ab.append((ca, ca_idx, ba, cb, cb_idx, bb))
            pair_count[(ba.name, ca_idx, bb.name, cb_idx)] = 1
            pair_count[(bb.name, cb_idx, ba.name, ca_idx)] = 1

      for b_is_frozen in (True, False):
        cols_ab_filtered = [
            x for x in cols_ab if x[-1].frozen.all == b_is_frozen
        ]
        if not cols_ab_filtered:
          continue

        bodies_a, bodies_b = [], []
        unique_check, unique_bodies = {}, []
        for (ca, ca_idx, ba, cb, cb_idx, bb) in cols_ab_filtered:
          for c, c_idx, b, arr in [(ca, ca_idx, ba, bodies_a),
                                   (cb, cb_idx, bb, bodies_b)]:
            nb = config_pb2.Body()
            bb = b if isinstance(b, config_pb2.Body) else b.msg
            cc = c if isinstance(c, config_pb2.Collider) else c.msg
            nb.CopyFrom(bb)
            nb.ClearField('colliders')
            nb.colliders.append(cc)
            if not isinstance(b, config_pb2.Body):
              arr.append(customize.TracedConfig(nb, custom_tree=b.custom_tree))
            else:
              arr.append(nb)
            if (b.name, c_idx) not in unique_check:
              unique_bodies.append(nb)
              unique_check[(b.name, c_idx)] = 1
        if config.collider_cutoff and len(
            bodies_a) > config.collider_cutoff and (
                type_a, type_b) in supported_near_neighbors:
          # pytype: disable=wrong-arg-types
          col_a = cls_a(bodies_a, body)
          col_b = cls_b(bodies_b, body)
          cull = NearNeighbors(
              cls_a(unique_bodies, body), cls_b(unique_bodies, body),
              (col_a.body.idx, col_b.body.idx), config.collider_cutoff)
        else:
          # pytype: disable=wrong-arg-types
          cull = Pairs(cls_a(bodies_a, body), cls_b(bodies_b, body))
        if b_is_frozen:
          collider = OneWayCollider(contact_fn, cull, config)
        else:
          collider = TwoWayCollider(contact_fn, cull, config)
        ret.append(collider)

  return ret
