# Copyright 2022 The Brax Authors.
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
import itertools
from typing import Any, Callable, List, Optional, Tuple
import warnings

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import geometry
from brax.physics.base import P, Q, QP, vec_to_arr


class Cull(abc.ABC):
  """Selects collidable pair candidates for collision detection."""

  @abc.abstractmethod
  def get(self) -> Tuple[geometry.Collidable, geometry.Collidable]:
    """Returns collidable pair candidates for collision detection."""

  def update(self, qp: QP):
    """Updates candidate statistics given qp world state."""


@pytree.register
class AllPairs(Cull):
  """Naive strategy: returns all possible pairs for collision detection."""

  def __init__(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               mask: List[Tuple[int, int]]):
    self.col_a = jp.take(col_a, [a for a, _ in mask])
    self.col_b = jp.take(col_b, [b for _, b in mask])

  def get(self) -> Tuple[geometry.Collidable, geometry.Collidable]:
    return self.col_a, self.col_b


@pytree.register
class NearNeighbors(Cull):
  """Returns top K nearest neighbor collidables for collision detection."""

  __pytree_ignore__ = ('cutoff',)

  def __init__(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               mask: List[Tuple[int, int]], cutoff: int):

    dist_off = jp.zeros(col_a.body.idx.shape + col_b.body.idx.shape)
    # TODO: consider accounting for bounds/radius of a collidable
    dist_mask = dist_off + float('inf')
    mask = (jp.array([a for a, _ in mask]), jp.array([b for _, b in mask]))
    dist_off += jp.index_update(dist_mask, mask, 0)
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
    self.collide_scale = config.solver_scale_collide or 1.0
    # updates only applied if velocity differences exceed this threshold
    self.velocity_threshold = jp.norm(vec_to_arr(config.gravity)) * self.h * 4.0

  def apply(self, qp: QP) -> P:
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
    dp_a, dp_b = jp.vmap(self._contact)(col_a, col_b, qp_a, qp_b, contact)

    if dp_b is None:
      dp_vel, dp_ang, body_idx = dp_a.vel, dp_a.ang, col_a.body.idx
    else:
      body_idx = jp.concatenate((col_a.body.idx, col_b.body.idx))
      dp_vel = jp.concatenate((dp_a.vel, dp_b.vel))
      dp_ang = jp.concatenate((dp_a.ang, dp_b.ang))
    contact = jp.where(jp.any(dp_vel, axis=-1), 1.0, 0.0)
    contact = jp.segment_sum(contact, body_idx, qp.pos.shape[0])
    dp_vel = jp.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    contact = jp.reshape(1e-8 + contact, (dp_vel.shape[0], 1))
    dp_vel = dp_vel / contact
    dp_ang = dp_ang / contact
    return P(vel=dp_vel, ang=dp_ang)

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
    dp_a, dp_b = jp.vmap(self._velocity_contact)(
        col_a,
        col_b,
        qp_a,
        qp_b,  # pytype: disable=attribute-error
        contact,
        dlambda,
        qp_a_prev,
        qp_b_prev)

    if dp_b is None:
      dp_vel, dp_ang, body_idx = dp_a.vel, dp_a.ang, col_a.body.idx
    else:
      body_idx = jp.concatenate((col_a.body.idx, col_b.body.idx))
      dp_vel = jp.concatenate((dp_a.vel, dp_b.vel))
      dp_ang = jp.concatenate((dp_a.ang, dp_b.ang))
    contact = jp.where(jp.any(dp_vel, axis=-1), 1.0, 0.0)
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

    if dq_b is None:
      dq_pos, dq_rot, body_idx = dq_a.pos, dq_a.rot, col_a.body.idx
    else:
      body_idx = jp.concatenate((col_a.body.idx, col_b.body.idx))
      dq_pos = jp.concatenate((dq_a.pos, dq_b.pos))
      dq_rot = jp.concatenate((dq_a.rot, dq_b.rot))
    contact = jp.where(jp.any(dq_pos, axis=-1), 1.0, 0.0)
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
    apply_n = jp.where(
        (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)

    dp_a = dp_n * apply_n + dp_d * apply_d
    return dp_a, None

  def _position_contact(
      self, col_a: geometry.Collidable, col_b: geometry.Collidable, qp_a: QP,
      qp_b: QP, qp_a_old: QP, qp_b_old: QP,
      contact: geometry.Contact) -> Tuple[Q, Optional[Q], jp.ndarray]:
    """Calculates impulse on a body due to a contact."""

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

    coll_mask = jp.where(c < 0, 1., 0.)
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
    static_mask = jp.where(
        jp.abs(dlambdat) < jp.abs(friction * dlambda), 1., 0.)
    p = dlambdat * n * static_mask * coll_mask

    dq_p_pos = p / col_a.body.mass
    dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                      qp_a.rot)

    dq_p = Q(
        pos=dq_p.pos + self.collide_scale * dq_p_pos,
        rot=dq_p.rot + self.collide_scale * dq_p_rot)

    return dq_p, None, dlambda * coll_mask

  def _velocity_contact(self, col_a: geometry.Collidable,
                        col_b: geometry.Collidable, qp_a: QP, qp_b: QP,
                        contact: geometry.Contact, dlambda: jp.ndarray,
                        qp_a_old: QP, qp_b_old: QP) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""
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
    static_mask = jp.where(contact.penetration > 0, 1., 0.)
    sinking = jp.where(v_n_old <= -self.velocity_threshold, 1., 0.)

    p = (dlambda_rest * n * sinking + p_dyn) * static_mask

    dp_p = P(
        vel=p / col_a.body.mass,
        ang=jp.cross(col_a.body.inertia * (contact.pos - qp_a.pos), p))

    return dp_p, None


@pytree.register
class TwoWayCollider(Collider):
  """Calculates two-way impulses on collidable pairs."""

  def _contact(self, col_a: geometry.Collidable, col_b: geometry.Collidable,
               qp_a: QP, qp_b: QP,
               contact: geometry.Contact) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""
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
    dp_d_b = col_a.body.impulse(qp_b, impulse_d * dir_d, contact.pos)
    # apply collision normal if penetrating, approaching, and oriented correctly
    apply_n = jp.where(
        (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)

    dp_a = dp_n_a * apply_n + dp_d_a * apply_d
    dp_b = dp_n_b * apply_n + dp_d_b * apply_d
    return dp_a, dp_b

  def _position_contact(
      self, col_a: geometry.Collidable, col_b: geometry.Collidable, qp_a: QP,
      qp_b: QP, qp_a_old: QP, qp_b_old: QP,
      contact: geometry.Contact) -> Tuple[Q, Optional[Q], jp.ndarray]:
    """Calculates impulse on a body due to a contact."""

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
    coll_mask = jp.where(c < 0, 1., 0.)
    p = dlambda * n * coll_mask

    dq_p_pos = p / col_a.body.mass
    dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                      qp_a.rot)

    dq_c_pos = -p / col_b.body.mass
    dq_c_rot = -.5 * math.vec_quat_mul(col_b.body.inertia * jp.cross(pos_c, p),
                                       qp_b.rot)

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
    static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1., 0.)
    p = dlambdat * n * static_mask * coll_mask

    dq_p_pos = p / col_a.body.mass
    dq_p_rot = .5 * math.vec_quat_mul(col_a.body.inertia * jp.cross(pos_p, p),
                                      qp_a.rot)

    dq_c_pos = -p / col_b.body.mass
    dq_c_rot = .5 * math.vec_quat_mul(col_b.body.inertia * jp.cross(pos_c, -p),
                                      qp_b.rot)

    dq_p += Q(
        pos=self.collide_scale * dq_p_pos, rot=self.collide_scale * dq_p_rot)
    dq_c += Q(
        pos=self.collide_scale * dq_c_pos, rot=self.collide_scale * dq_c_rot)

    return dq_p, dq_c, dlambda  # pytype: disable=bad-return-type

  def _velocity_contact(self, col_a: geometry.Collidable,
                        col_b: geometry.Collidable, qp_a: QP, qp_b: QP,
                        contact: geometry.Contact, dlambda: jp.ndarray,
                        qp_a_old: QP, qp_b_old: QP) -> Tuple[P, Optional[P]]:
    """Calculates impulse on a body due to a contact."""
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
    static_mask = jp.where(contact.penetration > 0, 1., 0.)
    sinking = jp.where(v_n_old <= 0., 1., 0.)

    p = (dlambda_rest * n * sinking + p_dyn) * static_mask

    dp_p = P(
        vel=p / col_a.body.mass,
        ang=jp.cross(col_a.body.inertia * (contact.pos - qp_a.pos), p))

    dp_c = P(
        vel=-p / col_a.body.mass,
        ang=jp.cross(col_b.body.inertia * (contact.pos - qp_b.pos), -p))

    return dp_p, dp_c


def _endpoints(end: jp.ndarray, qp: QP, offset: jp.ndarray):
  pos = qp.pos + math.rotate(offset, qp.rot)
  end = math.rotate(end, qp.rot)
  return pos + end, pos - end


def box_plane(box: geometry.BoxCorner, _: geometry.Plane, qp_a: QP,
              qp_b: QP) -> geometry.Contact:
  """Returns contact between a box corner and a plane."""
  pos, vel = qp_a.to_world(box.corner)
  normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
  penetration = jp.dot(qp_b.pos - pos, normal)
  return geometry.Contact(pos, vel, normal, penetration)


def mesh_plane(mesh: geometry.Mesh, _: geometry.Plane, qp_a: QP,
               qp_b: QP) -> geometry.Contact:
  # Mesh-plane collision is similar to box-plane collision, using the vertices
  # instead of the box corners.
  pos, vel = qp_a.to_world(mesh.vertices)
  normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
  penetration = jp.dot(qp_b.pos - pos, normal)
  return geometry.Contact(pos, vel, normal, penetration)


def box_heightmap(box: geometry.BoxCorner, hm: geometry.HeightMap, qp_a: QP,
                  qp_b: QP) -> geometry.Contact:
  """Returns contact between a box corner and a height map."""
  # TODO: this only checks box corners against height map surfaces,
  # and is missing box planes against height map points.
  # TODO: collisions are not well defined outside of the height
  # map coordinates.
  box_pos, vel = qp_a.to_world(box.corner)
  pos = math.inv_rotate(box_pos - qp_b.pos, qp_b.rot)
  uv_pos = pos[:2] / hm.cell_size

  # Find the square mesh that enclose the candidate point, which we split into
  # two triangles.
  uv_idx = jp.floor(uv_pos).astype(int)
  delta_uv = uv_pos - uv_idx
  lower_triangle = jp.sum(delta_uv) < 1
  mu = jp.where(lower_triangle, -1, 1)

  # Compute the triangle vertices (u, v) that enclose the candidate point.
  triangle_u = uv_idx[0] + jp.where(lower_triangle, jp.array([0, 1, 0]),
                                    jp.array([1, 0, 1]))
  triangle_v = uv_idx[1] + jp.where(lower_triangle, jp.array([0, 0, 1]),
                                    jp.array([1, 1, 0]))

  # Get the heights for each triangle vertice. The height data is stored in row
  # major order where the row index is the x-position and the column index is
  # the y-position. The heightmap origin (u, v) is at the top-left corner.
  h = hm.height[triangle_u, -triangle_v]

  raw_normal = jp.array([mu * (h[1] - h[0]), mu * (h[2] - h[0]), hm.cell_size])
  normal = raw_normal / jp.safe_norm(raw_normal)
  p0 = jp.array(
      [triangle_u[0] * hm.cell_size, triangle_v[0] * hm.cell_size, h[0]])
  penetration = jp.dot(p0 - pos, normal)
  # Rotate back to the world frame.
  normal = math.rotate(normal, qp_b.rot)
  return geometry.Contact(box_pos, vel, normal, penetration)


def capsule_plane(cap: geometry.CapsuleEnd, _: geometry.Plane, qp_a: QP,
                  qp_b: QP) -> geometry.Contact:
  """Returns contact between a capsule and a plane."""
  cap_end_world = qp_a.pos + math.rotate(cap.end, qp_a.rot)
  normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
  pos = cap_end_world - normal * cap.radius
  vel = qp_a.vel + jp.cross(qp_a.ang, pos - qp_a.pos)
  penetration = jp.dot(qp_b.pos - pos, normal)
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
  return geometry.Contact(pos, vel, normal, penetration)


def capsule_mesh(cap: geometry.Capsule, mesh: geometry.BaseMesh, qp_a: QP,
                 qp_b: QP) -> geometry.Contact:
  """Returns the contacts for capsule-mesh collision."""
  # Determine the capsule line.
  a, b = _endpoints(cap.end, qp_a, cap.pos)
  triangle_normal = math.rotate(mesh.face_normals, qp_b.rot)

  pt = qp_b.pos + jp.vmap(
      math.rotate, include=[True, False])(mesh.faces, qp_b.rot)
  p0, p1, p2 = pt[..., 0, :], pt[..., 1, :], pt[..., 2, :]

  segment_p, triangle_p = geometry.closest_segment_triangle_points(
      a, b, p0, p1, p2, triangle_normal)

  penetration_vec = segment_p - triangle_p
  dist = jp.safe_norm(penetration_vec)
  normal = penetration_vec / (1e-6 + dist)
  penetration = cap.radius - dist

  pos = triangle_p
  vel = qp_a.world_velocity(pos) - qp_b.world_velocity(pos)
  return geometry.Contact(pos, vel, normal, penetration)


def get(config: config_pb2.Config, body: bodies.Body) -> List[Collider]:
  """Creates all colliders given a config."""

  def key_fn(x, y):
    return tuple(sorted((body.index.get(x, -1), body.index.get(y, -1))))

  include = {key_fn(f.first, f.second) for f in config.collide_include}
  # exclude colliders for joint parents and children, unless explicitly included
  ignore = {key_fn(j.parent, j.child) for j in config.joints} - include
  # exclude self collisions within a body
  self_collide = {(body.index[b1.name], body.index[b2.name])
                  for b1, b2 in zip(config.bodies, config.bodies)}
  ignore.update(self_collide)
  # exclude colliders where both bodies are frozen
  frozen = [b.name for b in config.bodies if b.frozen.all]
  ignore.union({key_fn(x, y) for x, y in itertools.combinations(frozen, 2)})

  # flatten and emit one collider per body
  flat_bodies = []
  for b in config.bodies:
    for collider in b.colliders:
      # ignore no-contact colliders
      if collider.no_contact:
        continue

      # we treat spheres as sphere-shaped capsules with a single end
      if collider.WhichOneof('type') == 'sphere':
        new_collider = config_pb2.Collider()
        new_collider.CopyFrom(collider)
        new_collider.ClearField('sphere')
        new_collider.capsule.radius = collider.sphere.radius
        new_collider.capsule.length = 2 * collider.sphere.radius
        new_collider.capsule.end = 1
        collider = new_collider

      new_body = config_pb2.Body()
      new_body.CopyFrom(b)
      new_body.ClearField('colliders')
      new_body.colliders.append(collider)
      flat_bodies.append(new_body)

  # group by type
  type_colliders = {}
  for b in flat_bodies:
    key = b.colliders[0].WhichOneof('type')
    if key not in type_colliders:
      type_colliders[key] = []
    type_colliders[key].append(b)

  # add colliders
  supported_types = {
      ('box', 'plane'): (geometry.BoxCorner, geometry.Plane, box_plane),
      ('box', 'heightMap'):
          (geometry.BoxCorner, geometry.HeightMap, box_heightmap),
      ('capsule', 'box'):
          (geometry.Capsule, geometry.TriangulatedBox, capsule_mesh),
      ('capsule', 'plane'):
          (geometry.CapsuleEnd, geometry.Plane, capsule_plane),
      ('capsule', 'capsule'):
          (geometry.Capsule, geometry.Capsule, capsule_capsule),
      ('capsule', 'mesh'): (geometry.Capsule, geometry.Mesh, capsule_mesh),
      ('mesh', 'plane'): (geometry.PointMesh, geometry.Plane, mesh_plane),
  }
  supported_near_neighbors = {('capsule', 'capsule')}
  collidable_cache = {}

  mesh_geoms = {
      mesh_geom.name: mesh_geom for mesh_geom in config.mesh_geometries
  }

  def create_collidable(cls, cols, body):
    kwargs = {}
    if issubclass(cls, geometry.Mesh):
      kwargs = {'mesh_geoms': mesh_geoms}
    return cls(cols, body, **kwargs)

  def get_supported_types(type_a: str, type_b: str) -> Tuple[str, str]:
    # Use the supported type order if possible.
    if (type_b, type_a) in supported_types:
      return type_b, type_a
    return type_a, type_b

  ret = []
  for type_a, type_b in itertools.combinations_with_replacement(
      type_colliders, 2):
    type_a, type_b = get_supported_types(type_a, type_b)

    # calculate all possible body pairs that can collide via these two types
    cols_a, cols_b = type_colliders[type_a], type_colliders[type_b]
    if type_a == type_b:
      candidates = itertools.combinations(cols_a, 2)
    else:
      candidates = itertools.product(cols_a, cols_b)
    pairs = [(body.index[a.name], body.index[b.name]) for a, b in candidates]
    if include:
      pairs = [x for x in pairs if tuple(sorted(x)) in include]
    pairs = [x for x in pairs if tuple(sorted(x)) not in ignore]

    if not pairs:
      continue

    if (type_a, type_b) not in supported_types:
      warnings.warn(f'unsupported collider pair: {type_a}, {type_b}')
      continue

    # create our collidables
    col_cls_a, col_cls_b, contact_fn = supported_types[(type_a, type_b)]
    if col_cls_a not in collidable_cache:
      collidable_cache[col_cls_a] = create_collidable(col_cls_a, cols_a, body)
    if col_cls_b not in collidable_cache:
      collidable_cache[col_cls_b] = create_collidable(col_cls_b, cols_b, body)
    col_a = collidable_cache[col_cls_a]
    col_b = collidable_cache[col_cls_b]

    # convert pairs from body idx to collidable idx
    body_to_collidable_a, body_to_collidable_b = {}, {}
    for i, body_idx in enumerate(col_a.body.idx):
      body_to_collidable_a.setdefault(body_idx, []).append(i)
    for i, body_idx in enumerate(col_b.body.idx):
      body_to_collidable_b.setdefault(body_idx, []).append(i)
    mask = []
    for body_idx_a, body_idx_b in set(pairs):
      ias = body_to_collidable_a[body_idx_a]
      ibs = body_to_collidable_b[body_idx_b]
      for ia, ib in itertools.product(ias, ibs):
        mask.append((ia, ib))

    if config.collider_cutoff and len(pairs) > config.collider_cutoff and (
        type_a, type_b) in supported_near_neighbors:
      cull = NearNeighbors(col_a, col_b, mask, config.collider_cutoff)
    else:
      cull = AllPairs(col_a, col_b, mask)
    if all(b.frozen.all for b in cols_b):
      collider = OneWayCollider(contact_fn, cull, config)
    else:
      collider = TwoWayCollider(contact_fn, cull, config)

    ret.append(collider)

  return ret

