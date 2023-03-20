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

"""Functions to resolve collisions."""
# pylint:disable=g-multiple-import
from typing import Optional, Tuple

from brax.v2 import math
from brax.v2.base import Contact, Motion, System, Transform, Force
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def resolve_position(
    sys: System,
    x_i: Transform,
    x_i_old: Transform,
    inv_inertia: jp.ndarray,
    inv_mass: jp.ndarray,
    contact: Optional[Contact],
) -> Tuple[Transform, jp.ndarray]:
  """Resolves springy collision constraint.

  Args:
    sys: System to forward propagate
    x_i: Transform state of link center of mass
    x_i_old: Transform state of link center of mass after velocity projection
    inv_inertia: inverse inertia tensor at the center of mass in world frame
    inv_mass: inverse link mass
    contact: Contact pytree

  Returns:
    Tuple of
    p: world space translations to apply to each link
    dlambda: normal force information
  """
  if contact is None:
    return (
        Transform(pos=jp.zeros((1, 3)), rot=jp.zeros((1, 4))),
        jp.array([0.0]),
    )

  x_i = jax.tree_map(lambda x, y: jp.vstack((x, y)), x_i, Transform.zero((1,)))
  x_i_old = jax.tree_map(
      lambda x, y: jp.vstack((x, y)), x_i_old, Transform.zero((1,))
  )

  @jax.vmap
  def translate(contact) -> Tuple[Transform, jp.ndarray, jp.ndarray]:
    link_idx = jp.array(contact.link_idx)
    x = x_i.take(link_idx)
    x_old = x_i_old.take(link_idx)

    # TODO: rewrite these updates to use pbd methods

    pos_p = contact.pos - contact.normal * contact.penetration / 2.0
    pos_c = contact.pos + contact.normal * contact.penetration / 2.0

    pos_p = pos_p - x.pos[0]
    pos_c = pos_c - x.pos[1]

    n = contact.normal
    c = -contact.penetration

    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    invmass = inv_mass.take(link_idx) * (link_idx > -1)

    # only spherical inertia effects
    cr1 = jp.cross(pos_p, n)
    w1 = invmass[0] + jp.dot(cr1, i_inv[0] @ cr1)

    cr2 = jp.cross(pos_c, n)
    w2 = invmass[1] + jp.dot(cr1, i_inv[1] @ cr1)

    dlambda = -c / (w1 + w2 + 1e-6)
    coll_mask = jp.where(c < 0, 1.0, 0.0)
    p = dlambda * n * coll_mask

    dp_p_pos = p * invmass[0]
    dp_p_rot = math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])

    dp_c_pos = -p * invmass[1]
    dp_c_rot = -math.vec_quat_mul(i_inv[1] @ jp.cross(pos_c, p), x.rot[1])

    # static friction stuff

    q1inv = math.quat_inv(x.rot[0])
    r1 = math.rotate(contact.pos - x.pos[0], q1inv)

    q2inv = math.quat_inv(x.rot[1])
    r2 = math.rotate(contact.pos - x.pos[1], q2inv)

    p1bar = x_old.pos[0] + math.rotate(r1, x_old.rot[0])
    p2bar = x_old.pos[1] + math.rotate(r2, x_old.rot[1])
    p0 = contact.pos

    deltap = (p0 - p1bar) - (p0 - p2bar)
    deltap_t = deltap - jp.dot(deltap, n) * n

    pos_p = contact.pos - x.pos[0]
    pos_c = contact.pos - x.pos[1]

    c = math.safe_norm(deltap_t)
    n = deltap_t / (c + 1e-6)

    cr1 = jp.cross(pos_p, n)
    w1 = invmass[0] + jp.dot(cr1, i_inv[0] @ cr1)

    cr2 = jp.cross(pos_c, n)
    w2 = invmass[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambdat = -c / (w1 + w2)
    static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1.0, 0.0)
    p = dlambdat * n * static_mask * coll_mask

    dp_p_pos += p * invmass[0]
    dp_p_rot += 0.5 * math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])
    dp_p = Transform(
        pos=sys.collide_scale * dp_p_pos, rot=sys.collide_scale * dp_p_rot
    )

    dp_c_pos += -p * invmass[1]
    dp_c_rot += -0.5 * math.vec_quat_mul(
        i_inv[1] @ jp.cross(pos_c, p), x.rot[1]
    )
    dp_c = Transform(
        pos=sys.collide_scale * dp_c_pos,
        rot=sys.collide_scale * dp_c_rot,
    )

    # dp = jax.tree_map(lambda x, y: jp.vstack([x, y]), dp_p, dp_c)

    return dp_p, dp_c, dlambda * coll_mask  # pytype: disable=bad-return-type  # jax-ndarray

  dp_p, dp_c, dlambda = translate(contact)
  dp = jax.tree_map(lambda x, y: jp.vstack([x, y]), dp_p, dp_c)
  dp = jax.tree_map(lambda x: jp.where(jp.isnan(x), 0.0, x), dp)
  link_idx = jp.concatenate(contact.link_idx)
  dp *= link_idx.reshape((-1, 1)) != -1

  # subtract 1 from shape to remove dummy transform
  dp = jax.tree_map(
      lambda f: jax.ops.segment_sum(f, link_idx, x_i.pos.shape[0] - 1), dp  # pytype: disable=attribute-error
  )

  return dp, dlambda


def resolve_velocity(
    sys: System,
    x_i: Transform,
    xd_i: Motion,
    x_i_old: Transform,
    xd_i_old: Motion,
    inv_inertia: jp.ndarray,
    inv_mass: jp.ndarray,
    contact: Contact,
    dlambda: jp.ndarray,
) -> Motion:
  """Velocity-level collision update for position based dynamics."""

  if contact is None:
    return Motion.zero((xd_i.vel.shape[0],))

  x_i = jax.tree_map(lambda x, y: jp.vstack((x, y)), x_i, Transform.zero((1,)))
  x_i_old = jax.tree_map(
      lambda x, y: jp.vstack((x, y)), x_i_old, Transform.zero((1,))
  )

  xd_i = jax.tree_map(lambda x, y: jp.vstack((x, y)), xd_i, Motion.zero((1,)))
  xd_i_old = jax.tree_map(
      lambda x, y: jp.vstack((x, y)), xd_i_old, Motion.zero((1,))
  )

  @jax.vmap
  def impulse(contact, dlambda):
    link_idx = jp.array(contact.link_idx)
    x = x_i.take(link_idx)
    xd = xd_i.take(link_idx)
    x_old = x_i_old.take(link_idx)
    xd_old = xd_i_old.take(link_idx)
    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    invmass = inv_mass.take(link_idx) * (link_idx > -1)

    n = contact.normal
    rel_vel = (
        xd.vel[0]
        + jp.cross(xd.ang[0], contact.pos - x.pos[0])
        - (xd.vel[1] + jp.cross(xd.ang[1], contact.pos - x.pos[1]))
    )
    v_n = jp.dot(rel_vel, n)
    v_t = rel_vel - n * v_n
    v_t_norm = math.safe_norm(v_t)
    v_t_dir = v_t / (1e-6 + v_t_norm)

    dvel = -v_t_dir * jp.amin(
        jp.array([contact.friction * jp.abs(dlambda) / (sys.dt), v_t_norm])
    )

    angw_1 = jp.cross((contact.pos - x.pos[0]), v_t_dir)
    angw_2 = jp.cross((contact.pos - x.pos[1]), v_t_dir)
    w1 = invmass[0] + jp.dot(angw_1, i_inv[0] @ angw_1)
    w2 = invmass[1] + jp.dot(angw_2, i_inv[1] @ angw_2)

    p_dyn = dvel / (w1 + w2 + 1e-6)

    # restitution calculation

    rel_vel_old = (
        xd_old.vel[0] + jp.cross(xd_old.ang[0], contact.pos - x_old.pos[0])
    ) - (xd_old.vel[1] + jp.cross(xd_old.ang[1], contact.pos - x_old.pos[1]))
    v_n_old = jp.dot(rel_vel_old, n)

    dv_rest = n * (
        -v_n - jp.amin(jp.array([contact.elasticity * v_n_old, 0.0]))
    )

    pos_p = contact.pos
    pos_c = contact.pos + contact.normal * contact.penetration
    dx = dv_rest
    pos_p = pos_p - x.pos[0]
    pos_c = pos_c - x.pos[1]

    c = math.safe_norm(dx)
    n = dx / (c + 1e-6)

    # ignoring inertial effects for now
    cr1 = jp.cross(pos_p, n)
    w1 = invmass[0] + jp.dot(cr1, i_inv[0] @ cr1)

    cr2 = jp.cross(pos_c, n)
    w2 = invmass[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambda_rest = c / (w1 + w2 + 1e-6)
    static_mask = jp.where(contact.penetration > 0, 1.0, 0.0)
    sinking = jp.where(v_n_old <= 0.0, 1.0, 0.0)

    p = Force.create(
        vel=(dlambda_rest * n * sinking + p_dyn) * static_mask
    )

    return p, jp.array(static_mask, dtype=jp.float32)

  p, is_contact = impulse(contact, dlambda)

  # calculate the impulse to each link center of mass
  p = jax.tree_map(lambda x: jp.concatenate((x, -x)), p)
  pos = jp.tile(contact.pos, (2, 1))
  link_idx = jp.concatenate(contact.link_idx)
  xp_i = Transform.create(pos=pos - x_i.take(link_idx).pos).vmap().do(p)
  xp_i = jax.tree_map(lambda x: segment_sum(x, link_idx, sys.num_links()), xp_i)

  # average the impulse across multiple contacts
  num_contacts = segment_sum(jp.tile(is_contact, 2), link_idx, sys.num_links())
  xp_i = xp_i / (num_contacts.reshape((-1, 1)) + 1e-8)

  # convert impulse to delta-velocity
  xdv_i = Motion(
      vel=jax.vmap(lambda x, y: x * y)(inv_mass, xp_i.vel),
      ang=jax.vmap(lambda x, y: x @ y)(inv_inertia, xp_i.ang),
  )

  return xdv_i
