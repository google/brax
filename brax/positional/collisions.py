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

"""Functions to resolve collisions."""
# pylint:disable=g-multiple-import
from typing import Optional, Tuple

from brax import com
from brax import math
from brax.base import Contact, Motion, System, Transform, Force
from brax.positional.base import State
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def resolve_position(
    sys: System,
    state: State,
    x_i_prev: Transform,
    contact: Optional[Contact],
) -> Tuple[Transform, jax.Array]:
  """Resolves positional collision constraint.

  The update equations follow section 3.5 of Müller et al.'s Extended Position
  Based Dynamics, where we have removed the compliance terms.
  (Müller, Matthias, et al. "Detailed rigid body simulation with extended
  position based dynamics." Computer Graphics Forum. Vol. 39. No. 8. 2020.).

  Args:
    sys: System to forward propagate
    state: positional pipeline state
    x_i_prev: center of mass position from previous step
    contact: Contact pytree

  Returns:
    x_i: new position after update
    dlambda: normal force information
  """
  if contact is None:
    return state.x_i, jp.zeros((1,))

  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)

  @jax.vmap
  def translate(contact):
    link_idx = jp.array(contact.link_idx)
    x = state.x_i.concatenate(Transform.zero((1,))).take(link_idx)
    x_prev = x_i_prev.concatenate(Transform.zero((1,))).take(link_idx)

    # TODO: rewrite these updates to use pbd methods
    n = -contact.frame[0]
    c = contact.dist
    pos_p = contact.pos + n * c / 2.0 - x.pos[0]
    pos_c = contact.pos - n * c / 2.0 - x.pos[1]

    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    mass_inv = inv_mass.take(link_idx) * (link_idx > -1)

    # only spherical inertia effects
    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambda = -c / (w1 + w2 + 1e-6)
    coll_mask = c < 0
    p = dlambda * n * coll_mask

    dp_p_pos, dp_c_pos = p * mass_inv[0], -p * mass_inv[1]
    dp_p_rot = math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])
    dp_c_rot = -math.vec_quat_mul(i_inv[1] @ jp.cross(pos_c, p), x.rot[1])

    # static friction
    q1inv, q2inv = jax.vmap(math.quat_inv)(x.rot)
    r1 = math.rotate(contact.pos - x.pos[0], q1inv)
    r2 = math.rotate(contact.pos - x.pos[1], q2inv)
    p1bar = x_prev.pos[0] + math.rotate(r1, x_prev.rot[0])
    p2bar = x_prev.pos[1] + math.rotate(r2, x_prev.rot[1])
    p0 = contact.pos

    deltap = (p0 - p1bar) - (p0 - p2bar)
    deltap_t = deltap - jp.dot(deltap, n) * n

    pos_p, pos_c = contact.pos - x.pos
    c = math.safe_norm(deltap_t)
    n = deltap_t / (c + 1e-6)

    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambdat = -c / (w1 + w2)
    static_mask = jp.where(jp.abs(dlambdat) < jp.abs(dlambda), 1.0, 0.0)
    p = dlambdat * n * static_mask * coll_mask

    dp_p_pos += p * mass_inv[0]
    dp_p_rot += 0.5 * math.vec_quat_mul(i_inv[0] @ jp.cross(pos_p, p), x.rot[0])
    dp_p = Transform(pos=dp_p_pos, rot=dp_p_rot) * sys.collide_scale

    dp_c_pos -= p * mass_inv[1]
    dp_c_rot -= 0.5 * math.vec_quat_mul(i_inv[1] @ jp.cross(pos_c, p), x.rot[1])
    dp_c = Transform(pos=dp_c_pos, rot=dp_c_rot) * sys.collide_scale

    return dp_p, dp_c, dlambda * coll_mask

  dp_p, dp_c, dlambda = translate(contact)
  dp = jax.tree.map(lambda x, y: jp.vstack([x, y]), dp_p, dp_c)
  dp = jax.tree.map(lambda x: jp.where(jp.isnan(x), 0.0, x), dp)
  link_idx = jp.concatenate(contact.link_idx)
  dp *= link_idx.reshape((-1, 1)) > -1
  dp = jax.tree.map(
      lambda f: jax.ops.segment_sum(f, link_idx, sys.num_links()), dp
  )
  x_i = state.x_i + dp
  x_i = x_i.replace(rot=jax.vmap(math.normalize)(x_i.rot)[0])

  return x_i, dlambda


def resolve_velocity(
    sys: System,
    state: State,
    xd_i_prev: Motion,
    contact: Contact,
    dlambda: jax.Array,
) -> Motion:
  """Velocity-level collision update for position based dynamics.

  The update equations here follow section 3.6 of Müller et al.'s Extended
  Position Based Dynamics. (Müller, Matthias, et al. "Detailed rigid body
  simulation with extended position based dynamics." Computer Graphics Forum.
  Vol. 39. No. 8. 2020.).

  Args:
    sys: System to update
    state: positional pipeline state
    xd_i_prev: velocity immediately preceding PBD velocity projection
    contact: Contact information for collision
    dlambda: Normal force of contact times time step squared

  Returns:
    Velocity level update for system state.
  """
  if contact is None:
    return Motion.zero((sys.num_links(),))

  x_i = state.x_i.concatenate(Transform.zero((1,)))
  inv_mass = 1 / (sys.link.inertia.mass ** (1 - sys.spring_mass_scale))
  inv_inertia = com.inv_inertia(sys, state.x)

  @jax.vmap
  def impulse(contact, dlambda):
    link_idx = jp.array(contact.link_idx)
    x = x_i.take(link_idx)
    xd = state.xd_i.concatenate(Motion.zero((1,))).take(link_idx)
    xd_prev = xd_i_prev.concatenate(Motion.zero((1,))).take(link_idx)
    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)
    mass_inv = inv_mass.take(link_idx) * (link_idx > -1)

    n = -contact.frame[0]
    rel_vel = (
        xd.vel[0]
        + jp.cross(xd.ang[0], contact.pos - x.pos[0])
        - (xd.vel[1] + jp.cross(xd.ang[1], contact.pos - x.pos[1]))
    )
    v_n = jp.dot(rel_vel, n)
    v_t = rel_vel - n * v_n
    v_t_dir, v_t_norm = math.normalize(v_t)
    dvel = -v_t_dir * jp.minimum(
        contact.friction[0] * jp.abs(dlambda) / sys.opt.timestep, v_t_norm
    )

    angw_1 = jp.cross((contact.pos - x.pos[0]), v_t_dir)
    angw_2 = jp.cross((contact.pos - x.pos[1]), v_t_dir)
    w1 = mass_inv[0] + jp.dot(angw_1, i_inv[0] @ angw_1)
    w2 = mass_inv[1] + jp.dot(angw_2, i_inv[1] @ angw_2)

    p_dyn = dvel / (w1 + w2 + 1e-6)

    # restitution
    rel_vel_prev = (
        xd_prev.vel[0] + jp.cross(xd_prev.ang[0], contact.pos - x.pos[0])
    ) - (xd_prev.vel[1] + jp.cross(xd_prev.ang[1], contact.pos - x.pos[1]))
    v_n_prev = jp.dot(rel_vel_prev, n)
    dv_rest = n * (-v_n - jp.minimum(contact.elasticity * v_n_prev, 0))

    pos_p = contact.pos
    pos_c = contact.pos + contact.frame[0] * contact.dist
    dx = dv_rest
    pos_p, pos_c = pos_p - x.pos[0], pos_c - x.pos[1]

    c = math.safe_norm(dx)
    n = dx / (c + 1e-6)

    # ignoring inertial effects for now
    cr1, cr2 = jp.cross(pos_p, n), jp.cross(pos_c, n)
    w1 = mass_inv[0] + jp.dot(cr1, i_inv[0] @ cr1)
    w2 = mass_inv[1] + jp.dot(cr2, i_inv[1] @ cr2)

    dlambda_rest = c / (w1 + w2 + 1e-6)
    penetrating = contact.dist < 0
    sinking = v_n_prev <= 0.0

    p = Force.create(vel=(dlambda_rest * n * sinking + p_dyn) * penetrating)

    return p, jp.asarray(penetrating, dtype=jp.float32)

  p, is_contact = impulse(contact, dlambda)

  # calculate the impulse to each link center of mass
  p = jax.tree.map(lambda x: jp.concatenate((x, -x)), p)
  pos = jp.tile(contact.pos, (2, 1))
  link_idx = jp.concatenate(contact.link_idx)
  xp_i = Transform.create(pos=pos - x_i.take(link_idx).pos).vmap().do(p)
  xp_i = jax.tree.map(lambda x: segment_sum(x, link_idx, sys.num_links()), xp_i)

  # average the impulse across multiple contacts
  num_contacts = segment_sum(jp.tile(is_contact, 2), link_idx, sys.num_links())
  xp_i = xp_i / (num_contacts.reshape((-1, 1)) + 1e-8)

  # convert impulse to delta-velocity
  xdv_i = Motion(
      vel=jax.vmap(lambda x, y: x * y)(inv_mass, xp_i.vel),
      ang=jax.vmap(lambda x, y: x @ y)(inv_inertia, xp_i.ang),
  )

  return xdv_i
