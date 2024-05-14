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

"""Function to resolve collisions."""
# pylint:disable=g-multiple-import
from brax import contact
from brax import math
from brax.base import Force, Motion, System, Transform
from brax.spring.base import State
import jax
from jax import numpy as jp
from jax.ops import segment_sum


def resolve(sys: System, state: State) -> Motion:
  """Resolves springy collision constraint.

  Args:
    sys: System to forward propagate
    state: spring pipeline state

  Returns:
    xdv_i: delta-velocity to apply to link center of mass in world frame
  """
  c = contact.get(sys, state.x)

  if c is None:
    return Motion.zero((sys.num_links(),))

  @jax.vmap
  def impulse(c, link_idx, x_i, xd_i, i_inv, i_mass):
    """Calculates a velocity level update with baumgarte stabilization."""
    rel_pos = c.pos - x_i.pos
    rel_vel = xd_i.vel + jax.vmap(jp.cross)(xd_i.ang, rel_pos)
    rel_vel *= (link_idx > -1).reshape(-1, 1)
    contact_vel = rel_vel[0] - rel_vel[1]
    normal_vel = jp.dot(-c.frame[0], contact_vel)

    # the form of these update equations closely follows the smooth constraint
    # solver in Heiden et al's Tiny Differentiable Simulator
    temp = jax.vmap(jp.dot)(i_inv, jp.cross(rel_pos, -c.frame[0]))
    ang = jp.dot(-c.frame[0], jp.sum(jp.cross(temp, rel_pos), axis=0))
    baumgarte_vel = sys.baumgarte_erp / sys.opt.timestep * c.dist
    impulse = (-1.0 * (1.0 + c.elasticity) * normal_vel - baumgarte_vel) / (
        i_mass[0] + i_mass[1] + ang
    )
    impulse_vec = impulse * -c.frame[0]

    # apply drag due to friction acting parallel to the surface contact
    vel_d = contact_vel + normal_vel * c.frame[0]
    dir_d = vel_d / (1e-6 + math.safe_norm(vel_d))
    temp = jax.vmap(jp.dot)(i_inv, jp.cross(rel_pos, dir_d))
    ang_d = jp.dot(dir_d, jp.sum(jp.cross(temp, rel_pos), axis=0))
    impulse_d = math.safe_norm(vel_d) / (i_mass[0] + i_mass[1] + ang_d)

    # drag magnitude cannot exceed max friction
    impulse_d = jp.minimum(impulse_d, c.friction[0] * impulse)
    impulse_d_vec = -1.0 * impulse_d * dir_d

    # apply collision if penetrating, approaching, and oriented correctly
    apply_n = (c.dist < 0.0) & (normal_vel < 0) & (impulse > 0.0)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * (math.safe_norm(vel_d) > 1e-3)

    f = Force.create(vel=impulse_vec * apply_n + impulse_d_vec * apply_d)

    return f, jp.array(apply_n, dtype=jp.float32)

  link_idx = jp.array(c.link_idx).T
  x_i, xd_i = state.x_i.take(link_idx), state.xd_i.take(link_idx)
  i_inv = state.i_inv.take(link_idx) * (link_idx > -1)
  i_mass = 1 / state.mass.take(link_idx) * (link_idx > -1)
  p, is_contact = impulse(c, link_idx, x_i, xd_i, i_inv, i_mass)

  # calculate the impulse to each link center of mass
  p = jax.tree.map(lambda x: jp.concatenate((x, -x)), p)
  pos = jp.tile(c.pos, (2, 1))
  link_idx = jp.concatenate(c.link_idx)
  xp_i = Transform.create(pos=pos - state.x_i.take(link_idx).pos).vmap().do(p)
  xp_i = jax.tree.map(lambda x: segment_sum(x, link_idx, sys.num_links()), xp_i)

  # average the impulse across multiple contacts
  num_contacts = segment_sum(jp.tile(is_contact, 2), link_idx, sys.num_links())
  xp_i = xp_i / (num_contacts.reshape((-1, 1)) + 1e-8)

  # convert impulse to delta-velocity
  xdv_i = Motion(
      vel=jax.vmap(lambda x, y: x / y)(xp_i.vel, state.mass),
      ang=jax.vmap(lambda x, y: x @ y)(state.i_inv, xp_i.ang),
  )

  return xdv_i
