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

"""Function to resolve collisions."""
# pylint:disable=g-multiple-import
from typing import Optional, Tuple

from brax.v2 import math
from brax.v2.base import Contact, Motion, System, Transform
import jax
from jax import numpy as jp
from jax import tree_map


def resolve(
    sys: System,
    xi: Transform,
    xdi: Motion,
    inv_inertia: jp.ndarray,
    contact: Optional[Contact],
) -> Tuple[Motion, jp.ndarray, jp.ndarray]:
  """Resolves springy collision constraint.

  Args:
    sys: System to forward propagate
    xi: Transform state of link center of mass
    xdi: Motion state of link center of mass
    inv_inertia: inverse inertia tensor at the center of mass in world frame
    contact: Contact pytree

  Returns:
    Tuple of
    p: world space impulse to apply to each link
    positions: location in world space to apply each impulse
    idxs: link to which impulse is applied
  """
  if contact is None:
    return Motion.zero((1,)), jp.array([]), jp.array([])

  @jax.vmap
  def impulse(contact):
    link_idx = jp.array(contact.link_idx)
    rel_pos = contact.pos - xi.take(link_idx).pos
    xd = xdi.take(link_idx)
    rel_vel = xd.vel + jax.vmap(jp.cross)(xd.ang, rel_pos)
    rel_vel *= (link_idx > -1).reshape(-1, 1)
    contact_vel = rel_vel[0] - rel_vel[1]
    normal_vel = jp.dot(contact.normal, contact_vel)

    link = sys.link.take(link_idx)
    i_inv = inv_inertia.take(link_idx, axis=0)
    i_inv *= (link_idx > -1).reshape(-1, 1, 1)

    temp1 = i_inv[0] @ jp.cross(rel_pos[0], contact.normal)
    temp2 = i_inv[1] @ jp.cross(rel_pos[1], contact.normal)
    ang = jp.dot(
        contact.normal,
        jp.cross(temp1, rel_pos[0]) + jp.cross(temp2, rel_pos[1]),
    )
    invmass = (1 / link.inertia.mass) * (link_idx > -1)
    denom = invmass[0] + invmass[1] + ang
    baumgarte_vel = sys.baumgarte_erp / sys.dt * contact.penetration
    impulse = (
        -1.0 * (1.0 + contact.elasticity) * normal_vel + baumgarte_vel
    ) / denom
    impulse_vec = impulse * contact.normal

    # apply drag due to friction acting parallel to the surface contact
    vel_d = contact_vel - normal_vel * contact.normal
    dir_d = vel_d / (1e-6 + math.safe_norm(vel_d))
    temp1 = i_inv[0] @ jp.cross(rel_pos[0], dir_d)
    temp2 = i_inv[1] @ jp.cross(rel_pos[1], dir_d)
    ang_d = jp.dot(
        dir_d, jp.cross(temp1, rel_pos[0]) + jp.cross(temp2, rel_pos[1])
    )
    impulse_d = math.safe_norm(vel_d) / (invmass[0] + invmass[1] + ang_d)

    # drag magnitude cannot exceed max friction
    impulse_d = jp.minimum(impulse_d, contact.friction * impulse)
    impulse_d_vec = -1.0 * impulse_d * dir_d

    # apply collision if penetrating, approaching, and oriented correctly
    apply_n = (contact.penetration >= 0.0) & (normal_vel < 0) & (impulse > 0.0)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * (math.safe_norm(vel_d) > 1e-3)

    return impulse_vec * apply_n + impulse_d_vec * apply_d, contact.pos

  p, pos = impulse(contact)

  link_idx = jp.concatenate(contact.link_idx)
  p = tree_map(lambda *t: jp.concatenate(t), p, -p)
  p *= link_idx.reshape((-1, 1)) != -1
  p = Motion.create(vel=p)
  pos = jp.tile(pos, (2, 1))

  return p, pos, link_idx
