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

"""Functions for integrating maximal coordinate dynamics."""
# pylint:disable=g-multiple-import
from typing import Optional, Tuple

from brax.v2 import math
from brax.v2.base import Motion, System, Transform
from brax.v2.spring import maximal
import jax
from jax import numpy as jp


def _kinetic(sys: System, x: Transform, xd: Motion) -> Transform:
  """Performs a kinetic integration step.

  Args:
    sys: System to integrate
    x: Current world Transform
    xd: Current world Motion

  Returns:
    Position state integrated with current velocity state.
  """

  @jax.vmap
  def op(x: Transform, xd: Motion) -> Transform:
    pos = x.pos + xd.vel * sys.dt
    rot_at_ang_quat = math.ang_to_quat(xd.ang) * 0.5 * sys.dt
    rot = x.rot + math.quat_mul(rot_at_ang_quat, x.rot)
    rot = rot / jp.linalg.norm(rot)
    return Transform(pos=pos, rot=rot)

  return op(x, xd)


def _potential(
    sys: System,
    x: Transform,
    xd: Motion,
    dxdd: Optional[Motion] = None,
    dxd: Optional[Motion] = None,
    dx: Optional[Transform] = None,
) -> Motion:
  """Performs an arg dependent integrator step.

  Args:
    sys: System to integrate
    x: Current world Transform
    xd: Current world Motion
    dxdd: Acceleration level update
    dxd: Velocity level update
    dx: Position level update

  Returns:
    State data advanced by one potential integration step.
  """

  del x, dx

  @jax.vmap
  def op_acc(xd, dxdd) -> Motion:
    vel = jp.exp(sys.vel_damping * sys.dt) * xd.vel
    vel += (dxdd.vel + sys.gravity) * sys.dt
    ang = jp.exp(sys.ang_damping * sys.dt) * xd.ang
    ang += dxdd.ang * sys.dt
    return Motion(vel=vel, ang=ang)

  # TODO: fill-in position level updates

  if dxd is not None:
    return xd + dxd
  elif dxdd is not None:
    return op_acc(xd, dxdd)
  return xd


def forward(
    sys: System,
    xi: Transform,
    xdi: Motion,
    inv_inertia: jp.ndarray,
    f: Motion,
    pos: jp.ndarray,
    link_idx: jp.ndarray,
) -> Motion:
  """Updates state with forces in the center of mass frame.

  Args:
    sys: System to forward propagate
    xi: Transform state of links to update in maximal CoM coordinates
    xdi: Motion state of links to update in maximal CoM coordinates
    inv_inertia: Inverse inertia matrix at the CoM in world frame
    f: Motion with dim (m, 3) with forces to apply to links
    pos: (m, 3) ndarray of world-frame locations to apply force
    link_idx: (m,) ndarray of link indices which receive `f` forces

  Returns:
    Updated state Motion.
  """
  # apply forces
  xi_all = xi.take(link_idx)
  links = sys.link.take(link_idx)
  inv_inertia = inv_inertia.take(link_idx, axis=0)
  xddi_all = jax.vmap(maximal.world_impulse)(
      xi_all.pos, links.inertia.mass, inv_inertia, f.vel, pos, torque=f.ang
  )
  # sum forces over links
  dxddi = jax.tree_map(
      lambda f: jax.ops.segment_sum(f, link_idx, xi.pos.shape[0]), xddi_all
  )
  xdi = _potential(sys, xi, xdi, dxdd=dxddi)
  return xdi


def forward_c(
    sys: System,
    xi: Transform,
    xdi: Motion,
    inv_inertia: jp.ndarray,
    p: Motion,
    pos: jp.ndarray,
    link_idx: jp.ndarray,
) -> Tuple[Transform, Motion]:
  """Updates state with velocity update in the center of mass frame.

  Args:
    sys: System to forward propagate
    xi: Transform state of links to update in maximal CoM coordinates
    xdi: Motion state of links to update in maximal CoM coordinates
    inv_inertia: Inverse inertia matrix at the CoM in world frame
    p: Motion with dim (m, 3) of impulses to apply to links
    pos: (m, 3) ndarray of world-frame locations to apply impulse
    link_idx: (m,) ndarray of link indices which receive impulse

  Returns:
    Updated state Transform and Motion.
  """
  # transform into center of mass frame
  if link_idx.shape[0] != 0:
    # apply velocity update `p`
    xi_all = xi.take(link_idx)
    links = sys.link.take(link_idx)
    inv_inertia = inv_inertia.take(link_idx, axis=0)
    xdi_all = jax.vmap(maximal.world_impulse)(
        xi_all.pos, links.inertia.mass, inv_inertia, p.vel, pos, torque=None
    )
    # average xd over links
    apply_v = jp.where(jp.any(p.vel, axis=-1) & (link_idx != -1), 1.0, 0.0)
    n_v = jax.ops.segment_sum(apply_v, link_idx, sys.num_links())
    n_v = jp.reshape(1e-8 + n_v, (sys.num_links(), 1))
    dxdi = jax.tree_map(
        lambda p: jax.ops.segment_sum(  # pylint:disable=g-long-lambda
            p, link_idx, xi.pos.shape[0]
        )
        / n_v,
        xdi_all,
    )
    xdi = _potential(sys, xi, xdi, dxd=dxdi)

  xi = _kinetic(sys, xi, xdi)
  return xi, xdi
