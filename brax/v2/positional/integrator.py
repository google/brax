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
from typing import Tuple

from brax.v2 import math
from brax.v2.base import Motion, System, Transform
import jax
from jax import numpy as jp


def integrate(
    sys: System,
    x_i: Transform,
    xd_i: Motion,
    xdv_i: Motion,
    include_kinetic: bool = True,
) -> Tuple[Transform, Motion]:
  """Updates state with velocity update in the center of mass frame.

  Args:
    sys: System to forward propagate
    x_i: link center of mass transform in world frame
    xd_i: link center of mass motion in world frame
    xdv_i: link center of mass delta-velocity in world frame
    include_kinetic: whether to update the positional data

  Returns:
    x_i: updated link center of mass transform in world frame
    xd_i: updated link center of mass motion in world frame
  """

  @jax.vmap
  def op(x_i, xd_i, xdv_i):
    # damp velocity and add acceleration
    xd_i = Motion(
        vel=jp.exp(sys.vel_damping * sys.dt) * xd_i.vel,
        ang=jp.exp(sys.ang_damping * sys.dt) * xd_i.ang)
    xd_i += xdv_i

    if include_kinetic:
      rot_at_ang_quat = math.ang_to_quat(xd_i.ang) * 0.5 * sys.dt
      rot = x_i.rot + math.quat_mul(rot_at_ang_quat, x_i.rot)
      x_i = Transform(
          pos=x_i.pos + xd_i.vel * sys.dt, rot=rot / jp.linalg.norm(rot)
      )

    return x_i, xd_i

  return op(x_i, xd_i, xdv_i)


def kinetic(sys: System, x: Transform, xd: Motion) -> Transform:
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


def velocity_projection(
    sys: System, x: Transform, x_prev: Transform
) -> Tuple[Transform, Motion]:
  """Performs the position based dynamics velocity projection step.

  The velocity and angular velocity must respect the spatial and quaternion
  distance (respectively) between qp and qpold.

  Args:
    sys: The system definition
    x: The current transform
    x_prev: The transform at the previous step

  Returns:
    New state with velocity pinned to respect distance traveled since x_prev
  """

  @jax.vmap
  def op(x, x_prev) -> Tuple[Transform, Motion]:
    new_rot, _ = math.normalize(x.rot)
    vel = (x.pos - x_prev.pos) / sys.dt
    dq = math.relative_quat(x_prev.rot, new_rot)
    ang = 2.0 * dq[1:] / sys.dt
    scale = jp.where(dq[0] >= 0.0, 1.0, -1.0)
    ang = scale * ang
    return Transform(pos=x.pos, rot=new_rot), Motion(vel=vel, ang=ang)

  return op(x, x_prev)
