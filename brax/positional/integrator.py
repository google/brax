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

"""Functions for integrating maximal coordinate dynamics."""
# pylint:disable=g-multiple-import
from typing import Tuple

from brax import math
from brax.base import Motion, System, Transform
import jax
from jax import numpy as jp


def integrate_xdv(sys: System, xd: Motion, xdv: Motion) -> Motion:
  """Updates velocity by applying delta-velocity.

  Args:
    sys: System to forward propagate
    xd: velocity
    xdv: delta-velocity

  Returns:
    xd: updated velocity
  """
  damp = Motion(vel=sys.vel_damping, ang=sys.ang_damping)
  xd = (
      jax.tree.map(lambda d, x: jp.exp(d * sys.opt.timestep) * x, damp, xd)
      + xdv
  )

  return xd


def integrate_xdd(
    sys: System,
    x: Transform,
    xd: Motion,
    xdd: Motion,
) -> Tuple[Transform, Motion]:
  """Updates position and velocity for a system time step given acceleration.

  Args:
    sys: System to forward propagate
    x: position
    xd: velocity
    xdd: acceleration

  Returns:
    x: updated position
    xd: updated velocity
  """

  xd = xd + xdd * sys.opt.timestep
  damp = Motion(vel=sys.vel_damping, ang=sys.ang_damping)
  xd = jax.tree.map(lambda d, x: jp.exp(d * sys.opt.timestep) * x, damp, xd)

  @jax.vmap
  def op(x, xd):
    pos = x.pos + xd.vel * sys.opt.timestep
    rot_at_ang_quat = math.ang_to_quat(xd.ang) * 0.5 * sys.opt.timestep
    rot, _ = math.normalize(x.rot + math.quat_mul(rot_at_ang_quat, x.rot))
    return Transform(pos=pos, rot=rot)

  x = op(x, xd)

  return x, xd


def project_xd(sys: System, x: Transform, x_prev: Transform) -> Motion:
  """Performs the position based dynamics velocity projection step.

  The velocity and angular velocity must respect the spatial and quaternion
  distance (respectively) between x and x_prev.

  Args:
    sys: The system definition
    x: The current transform
    x_prev: The transform at the previous step

  Returns:
    New state with velocity pinned to respect distance traveled since x_prev
  """

  @jax.vmap
  def op(x, x_prev):
    vel = (x.pos - x_prev.pos) / sys.opt.timestep
    dq = math.relative_quat(x_prev.rot, x.rot)
    ang = 2.0 * dq[1:] / sys.opt.timestep
    scale = jp.where(dq[0] >= 0.0, 1.0, -1.0)
    ang = scale * ang
    return Motion(vel=vel, ang=ang)

  return op(x, x_prev)
