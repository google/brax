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


def integrate(
    sys: System,
    x_i: Transform,
    xd_i: Motion,
    xdv_i: Motion,
) -> Tuple[Transform, Motion]:
  """Updates state with velocity update in the center of mass frame.

  Args:
    sys: System to forward propagate
    x_i: link center of mass transform in world frame
    xd_i: link center of mass motion in world frame
    xdv_i: link center of mass delta-velocity in world frame

  Returns:
    x_i: updated link center of mass transform in world frame
    xd_i: updated link center of mass motion in world frame
  """

  @jax.vmap
  def op(x_i, xd_i, xdv_i):
    # damp velocity and add acceleration
    xd_i = Motion(
        vel=jp.exp(sys.vel_damping * sys.opt.timestep) * xd_i.vel,
        ang=jp.exp(sys.ang_damping * sys.opt.timestep) * xd_i.ang,
    )
    xd_i += xdv_i

    rot_at_ang_quat = math.ang_to_quat(xd_i.ang) * 0.5 * sys.opt.timestep
    rot = x_i.rot + math.quat_mul(rot_at_ang_quat, x_i.rot)
    x_i = Transform(
        pos=x_i.pos + xd_i.vel * sys.opt.timestep, rot=rot / jp.linalg.norm(rot)
    )

    return x_i, xd_i

  return op(x_i, xd_i, xdv_i)
