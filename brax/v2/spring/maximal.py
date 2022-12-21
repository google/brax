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

"""Helper functions for physics calculations in maximal coordinates."""
# pylint:disable=g-multiple-import
from typing import Tuple, Optional

from brax.v2 import math
from brax.v2.base import Motion, System, Transform
import jax
from jax import numpy as jp


def maximal_to_com(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion]:
  """Translates link-frame Transforms and Motions into CoM-frame."""
  xi = x.vmap().do(sys.link.inertia.transform)
  com_transform = Transform(pos=xi.pos, rot=x.rot)
  com_motion = Motion(
      vel=xd.vel + jax.vmap(jp.cross)(xd.ang, xi.pos - x.pos), ang=xd.ang
  )
  return com_transform, com_motion


def com_to_maximal(
    x_com: Transform, xd_com: Motion, coord_transform: Transform
) -> Tuple[Transform, Motion]:
  """Translates CoM-frame Transforms and Motions into link-frame."""

  final_shift = jax.vmap(math.rotate)(
      jax.vmap(math.rotate)(
          coord_transform.pos, math.quat_inv(coord_transform.rot)
      ),
      x_com.rot,
  )

  maximal_transform = Transform(pos=x_com.pos - final_shift, rot=x_com.rot)
  maximal_motion = Motion(
      vel=xd_com.vel - jax.vmap(jp.cross)(xd_com.ang, final_shift),
      ang=xd_com.ang,
  )

  return maximal_transform, maximal_motion


def com_inv_inertia(sys, x) -> jp.ndarray:
  """Gets the inverse inertia at the center of mass in world frame."""

  @jax.vmap
  def inv_i(link_inertia, x_rot):
    r_inv = math.quat_inv(link_inertia.transform.rot)
    ri = math.quat_mul(r_inv, x_rot)
    i_rot_row = jax.vmap(math.rotate, in_axes=[0, None])(link_inertia.i, ri)
    i_rot_col = jax.vmap(math.rotate, in_axes=[0, None])(i_rot_row.T, ri)
    return math.inv_3x3(i_rot_col)

  return inv_i(sys.link.inertia, x.rot)


def to_world(x, xd, rpos: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
  """Returns world information about a point relative to a part.

  Args:
    x: Transform world
    xd: Motion world
    rpos: Point relative to center of mass of part.

  Returns:
      world_pos: World-space coordinates of rpos
      world_vel: World-space velocity of rpos
  """
  rpos_off = math.rotate(rpos, x.rot)
  rvel = jp.cross(xd.ang, rpos_off)
  return x.pos + rpos_off, xd.vel + rvel


def world_impulse(
    x,
    mass,
    inv_inertia,
    impulse: jp.ndarray,
    pos: jp.ndarray,
    torque: Optional[jp.ndarray],
) -> Motion:
  """Calculates updates to world state information based on an impulse.

  Args:
    x: world position
    mass: body mass
    inv_inertia: body inverse inertia
    impulse: impulse vector
    pos: location of the impulse relative to the body's center of mass
    torque: additional torque to apply

  Returns:
    Change in velocity in world space
  """
  if torque is None:
    torque = jp.zeros(3)
  dvel = impulse / mass
  dang = inv_inertia @ (jp.cross(pos - x, impulse) + torque)
  return Motion(vel=dvel, ang=dang)
