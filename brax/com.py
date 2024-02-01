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

"""Helper functions for physics calculations in maximal coordinates."""
# pylint:disable=g-multiple-import
from typing import Tuple

from brax import math
from brax.base import Motion, System, Transform
import jax
from jax import numpy as jp


def from_world(
    sys: System, x: Transform, xd: Motion
) -> Tuple[Transform, Motion]:
  """Converts link transform and motion from world frame to com frame."""
  x_i = x.vmap().do(Transform.create(pos=sys.link.inertia.transform.pos))
  xd_i = Transform.create(pos=x_i.pos - x.pos).vmap().do(xd)
  return x_i, xd_i


def to_world(
    sys: System, x_i: Transform, xd_i: Motion
) -> Tuple[Transform, Motion]:
  """Converts link transform and motion from com frame to world frame."""
  x = x_i.vmap().do(Transform.create(pos=-sys.link.inertia.transform.pos))
  xd = Transform.create(pos=x.pos - x_i.pos).vmap().do(xd_i)
  return x, xd


def inv_inertia(sys, x) -> jax.Array:
  """Gets the inverse inertia at the center of mass in world frame."""

  @jax.vmap
  def inv_i(link_inertia, x_rot):
    ri = math.quat_mul(x_rot, link_inertia.transform.rot)
    i_diag = jp.diagonal(link_inertia.i) ** (1 - sys.spring_inertia_scale)
    i_inv_mx = jp.diag(1 / i_diag)
    i_rot_row = jax.vmap(math.rotate, in_axes=[0, None])(i_inv_mx, ri)
    i_rot_col = jax.vmap(math.rotate, in_axes=[0, None])(i_rot_row.T, ri)
    return i_rot_col

  return inv_i(sys.link.inertia, x.rot)
