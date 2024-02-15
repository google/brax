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

# pylint:disable=g-multiple-import, g-importing-member
"""Functions for forces/torques through fluids."""

from typing import Union
from brax.base import Force, Motion, System, Transform
import jax
import jax.numpy as jp


def _box_viscosity(box: jax.Array, xd_i: Motion, viscosity: jax.Array) -> Force:
  """Gets force due to motion through a viscous fluid."""
  diam = jp.mean(box, axis=-1)
  ang_scale = -jp.pi * diam**3 * viscosity
  vel_scale = -3.0 * jp.pi * diam * viscosity
  frc = Force(
      ang=ang_scale[:, None] * xd_i.ang, vel=vel_scale[:, None] * xd_i.vel
  )
  return frc


def _box_density(box: jax.Array, xd_i: Motion, density: jax.Array) -> Force:
  """Gets force due to motion through dense fluid."""

  @jax.vmap
  def apply(b: jax.Array, xd: Motion) -> Force:
    box_mult_vel = jp.array([b[1] * b[2], b[0] * b[2], b[0] * b[1]])
    vel = -0.5 * density * box_mult_vel * jp.abs(xd.vel) * xd.vel
    box_mult_ang = jp.array([
        b[0] * (b[1] ** 4 + b[2] ** 4),
        b[1] * (b[0] ** 4 + b[2] ** 4),
        b[2] * (b[0] ** 4 + b[1] ** 4),
    ])
    ang = -1.0 * density * box_mult_ang * jp.abs(xd.ang) * xd.ang / 64.0
    return Force(vel=vel, ang=ang)

  return apply(box, xd_i)


def force(
    sys: System,
    x: Transform,
    xd: Motion,
    mass: jax.Array,
    inertia: jax.Array,
    root_com: Union[jax.Array, None] = None,
) -> Force:
  """Returns force due to motion through a fluid."""
  # get the velocity at the com position/orientation
  x_i = x.vmap().do(sys.link.inertia.transform)
  # TODO: remove root_com when xd is fixed for stacked joints
  offset = x_i.pos - x.pos if root_com is None else x_i.pos - root_com
  xd_i = x_i.replace(pos=offset).vmap().do(xd)

  # TODO: add ellipsoid fluid model from mujoco
  # TODO: consider adding wind from mj.opt.wind
  diag_inertia = jax.vmap(jp.diag)(inertia)
  diag_inertia_v = jp.repeat(diag_inertia, 3, axis=-2).reshape((-1, 3, 3))
  diag_inertia_v *= jp.ones((3, 3)) - 2 * jp.eye(3)
  box = 6.0 * jp.clip(jp.sum(diag_inertia_v, axis=-1), a_min=1e-12)
  box = jp.sqrt(box / mass[:, None])

  frc = _box_viscosity(box, xd_i, sys.viscosity)
  frc += _box_density(box, xd_i, sys.density)

  # rotate back to the world orientation
  frc = Transform.create(rot=x_i.rot).vmap().do(frc)

  return frc
