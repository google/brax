# Copyright 2021 The Brax Authors.
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

# pylint:disable=g-multiple-import
"""A collection of integrators."""

import jax
import jax.numpy as jnp

from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import P, QP, vec_to_np


def kinetic(_, qp: QP, dt: float, active: jnp.ndarray) -> QP:
  """Performs a kinetic integration step."""

  @jax.vmap
  def op(qp: QP, dt: float) -> QP:
    pos = qp.pos + qp.vel * dt
    rot_at_ang_quat = math.ang_to_quat(qp.ang)
    rot = jnp.matmul(
        jnp.matmul(jnp.eye(4) + .5 * dt * rot_at_ang_quat, qp.rot),
        jnp.eye(4) - .5 * dt * rot_at_ang_quat)
    rot = rot / jnp.linalg.norm(rot)
    return QP(pos=pos, rot=rot, vel=qp.vel, ang=qp.ang)

  return op(qp, active * dt)


def potential(config: config_pb2.Config, qp: QP, dp: P, dt: float,
              active: jnp.ndarray) -> QP:
  """Performs a potential integration step."""

  @jax.vmap
  def op(qp: QP, dp: P, dt: float) -> QP:
    vel = jnp.exp(config.velocity_damping *
                  dt) * qp.vel + (dp.vel + vec_to_np(config.gravity)) * dt
    ang = jnp.exp(config.angular_damping * dt) * qp.ang + dp.ang * dt
    return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

  return op(qp, dp, active * dt)


def potential_collision(_, qp: QP, dp: P, active: jnp.ndarray) -> QP:
  """Performs a potential collision integration step."""

  @jax.vmap
  def op(qp: QP, dp: P, dt: float) -> QP:
    vel = qp.vel + dp.vel * dt
    ang = qp.ang + dp.ang * dt
    return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

  return op(qp, dp, active * 1.)
