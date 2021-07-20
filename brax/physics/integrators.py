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


def kinetic(_, qp: QP, dt: float, active_pos: jnp.ndarray,
            active_rot: jnp.ndarray) -> QP:
  """Performs a kinetic integration step.

  Args:
    qp: State data to be integrated
    dt: Timestep length
    active_pos: [# bodies, 3] mask for active translational dofs
    active_rot: [# bodes, 3] mask for active rotational dofs

  Returns:
    State data advanced by one kinematic integration step.
  """

  @jax.vmap
  def op(qp: QP, active_pos: jnp.ndarray,
         active_rot: jnp.ndarray) -> QP:
    pos = qp.pos + qp.vel * dt * active_pos
    rot_at_ang_quat = math.ang_to_quat(qp.ang * active_rot)
    rot = jnp.matmul(
        jnp.matmul(jnp.eye(4) + .5 * dt * rot_at_ang_quat, qp.rot),
        jnp.eye(4) - .5 * dt * rot_at_ang_quat)
    rot = rot / jnp.linalg.norm(rot)
    return QP(pos=pos, rot=rot, vel=qp.vel, ang=qp.ang)

  return op(qp, active_pos, active_rot)


def potential(config: config_pb2.Config, qp: QP, dp: P, dt: float,
              active_pos: jnp.ndarray, active_rot: jnp.ndarray) -> QP:
  """Performs a potential integration step.

  Args:
    config: Brax system configuration
    qp: State data to be integrated
    dp: Impulses to apply during this potential step
    dt: Timestep length
    active_pos: [# bodies, 3] mask for active translational dofs
    active_rot: [# bodes, 3] mask for active rotational dofs

  Returns:
    State data advanced by one potential integration step.
  """

  @jax.vmap
  def op(qp: QP, dp: P, active_pos: jnp.ndarray,
         active_rot: jnp.ndarray) -> QP:
    vel = (jnp.exp(config.velocity_damping * dt) * qp.vel +
           (dp.vel + vec_to_np(config.gravity)) * dt) * active_pos
    ang = (jnp.exp(config.angular_damping * dt) * qp.ang +
           dp.ang * dt) * active_rot
    return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

  return op(qp, dp, active_pos, active_rot)


def potential_collision(_, qp: QP, dp: P, active_pos: jnp.ndarray,
                        active_rot: jnp.ndarray) -> QP:
  """Performs a potential collision integration step.

  Args:
    qp: State data to be integrated
    dp: Velocity-level collision updates to apply this integration step
    active_pos: [# bodies, 3] mask for active translational dofs
    active_rot: [# bodes, 3] mask for active rotational dofs

  Returns:
    State data advanced by one velocity-level update."""

  @jax.vmap
  def op(qp: QP, dp: P, active_pos: jnp.ndarray,
         active_rot: jnp.ndarray) -> QP:
    vel = (qp.vel + dp.vel) * active_pos
    ang = (qp.ang + dp.ang) * active_rot
    return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

  return op(qp, dp, active_pos, active_rot)
