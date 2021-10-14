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
"""Numerical integrators."""

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import config_pb2
from brax.physics.base import P, QP, vec_to_arr


@pytree.register
class Euler:
  """Symplectic euler integration."""

  def __init__(self, config: config_pb2.Config):
    """Creates a Euler integrator.

    Args:
      config: brax config
    """
    self.pos_mask = 1. * jp.logical_not(
        jp.array([vec_to_arr(b.frozen.position) for b in config.bodies]))
    self.rot_mask = 1. * jp.logical_not(
        jp.array([vec_to_arr(b.frozen.rotation) for b in config.bodies]))
    self.dt = config.dt / config.substeps
    self.gravity = vec_to_arr(config.gravity)
    self.velocity_damping = config.velocity_damping
    self.angular_damping = config.angular_damping

  def kinetic(self, qp: QP) -> QP:
    """Performs a kinetic integration step.

    Args:
      qp: State data to be integrated

    Returns:
      State data advanced by one kinematic integration step.
    """
    @jp.vmap
    def op(qp, pos_mask, rot_mask) -> QP:
      pos = qp.pos + qp.vel * self.dt * pos_mask
      rot_at_ang_quat = math.ang_to_quat(qp.ang * rot_mask) * 0.5 * self.dt
      rot = qp.rot + math.quat_mul(rot_at_ang_quat, qp.rot)
      rot = rot / jp.norm(rot)
      return QP(pos, rot, qp.vel, qp.ang)

    return op(qp, self.pos_mask, self.rot_mask)

  def potential(self, qp: QP, dp: P) -> QP:
    """Performs a potential integration step.

    Args:
      qp: State data to be integrated
      dp: Impulses to apply during this potential step

    Returns:
      State data advanced by one potential integration step.
    """
    @jp.vmap
    def op(qp, dp, pos_mask, rot_mask) -> QP:
      vel = jp.exp(self.velocity_damping * self.dt) * qp.vel
      vel += (dp.vel + self.gravity) * self.dt
      vel *= pos_mask
      ang = jp.exp(self.angular_damping * self.dt) * qp.ang
      ang += dp.ang * self.dt
      ang *= rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

    return op(qp, dp, self.pos_mask, self.rot_mask)

  def potential_collision(self, qp: QP, dp: P) -> QP:
    """Performs a potential collision integration step.

    Args:
      qp: State data to be integrated
      dp: Velocity-level collision updates to apply this integration step

    Returns:
      State data advanced by one velocity-level update.
    """
    @jp.vmap
    def op(qp, dp, pos_mask, rot_mask) -> QP:
      vel = (qp.vel + dp.vel) * pos_mask
      ang = (qp.ang + dp.ang) * rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

    return op(qp, dp, self.pos_mask, self.rot_mask)
