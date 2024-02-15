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

# pylint:disable=g-multiple-import
"""Numerical integrators."""

import abc
from typing import Optional

from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1 import pytree
from brax.v1.physics import config_pb2
from brax.v1.physics.base import P, Q, QP, vec_to_arr


@pytree.register
class Euler(abc.ABC):
  """Base integrator class."""

  def __init__(self, config: config_pb2.Config):
    """Creates an integrator.

    Args:
      config: brax config
    """
    self.pos_mask = 1. * jp.logical_not(
        jp.array([vec_to_arr(b.frozen.position) for b in config.bodies]))
    self.rot_mask = 1. * jp.logical_not(
        jp.array([vec_to_arr(b.frozen.rotation) for b in config.bodies]))
    self.quat_mask = 1. * jp.logical_not(
        jp.array([[0.] + list(vec_to_arr(b.frozen.rotation))
                  for b in config.bodies]))
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

  def update(self,
             qp: QP,
             acc_p: Optional[P] = None,
             vel_p: Optional[P] = None,
             pos_q: Optional[Q] = None) -> QP:
    """Performs an arg dependent integrator step.

    Args:
      qp: State data to be integrated
      acc_p: Acceleration level updates to apply to qp
      vel_p: Velocity level updates to apply to qp
      pos_q: Position level updates to apply to qp

    Returns:
      State data advanced by one potential integration step.
    """

    @jp.vmap
    def op_acc(qp, dp, pos_mask, rot_mask) -> QP:
      vel = jp.exp(self.velocity_damping * self.dt) * qp.vel  # pytype: disable=wrong-arg-types  # jax-ndarray
      vel += (dp.vel + self.gravity) * self.dt
      vel *= pos_mask
      ang = jp.exp(self.angular_damping * self.dt) * qp.ang  # pytype: disable=wrong-arg-types  # jax-ndarray
      ang += dp.ang * self.dt
      ang *= rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

    @jp.vmap
    def op_vel(qp, dp, pos_mask, rot_mask) -> QP:
      vel = (qp.vel + dp.vel) * pos_mask
      ang = (qp.ang + dp.ang) * rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)

    @jp.vmap
    def op_pos(qp, dq, pos_mask, rot_mask) -> QP:
      qp = QP(
          pos=qp.pos + dq.pos * pos_mask,
          rot=qp.rot + dq.rot * rot_mask,
          ang=qp.ang,
          vel=qp.vel)
      return qp

    if acc_p:
      return op_acc(qp, acc_p, self.pos_mask, self.rot_mask)
    elif vel_p:
      return op_vel(qp, vel_p, self.pos_mask, self.rot_mask)
    elif pos_q:
      return op_pos(qp, pos_q, self.pos_mask, self.quat_mask)
    else:
      # no-op
      return qp

  def velocity_projection(self, qp: QP, qp_prev: QP) -> QP:
    """Performs the position based dynamics velocity projection step.

    The velocity and angular velocity must respect the spatial and quaternion
    distance (respectively) between qp and qpold.

    Args:
      qp: The current qp
      qp_prev: The qp at the previous timestep

    Returns:
      qp with velocities pinned to respect the distance traveled since qpold
    """

    @jp.vmap
    def op(qp, qp_prev, pos_mask, rot_mask) -> QP:
      new_rot = qp.rot / jp.norm(qp.rot)
      vel = ((qp.pos - qp_prev.pos) / self.dt) * pos_mask
      dq = math.relative_quat(qp_prev.rot, new_rot)
      ang = 2. * dq[1:] / self.dt
      scale = jp.where(dq[0] >= 0., 1., -1.) * rot_mask  # pytype: disable=wrong-arg-types  # jax-ndarray
      ang = scale * ang * rot_mask
      return QP(pos=qp.pos, vel=vel, rot=new_rot, ang=ang)

    return op(qp, qp_prev, self.pos_mask, self.rot_mask)
