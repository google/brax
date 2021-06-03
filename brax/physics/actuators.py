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
"""Actuation of joints."""

from typing import Tuple

from flax import struct
import jax
import jax.numpy as jnp

from brax.physics import config_pb2
from brax.physics import joints
from brax.physics.base import P, QP, take


@struct.dataclass
class Actuator:
  """Applies a torque to a joint."""
  joint: joints.Joint
  strength: jnp.ndarray
  act_idx: jnp.ndarray
  config: config_pb2.Config = struct.field(pytree_node=False)

  @classmethod
  def from_config(cls, config: config_pb2.Config,
                  joint: joints.Joint) -> "Actuator":
    """Creates an actuator from a config."""
    joint_type = type(joint).__name__.lower()
    act_type = cls.__name__.lower()[:-2]  # chop off 1d/2d
    joints_filtered = [
        j for j in config.joints if joints.type_to_dof[joint_type] ==
        joints.lim_to_dof[len(j.angle_limit)]
    ]
    joint_map = {j.name: i for i, j in enumerate(joints_filtered)}
    actuators = [
        a for a in config.actuators
        if a.HasField(act_type) and a.joint in joint_map
    ]
    if not actuators:
      return cls(*[None] * 4)
    joint_idx = jnp.array([joint_map[a.joint] for a in actuators])
    joint = take(joint, joint_idx)  # ensure joints are in the right order
    strength = jnp.array([a.strength for a in actuators])
    act_idx = jnp.array([_act_idx(config, a.name) for a in actuators])
    return cls(joint, strength, act_idx, config)

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    raise NotImplementedError()  # child must override

  def apply(self, qp: QP, target: jnp.ndarray) -> P:
    """Applies torque to satisfy a target angle on a revolute joint.

    Args:
      qp: State data for system
      target: User-defined target angle for the actuator in degrees

    Returns:
      dP: The impulses that drive a parent and child towards a target angle
    """
    if not self.config:
      return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

    qp_p = take(qp, self.joint.body_p.idx)
    qp_c = take(qp, self.joint.body_c.idx)
    target = take(target, self.act_idx)
    dang_p, dang_c = self._apply(target, qp_p, qp_c)

    # sum together all impulse contributions across parents and children
    body_idx = jnp.concatenate((self.joint.body_p.idx, self.joint.body_c.idx))
    dp_ang = jnp.concatenate((dang_p, dang_c))
    dp_ang = jax.ops.segment_sum(dp_ang, body_idx, len(self.config.bodies))

    return P(vel=jnp.zeros_like(qp.vel), ang=dp_ang)


@struct.dataclass
class Angle1D(Actuator):
  """Applies torque to satisfy a target angle of a revolute joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    (axis,), (angle,) = self.joint.axis_angle(qp_p, qp_c)

    # torque grows as target angle diverges from current angle
    target *= jnp.pi / 180.
    target = jnp.clip(target, self.joint.limit[0][0], self.joint.limit[0][1])
    torque = (target - angle) * self.strength

    dang_p = jnp.matmul(self.joint.body_p.inertia, -axis * torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, axis * torque)

    return dang_p, dang_c


@struct.dataclass
class Torque1D(Actuator):
  """Applies a target torque to a revolute joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    (axis,), (angle,) = self.joint.axis_angle(qp_p, qp_c)

    # clip torque if outside joint angle limits
    torque = target * self.strength
    torque = jnp.where(angle < self.joint.limit[0][0], 0, torque)
    torque = jnp.where(angle > self.joint.limit[0][1], 0, torque)

    dang_p = jnp.matmul(self.joint.body_p.inertia, -axis * torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, axis * torque)

    return dang_p, dang_c


@struct.dataclass
class Angle2D(Actuator):
  """Applies torque to satisfy a target angle of a revolute joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis_1, axis_2 = axis
    angle_1, angle_2 = angle

    target *= jnp.pi / 180.
    target = jnp.clip(target, self.joint.limit[:, 0], self.joint.limit[:, 1])
    torque_1 = -1. * (target[0] - angle_1) * self.strength
    torque_2 = -1. * (target[1] - angle_2) * self.strength

    torque = axis_1 * torque_1 + axis_2 * torque_2
    dang_p = jnp.matmul(self.joint.body_p.inertia, torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, -torque)

    return dang_p, dang_c


@struct.dataclass
class Torque2D(Actuator):
  """Applies torque to satisfy a target angle of a revolute joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis_1, axis_2 = axis
    angle_1, angle_2 = angle

    torque_1, torque_2 = target * self.strength
    limit_1, limit_2 = self.joint.limit
    torque_1 = jnp.where(angle_1 < limit_1[0], 0, torque_1)
    torque_1 = jnp.where(angle_1 > limit_1[1], 0, torque_1)
    torque_2 = jnp.where(angle_2 < limit_2[0], 0, torque_2)
    torque_2 = jnp.where(angle_2 > limit_2[1], 0, torque_2)

    torque = axis_1 * torque_1 + axis_2 * torque_2
    dang_p = jnp.matmul(self.joint.body_p.inertia, torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, -torque)

    return dang_p, dang_c


@struct.dataclass
class Torque3D(Actuator):
  """Applies torque along 3 axes."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis_1, axis_2, axis_3 = axis
    angle_1, angle_2, angle_3 = angle
    torque_1, torque_2, torque_3 = target * self.strength
    limit_1, limit_2, limit_3 = self.joint.limit
    torque_1 = jnp.where(angle_1 < limit_1[0], 0, torque_1)
    torque_1 = jnp.where(angle_1 > limit_1[1], 0, torque_1)
    torque_2 = jnp.where(angle_2 < limit_2[0], 0, torque_2)
    torque_2 = jnp.where(angle_2 > limit_2[1], 0, torque_2)
    torque_3 = jnp.where(angle_3 < limit_3[0], 0, torque_3)
    torque_3 = jnp.where(angle_3 > limit_3[1], 0, torque_3)

    torque = axis_1 * torque_1 + axis_2 * torque_2 + axis_3 * torque_3
    dang_p = jnp.matmul(self.joint.body_p.inertia, torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, -torque)
    return dang_p, dang_c


def _find_joint(config, name):
  for j in config.joints:
    if j.name == name:
      return j
  raise ValueError(f"Joint [{name}] does not exist.")


def _act_idx(config, name):
  """Returns the indices that an actuator occupies in an array."""
  current_idx = 0
  for actuator in config.actuators:
    joint = _find_joint(config, actuator.joint)
    if len(joint.angle_limit) == 1:
      if actuator.name == name:
        return current_idx
      current_idx += 1
    elif len(joint.angle_limit) == 2:
      if actuator.name == name:
        return (current_idx, current_idx + 1)
      current_idx += 2
    elif len(joint.angle_limit) == 3:
      if actuator.name == name:
        return (current_idx, current_idx + 1, current_idx + 2)
      current_idx += 3
    else:
      if actuator.name == name:
        return None
  raise ValueError(f"Actuator [{name}] does not exist.")
