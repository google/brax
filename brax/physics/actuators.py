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
    act_type = cls.__name__.lower()
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
class Angle(Actuator):
  """Applies torque to satisfy a target angle of a joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jnp.array(axis), jnp.array(angle)

    # torque grows as target angle diverges from current angle
    target *= jnp.pi / 180.
    target = jnp.clip(target, self.joint.limit[:, 0], self.joint.limit[:, 1])
    torque = (target - angle) * self.strength
    torque = jnp.sum(jax.vmap(jnp.multiply)(axis, torque), axis=0)

    dang_p = jnp.matmul(self.joint.body_p.inertia, -torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, torque)

    return dang_p, dang_c


@struct.dataclass
class Torque(Actuator):
  """Applies a target torque to a joint."""

  @jax.vmap
  def _apply(self, target: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jnp.array(axis), jnp.array(angle)

    # clip torque if outside joint angle limits
    torque = target * self.strength
    torque = jnp.where(angle < self.joint.limit[:, 0], 0, torque)
    torque = jnp.where(angle > self.joint.limit[:, 1], 0, torque)
    torque = jnp.sum(jax.vmap(jnp.multiply)(axis, torque), axis=0)

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
    dof = len(joint.angle_limit)
    if actuator.name == name:
      if dof == 1:
        return current_idx
      else:
        return tuple(range(current_idx, current_idx + dof))
    current_idx += dof
  raise ValueError(f"Actuator [{name}] does not exist.")
