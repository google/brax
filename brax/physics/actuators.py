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

import abc
from typing import List, Tuple

from brax.physics import config_pb2
from brax.physics import joints
from brax.physics import pytree
from brax.physics.base import P, QP, take

import jax
import jax.numpy as jnp


class Actuator(abc.ABC):
  """Applies a torque to a joint."""

  def __init__(self, joint: joints.Joint, actuators: List[config_pb2.Actuator],
               act_index: List[Tuple[int, int]]):
    """Creates an actuator that applies torque to a joint given an act array.

    Args:
      joint: (batched) joint for this actuator to act upon
      actuators: list of actuators (all of the same type) to batch together
      act_index: indices from the act array that drive this Actuator
    """
    joint_idx = jnp.array([joint.index[a.joint] for a in actuators])
    self.joint = take(joint, joint_idx)
    self.strength = jnp.array([a.strength for a in actuators])
    self.act_index = jnp.array(act_index)

  @abc.abstractmethod
  def apply_reduced(self, act: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in reduced joint space."""

  def apply(self, qp: QP, act: jnp.ndarray) -> P:
    """Applies torque to a joint.

    Args:
      qp: State data for system
      act: User-defined target for the actuator

    Returns:
      dP: The impulses that drive a joint
    """
    qp_p = take(qp, self.joint.body_p.idx)
    qp_c = take(qp, self.joint.body_c.idx)
    act = take(act, self.act_index)
    dang_p, dang_c = self.apply_reduced(act, qp_p, qp_c)

    # sum together all impulse contributions across parents and children
    body_idx = jnp.concatenate((self.joint.body_p.idx, self.joint.body_c.idx))
    dp_ang = jnp.concatenate((dang_p, dang_c))
    dp_ang = jax.ops.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    return P(vel=jnp.zeros_like(qp.vel), ang=dp_ang)


@pytree.register
class Angle(Actuator):
  """Applies torque to satisfy a target angle of a joint."""

  @jax.vmap
  def apply_reduced(self, act: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jnp.array(axis), jnp.array(angle)

    # torque grows as target angle diverges from current angle
    act *= jnp.pi / 180.
    act = jnp.clip(act, self.joint.limit[:, 0], self.joint.limit[:, 1])
    torque = (act - angle) * self.strength
    torque = jnp.sum(jax.vmap(jnp.multiply)(axis, torque), axis=0)

    dang_p = jnp.matmul(self.joint.body_p.inertia, -torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, torque)

    return dang_p, dang_c


@pytree.register
class Torque(Actuator):
  """Applies a direct torque to a joint."""

  @jax.vmap
  def apply_reduced(self, act: jnp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jnp.array(axis), jnp.array(angle)

    # clip torque if outside joint angle limits
    # * -1. so that positive actuation increases angle between parent and child
    torque = act * self.strength * -1.
    torque = jnp.where(angle < self.joint.limit[:, 0], 0, torque)
    torque = jnp.where(angle > self.joint.limit[:, 1], 0, torque)
    torque = jnp.sum(jax.vmap(jnp.multiply)(axis, torque), axis=0)

    dang_p = jnp.matmul(self.joint.body_p.inertia, torque)
    dang_c = jnp.matmul(self.joint.body_c.inertia, -torque)

    return dang_p, dang_c


def get(config: config_pb2.Config,
        all_joints: List[joints.Joint]) -> List[Actuator]:
  """Creates all actuators given a config and joints."""
  actuators = {}
  current_index = 0
  for actuator in config.actuators:
    joint = [j for j in all_joints if actuator.joint in j.index]
    if not joint:
      raise RuntimeError(f'joint not found: {actuator.joint}')
    joint = joint[0]
    act_index = tuple(range(current_index, current_index + joint.dof))
    current_index += joint.dof
    key = (actuator.WhichOneof('type'), joint.dof, joint)
    if key not in actuators:
      actuators[key] = []
    actuators[key].append((actuator, act_index))

  # ensure stable order for actuator application: actuator type, then dof
  actuators = sorted(actuators.items(), key=lambda kv: kv[0])
  ret = []
  for (actuator, _, joint), act_config_index in actuators:
    act_config = [c for c, _ in act_config_index]
    act_index = [a for _, a in act_config_index]
    if actuator == 'torque':
      ret.append(Torque(joint, act_config, act_index))
    elif actuator == 'angle':
      ret.append(Angle(joint, act_config, act_index))
    else:
      raise RuntimeError(f'unknown actuator type: {actuator}')
  return ret
