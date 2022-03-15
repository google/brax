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

# pylint:disable=g-multiple-import
"""Actuation of joints."""

import abc
from typing import List, Tuple

from brax import jumpy as jp
from brax import pytree
from brax.physics import config_pb2
from brax.physics import joints
from brax.physics.base import P, QP


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
    self.joint = jp.take(joint, [joint.index[a.joint] for a in actuators])
    self.strength = jp.array([a.strength for a in actuators])
    self.act_index = jp.array(act_index)

  @abc.abstractmethod
  def apply_reduced(self, act: jp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in reduced joint space."""

  def apply(self, qp: QP, act: jp.ndarray) -> P:
    """Applies torque to a joint.

    Args:
      qp: State data for system
      act: User-defined target for the actuator

    Returns:
      dP: The impulses that drive a joint
    """
    qp_p = jp.take(qp, self.joint.body_p.idx)
    qp_c = jp.take(qp, self.joint.body_c.idx)
    act = jp.take(act, self.act_index)
    dang_p, dang_c = jp.vmap(type(self).apply_reduced)(self, act, qp_p, qp_c)

    # sum together all impulse contributions across parents and children
    body_idx = jp.concatenate((self.joint.body_p.idx, self.joint.body_c.idx))
    dp_ang = jp.concatenate((dang_p, dang_c))
    dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    return P(vel=jp.zeros_like(qp.vel), ang=dp_ang)


@pytree.register
class Angle(Actuator):
  """Applies torque to satisfy a target angle of a joint."""

  def apply_reduced(self, act: jp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jp.array(axis), jp.array(angle)

    limit_min, limit_max = self.joint.limit[:, 0], self.joint.limit[:, 1]
    act = jp.clip(act * jp.pi / 180, limit_min, limit_max)
    torque = (act - angle) * self.strength
    torque = jp.sum(jp.vmap(jp.multiply)(axis, torque), axis=0)

    dang_p = -self.joint.body_p.inertia * torque
    dang_c = self.joint.body_c.inertia * torque

    return dang_p, dang_c


@pytree.register
class Torque(Actuator):
  """Applies a direct torque to a joint."""

  def apply_reduced(self, act: jp.ndarray, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    axis, angle = self.joint.axis_angle(qp_p, qp_c)
    axis, angle = jp.array(axis), jp.array(angle)

    # clip torque if outside joint angle limits
    # * -1. so that positive actuation increases angle between parent and child
    torque = act * self.strength * -1.
    torque = jp.where(angle < self.joint.limit[:, 0], 0, torque)
    torque = jp.where(angle > self.joint.limit[:, 1], 0, torque)
    torque = jp.sum(jp.vmap(jp.multiply)(axis, torque), axis=0)

    dang_p = self.joint.body_p.inertia * torque
    dang_c = -self.joint.body_c.inertia * torque

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
