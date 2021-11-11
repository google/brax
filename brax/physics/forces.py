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
"""Force applied to bodies."""

from typing import List, Tuple

from jax import numpy as jnp

from brax import jumpy as jp
from brax import pytree
from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics.base import P, QP, vec_to_arr

@pytree.register
class Thruster:
  """Applies an external force or torque to a body."""

  def __init__(self, forces: List[config_pb2.Force], body: bodies.Body,
               force_index: List[Tuple[int, int, int]],
               force_mask: List[Tuple[bool, bool, bool]],
               torque_index: List[Tuple[int, int, int]],
               torque_mask: List[Tuple[bool, bool, bool]]):
    """Creates an actuator that applies torque to a joint given an act array.

    Args:
      forces: list of forces (all of the same type) to batch together
      body: (batched) bodfies for this force to act upon
      force_index: indices from the act array that drive this force
      toque_index: indices from the act array that drive this torque
    """
    self.body = jp.take(body, [body.index[f.body] for f in forces])
    self.strength = jp.array([f.strength for f in forces])
    self.force_index = jp.array(force_index)
    self.force_mask = jp.array(force_mask)
    self.torque_index = jp.array(torque_index)
    self.torque_mask = jp.array(torque_mask)

  def force_apply_reduced(self, force: jp.ndarray, mask: jp.ndarray) -> jp.ndarray:
    dvel = force * self.strength / self.body.mass * mask
    return dvel

  def torque_apply_reduced(self, torque: jp.ndarray, mask: jp.ndarray) -> jp.ndarray:
    dang = jp.matmul(self.body.inertia, torque * self.strength) * mask
    return dang

  def apply(self, qp: QP, action_data: jp.ndarray) -> P:
    """Applies a force to a body.

    Args:
      qp: State data for system
      action_data: Data specifying the actions applied to system.

    Returns:
      dP: The impulses that result from apply a force to the body.
    """

    force_data = jp.take(action_data, self.force_index)
    force_dvel = jp.vmap(type(self).force_apply_reduced)(self, force_data, self.force_mask)

    torque_data = jp.take(action_data, self.torque_index)
    torque_dang = jp.vmap(type(self).torque_apply_reduced)(self, torque_data, self.torque_mask)

    # sum together all impulse contributions to all parts effected by forces and torques
    dvel = jp.segment_sum(force_dvel, self.body.idx, qp.pos.shape[0])
    dang = jp.segment_sum(torque_dang, self.body.idx, qp.pos.shape[0])

    return P(vel=dvel, ang=dang)


def get(config: config_pb2.Config,
        body: bodies.Body) -> List[Thruster]:
  """Creates all forces given a config and actuators."""
  # by convention, force indices are after actuator indices
  # get the next available index after actuator indices
  dofs = {j.name: len(j.angle_limit) for j in config.joints}
  current_index = sum([dofs[a.joint] for a in config.actuators])

  force_indices = []
  force_mask = []
  torque_indices = []
  torque_mask = []
  for force in config.forces:
    pos_mask = jnp.logical_not(vec_to_arr(force.frozen.position))
    force_dof = sum(pos_mask)
    # the index array needs to always have three elements to allow
    # vectorization. the other values all index the first action but 
    # the mask gives them no weight
    force_act_index = tuple(range(current_index, current_index + force_dof))
    force_act_index = pos_mask.astype(int).at[pos_mask].set(force_act_index)
    force_indices.append(force_act_index)
    force_mask.append(pos_mask)
    current_index += force_dof

    rot_mask = jnp.logical_not(vec_to_arr(force.frozen.rotation))
    torque_dof = sum(rot_mask)
    torque_act_index = tuple(range(current_index, current_index + torque_dof))
    torque_act_index = rot_mask.astype(int).at[rot_mask].set(torque_act_index)
    torque_indices.append(torque_act_index)
    torque_mask.append(rot_mask)
    current_index += torque_dof

  if force_indices or torque_indices:
    return [Thruster(list(config.forces), body, force_indices, force_mask, torque_indices, torque_mask)]
  else:
    return []
