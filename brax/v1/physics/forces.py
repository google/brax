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
"""Force applied to bodies."""

from typing import List, Tuple, Union

from brax.v1 import jumpy as jp
from brax.v1 import pytree
from brax.v1.physics import bodies
from brax.v1.physics import config_pb2
from brax.v1.physics.base import P, QP


@pytree.register
class Thruster:
  """Applies a force to a body."""

  def __init__(self, forces: List[config_pb2.Force], body: bodies.Body,
               act_index: List[Tuple[int, int]]):
    """Thruster applies linear force to a body given a 3d act array.

    Args:
      forces: list of forces (all of the same type) to batch together
      body: (batched) bodfies for this force to act upon
      act_index: indices from the act array that drive this force
    """
    self.body = jp.take(body, [body.index[f.body] for f in forces])
    self.strength = jp.array([f.strength for f in forces])
    self.act_index = jp.array(act_index)

  def apply_reduced(self, force: jp.ndarray) -> jp.ndarray:
    dvel = force * self.strength / self.body.mass
    return dvel, jp.zeros_like(dvel)  # pytype: disable=bad-return-type  # jax-ndarray

  def apply(self, qp: QP, force_data: jp.ndarray) -> P:
    """Applies a force to a body.

    Args:
      qp: State data for system
      force_data: Data specifying the force to apply to a body.

    Returns:
      dP: The impulses that result from apply a force to the body.
    """

    force_data = jp.take(force_data, self.act_index)
    dvel, dang = jp.vmap(type(self).apply_reduced)(self, force_data)

    # sum together all impulse contributions to all parts effected by forces
    dvel = jp.segment_sum(dvel, self.body.idx, qp.pos.shape[0])
    dang = jp.segment_sum(dang, self.body.idx, qp.pos.shape[0])

    return P(vel=dvel, ang=dang)


@pytree.register
class Twister:
  """Applies a torque to a body."""

  def __init__(self, forces: List[config_pb2.Force], body: bodies.Body,
               act_index: List[Tuple[int, int]]):
    """Twister applies torque to a single body.

    Args:
      forces: list of forces (all of the same type) to batch together
      body: (batched) bodfies for this force to act upon
      act_index: indices from the act array that drive this force
    """
    self.body = jp.take(body, [body.index[f.body] for f in forces])
    self.strength = jp.array([f.strength for f in forces])
    self.act_index = jp.array(act_index)

  def apply_reduced(self, torque: jp.ndarray) -> jp.ndarray:
    dang = torque * self.strength / self.body.mass
    return jp.zeros_like(dang), dang  # pytype: disable=bad-return-type  # jax-ndarray

  def apply(self, qp: QP, force_data: jp.ndarray) -> P:
    """Applies a force to a body.

    Args:
      qp: State data for system
      force_data: Data specifying the force to apply to a body.

    Returns:
      dP: The impulses that result from apply a force to the body.
    """
    force_data = jp.take(force_data, self.act_index)
    dvel, dang = jp.vmap(type(self).apply_reduced)(self, force_data)

    # sum together all impulse contributions to all parts effected by forces
    dvel = jp.segment_sum(dvel, self.body.idx, qp.pos.shape[0])
    dang = jp.segment_sum(dang, self.body.idx, qp.pos.shape[0])

    return P(vel=dvel, ang=dang)


def get(config: config_pb2.Config,
        body: bodies.Body) -> List[Union[Thruster, Twister]]:
  """Creates all forces given a config and actuators."""
  # by convention, force indices are after actuator indices
  # get the next available index after actuator indices
  dofs = {j.name: len(j.angle_limit) for j in config.joints}
  current_index = sum([dofs[a.joint] for a in config.actuators])

  thrusters, thruster_indices = [], []
  twisters, twister_indices = [], []
  for f in config.forces:
    act_index = tuple(range(current_index, current_index + 3))
    current_index += 3
    if f.WhichOneof('type') == 'thruster':
      thrusters.append(f)
      thruster_indices.append(act_index)
    elif f.WhichOneof('type') == 'twister':
      twisters.append(f)
      twister_indices.append(act_index)
    else:
      raise ValueError(f'unknown force type: {f.WhichOneof("type")}')

  forces = []
  if thrusters:
    forces.append(Thruster(thrusters, body, thruster_indices))
  if twisters:
    forces.append(Twister(twisters, body, twister_indices))

  return forces
