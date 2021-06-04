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
"""Functionality for brax bodies."""

from flax import struct
import jax
import jax.numpy as jnp

from brax.physics import config_pb2
from brax.physics.base import P, QP, vec_to_np


@struct.dataclass
class Body(object):
  """A body is a solid, non-deformable object with some mass and shape.

  Attributes:
    idx: Index of where body is found in the system.
    inertia: (3, 3) Inverse Inertia matrix represented in body frame.
    mass: Mass of the body.
    active: whether the body is effected by physics calculations
  """
  idx: jnp.ndarray
  inertia: jnp.ndarray
  mass: jnp.ndarray
  active: jnp.ndarray

  @classmethod
  def from_config(cls, config: config_pb2.Config) -> 'Body':
    """Returns Body from a brax config."""
    bodies = []
    for idx, body in enumerate(config.bodies):
      frozen = jnp.sum(
          vec_to_np(body.frozen.position) + vec_to_np(body.frozen.rotation))
      bodies.append(
          cls(
              idx=jnp.array(idx),
              inertia=jnp.linalg.inv(jnp.diag(vec_to_np(body.inertia))),
              mass=jnp.array(body.mass),
              active=jnp.array(jnp.sum(frozen) != 6),
          ))
    return jax.tree_multimap((lambda *args: jnp.stack(args)), *bodies)

  def impulse(self, qp: QP, impulse: jnp.ndarray, pos: jnp.ndarray) -> P:
    """Calculates updates to state information based on an impulse.

    Args:
      qp: State data of the system
      impulse: Impulse vector
      pos: Location of the impulse relative to the body's center of mass

    Returns:
      dP: An impulse to apply to this body
    """
    dvel = impulse / self.mass
    dang = jnp.matmul(self.inertia, jnp.cross(pos - qp.pos, impulse))
    return P(vel=dvel, ang=dang)
