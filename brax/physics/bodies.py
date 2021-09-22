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

from brax.physics import config_pb2
from brax.physics import math
from brax.physics import pytree
from brax.physics.base import P, QP, euler_to_quat, vec_to_np

import jax.numpy as jnp


@pytree.register
class Body:
  """A body is a solid, non-deformable object with some mass and shape.

  Attributes:
    idx: Index of where body is found in the system.
    inertia: (3, 3) Inverse Inertia matrix represented in body frame.
    mass: Mass of the body.
    active: whether the body is effected by physics calculations
    index: name->index dict for looking up body names
  """
  __pytree_ignore__ = ('index', 'count')

  def __init__(self, config: config_pb2.Body):
    self.idx = jnp.arange(len(config.bodies))
    self.inertia = jnp.array(
        [jnp.linalg.inv(jnp.diag(vec_to_np(b.inertia))) for b in config.bodies])
    self.mass = jnp.array([b.mass for b in config.bodies])
    self.active = jnp.array(
        [0.0 if b.frozen.all else 1.0 for b in config.bodies])
    self.index = {b.name: i for i, b in enumerate(config.bodies)}

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


def min_z(qp: QP, body: config_pb2.Body) -> float:
  """Returns the lowest z of all the colliders in a body."""
  result = float('inf')

  for col in body.colliders:
    if col.HasField('sphere'):
      sphere_pos = math.rotate(vec_to_np(col.position), qp.rot)
      z = qp.pos[2] + sphere_pos[2] - col.sphere.radius
      result = jnp.min(jnp.array([result, z]))
    elif col.HasField('capsule'):
      axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
      length = col.capsule.length / 2 - col.capsule.radius
      for end in (-1, 1):
        sphere_pos = vec_to_np(col.position) + end * axis * length
        sphere_pos = math.rotate(sphere_pos, qp.rot)
        z = qp.pos[2] + sphere_pos[2] - col.capsule.radius
        result = jnp.min(jnp.array([result, z]))
    elif col.HasField('box'):
      corners = [(i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1)
                 for i in range(8)]
      corners = jnp.array(corners, dtype=jnp.float32)
      for corner in corners:
        corner = corner * vec_to_np(col.box.halfsize)
        corner = math.rotate(corner, euler_to_quat(col.rotation))
        corner = corner + vec_to_np(col.position)
        corner = math.rotate(corner, qp.rot) + qp.pos
        result = jnp.min(jnp.array([result, corner[2]]))
    else:
      # ignore planes and other stuff
      result = jnp.min(jnp.array([result, 0.0]))

  return result
