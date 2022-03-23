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
"""Functionality for brax bodies."""

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import config_pb2
from brax.physics.base import P, QP, vec_to_arr


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
  __pytree_ignore__ = ('index',)

  def __init__(self, config: config_pb2.Config):
    self.idx = jp.arange(0, len(config.bodies))
    self.inertia = 1. / jp.array([vec_to_arr(b.inertia) for b in config.bodies])
    self.mass = jp.array([b.mass for b in config.bodies])
    self.active = jp.array(
        [0.0 if b.frozen.all else 1.0 for b in config.bodies])
    self.index = {b.name: i for i, b in enumerate(config.bodies)}

  def impulse(self, qp: QP, impulse: jp.ndarray, pos: jp.ndarray) -> P:
    """Calculates updates to state information based on an impulse.

    Args:
      qp: State data of the system
      impulse: Impulse vector
      pos: Location of the impulse relative to the body's center of mass

    Returns:
      dP: An impulse to apply to this body
    """
    dvel = impulse / self.mass
    dang = self.inertia * jp.cross(pos - qp.pos, impulse)
    return P(vel=dvel, ang=dang)


def min_z(qp: QP, body: config_pb2.Body) -> float:
  """Returns the lowest z of all the colliders in a body."""
  if not body.colliders:
    return 0.0

  result = float('inf')

  for col in body.colliders:
    if col.HasField('sphere'):
      sphere_pos = math.rotate(vec_to_arr(col.position), qp.rot)
      z = qp.pos[2] + sphere_pos[2] - col.sphere.radius
      result = jp.amin(jp.array([result, z]))
    elif col.HasField('capsule'):
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      axis = math.rotate(jp.array([0., 0., 1.]), rot)
      length = col.capsule.length / 2 - col.capsule.radius
      for end in (-1, 1):
        sphere_pos = vec_to_arr(col.position) + end * axis * length
        sphere_pos = math.rotate(sphere_pos, qp.rot)
        z = qp.pos[2] + sphere_pos[2] - col.capsule.radius
        result = jp.amin(jp.array([result, z]))
    elif col.HasField('box'):
      corners = [(i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1)
                 for i in range(8)]
      corners = jp.array(corners, dtype=float)
      for corner in corners:
        corner = corner * vec_to_arr(col.box.halfsize)
        rot = math.euler_to_quat(vec_to_arr(col.rotation))
        corner = math.rotate(corner, rot)
        corner = corner + vec_to_arr(col.position)
        corner = math.rotate(corner, qp.rot) + qp.pos
        result = jp.amin(jp.array([result, corner[2]]))
    else:
      # ignore planes and other stuff
      result = jp.amin(jp.array([result, 0.0]))

  return result
