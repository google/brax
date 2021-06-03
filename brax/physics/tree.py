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
"""Functions for manipulating a nested kinematic tree."""

from typing import List, Union, Optional

import dataclasses
import jax
import jax.numpy as jnp
from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import euler_to_quat, vec_to_np


@dataclasses.dataclass
class Link(object):
  """A node within the kinematic tree of a system."""
  body: config_pb2.Body = None
  pos: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(3))
  rot: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array([1., 0., 0., 0.]))
  parent: 'Link' = None
  children: List['Link'] = dataclasses.field(default_factory=list)

  @classmethod
  def from_config(cls,
                  config: config_pb2.Config) -> 'Link':
    """Creates a nested tree and returns its root Link."""
    root = Link()
    body_map = {b.name: b for b in config.bodies}

    for joint in config.joints:
      rot = euler_to_quat(joint.rotation)
      v_rot = jax.vmap(math.rotate, in_axes=[0, None])
      axis = v_rot(jnp.eye(3), rot)

      limits = joint.angle_limit
      if not limits:
        angle = 0.
        child_rotation = jnp.array([1., 0., 0., 0.])
      elif len(limits) == 1:
        angle = (limits[0].min + limits[0].max) * jnp.pi / 360
        child_rotation = math.quat_rot_axis(axis[0], angle)
      elif len(limits) == 2:
        angle = (limits[0].min + limits[0].max) * jnp.pi / 360
        angle_secondary = (limits[1].min + limits[1].max) * jnp.pi / 360

        secondary_rotation = math.quat_rot_axis(axis[0], angle)
        offset_vec = math.rotate(axis[1], secondary_rotation)
        child_rotation = math.quat_rot_axis(offset_vec, angle_secondary)
        child_rotation = math.qmult(child_rotation, secondary_rotation)
      elif len(limits) == 3:
        # TODO: double check angle limit calcs for 3d joints
        angle = (limits[0].min + limits[0].max) * jnp.pi / 360
        angle_secondary = (limits[1].min + limits[1].max) * jnp.pi / 360
        angle_tertiary = (limits[2].min + limits[2].max) * jnp.pi / 360

        secondary_rotation = math.quat_rot_axis(axis[2], angle)
        offset_vec = math.rotate(axis[0], secondary_rotation)
        child_rotation = math.quat_rot_axis(offset_vec, angle_secondary)
        child_rotation = math.qmult(child_rotation, secondary_rotation)
        offset_vec_tert = math.rotate(axis[1], child_rotation)
        final_rot = math.quat_rot_axis(offset_vec_tert, angle_tertiary)
        child_rotation = math.qmult(final_rot, child_rotation)
      else:
        raise AttributeError(
            f'Joint {joint} must have three or fewer angle limits.')

      child_offset = -vec_to_np(joint.child_offset)
      rotated_child_offset = math.rotate(child_offset, child_rotation)
      child_pos = vec_to_np(joint.parent_offset) + rotated_child_offset
      parent = root.rfind(joint.parent)
      child = root.rfind(joint.child)

      if not parent:
        parent = Link(body=body_map[joint.parent], parent=root)
        root.children.append(parent)
      if not child:
        child = Link(body=body_map[joint.child])
      if child.parent:
        child.parent.children.remove(child)
      child = Link(
          body=child.body,
          pos=child_pos,
          rot=child_rotation,
          parent=parent,
          children=child.children)
      parent.children.append(child)
    return root

  def rfind(self, name: str) -> Union['Link', None]:
    """Recursively finds a node with this name."""
    if self.body and self.body.name == name:
      return self
    for c in self.children:
      v = c.rfind(name)
      if v:
        return v
    return None

  def to_world(self,
               offset_pos: Optional[jnp.ndarray] = None,
               offset_rot: Optional[jnp.ndarray] = None) -> 'Link':
    """Converts a tree from local to world coords."""
    if offset_pos is None:
      offset_pos = jnp.array([0., 0., 0.])
    if offset_rot is None:
      offset_rot = jnp.array([1., 0., 0., 0.])
    new_rot = math.qmult(offset_rot, self.rot)
    new_pos = math.rotate(self.pos, offset_rot) + offset_pos

    link = Link(self.body, new_pos, new_rot, self.parent)
    for child in self.children:
      child = child.to_world(new_pos, new_rot)
      child.parent = link
      link.children.append(child)
    return link

  def min_z(self) -> float:
    """Returns the min_z of all bodies in a tree."""
    result = float('inf')

    for col in self.body.colliders:
      if col.HasField('sphere'):
        sphere_pos = math.rotate(vec_to_np(col.position), self.rot)
        min_z = self.pos[2] + sphere_pos[2] - col.sphere.radius
        result = jnp.min(jnp.array([result, min_z]))
      elif col.HasField('capsule'):
        axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
        length = col.capsule.length / 2 - col.capsule.radius
        for end in (-1, 1):
          sphere_pos = vec_to_np(col.position) + end * axis * length
          sphere_pos = math.rotate(sphere_pos, self.rot)
          min_z = self.pos[2] + sphere_pos[2] - col.capsule.radius
          result = jnp.min(jnp.array([result, min_z]))
      elif col.HasField('box'):
        corners = [(i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1)
                   for i in range(8)]
        corners = jnp.array(corners, dtype=jnp.float32)
        for corner in corners:
          corner = corner * vec_to_np(col.box.halfsize)
          corner = math.rotate(corner, euler_to_quat(col.rotation))
          corner = corner + vec_to_np(col.position)
          corner = math.rotate(corner, self.rot) + self.pos
          result = jnp.min(jnp.array([result, corner[2]]))

    for child in self.children:
      result = jnp.min(jnp.array([result, child.min_z()]))

    return result
