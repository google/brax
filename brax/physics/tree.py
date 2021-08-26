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

"""Just some tree functions."""

from typing import List, Optional

import dataclasses
import brax

@dataclasses.dataclass
class Node(object):
  """A node within the kinematic tree of a system."""
  name: str = ''
  parent: Optional['Node'] = None
  children: List['Node'] = dataclasses.field(default_factory=list)

  def add(self, node: 'Node'):
    if node.parent:
      node.parent.children.remove(node)
    node.parent = self
    self.children.append(node)

  def find(self, name: str) -> Optional['Node']:
    """Recursively finds a node with parent with this name."""
    if self.name == name:
      return self
    for c in self.children:
      v = c.find(name)
      if v:
        return v
    return None

  def depth_first(self):
    """Iterates through a tree depth first."""
    for c in self.children:
      yield c
      yield from c.depth_first()

  @classmethod
  def from_config(cls, config: brax.Config) -> 'Node':
    root = Node()

    for joint in config.joints:
      parent = root.find(joint.parent)
      child = root.find(joint.child)
      if not parent:
        parent = Node(name=joint.parent)
        root.add(parent)
      if not child:
        child = Node(name=joint.child)
      parent.add(child)

    for body in config.bodies:
      if not root.find(body.name):
        root.add(Node(name=body.name))

    return root
