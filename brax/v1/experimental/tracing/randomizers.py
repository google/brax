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

"""Convenience functions for constructing domain randomizers."""

from typing import List, Optional

from brax.v1.envs.env import Env
import brax.v1.jumpy as jp
import jax

# Here are examples of how to randomize across different config properties.


def friction_randomizer(env: Env, friction_range: jp.ndarray):
  """Constructs tree and in_axes objects for a friction domain randomizer.

  Args:
    env: Environment to randomize over
    friction_range: (m,) friction values to use for domain randomization

  Returns:
    Tuple of a pytree containing the randomized values packed into a tree
    structure parseable by the TracedConfig class, as well as in_axes
    describing which entries of that pytree are vectorized.
  """

  custom_tree = {'bodies': []}
  for b in env.sys.config.bodies:
    custom_tree['bodies'].append({'colliders': []})
    for _ in b.colliders:
      custom_tree['bodies'][-1]['colliders'].append({
          'material': {
              'friction': friction_range
          }
      })

    in_axes = (jax.tree.map(lambda x: 0
                            if hasattr(x, 'shape') else None, custom_tree),)
  return custom_tree, in_axes


def joint_randomizer(env: Env,
                     joint_offsets: jp.ndarray,
                     joint_offsets_key: Optional[List[str]] = None):
  """Constructs tree and in_axes objects for a joint socket randomizer.

  Adds an offset to any joints that match a key appearing in joint_key.  If
  no joint_key, then does nothing.

  Args:
    env: Environment to randomize over
    joint_offsets: (m,3) offset vectors for joints
    joint_offsets_key: (k,) list of string keys for marking joints to be shifted

  Returns:
    Tuple of a pytree containing the randomized values packed into a tree
    structure parseable by the TracedConfig class, as well as in_axes
    describing which entries of that pytree are vectorized.
  """

  custom_tree = {'joints': []}

  for j in env.sys.config.joints:

    if any([key in j.name for key in joint_offsets_key]):
      custom_tree['joints'].append({
          'parent_offset': {
              'x': joint_offsets[:, 0] + j.parent_offset.x,
              'y': joint_offsets[:, 1] + j.parent_offset.y,
              'z': joint_offsets[:, 2] + j.parent_offset.z
          },
      })
    else:
      custom_tree['joints'].append({
          'parent_offset': {
              'x': j.parent_offset.x,
              'y': j.parent_offset.y,
              'z': j.parent_offset.z
          },
      })

  in_axes = (jax.tree.map(lambda x: 0
                          if hasattr(x, 'shape') else None, custom_tree),)

  return custom_tree, in_axes
