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

"""Wrapper class for facilitating jax transformations on static config data."""

from collections.abc import Iterable
from typing import Any, Dict

from brax.v1 import pytree


@pytree.register
class TracedConfig:
  """A wrapper around the config that faciltates jax tracing of static data."""

  __pytree_ignore__ = ('msg',)

  def __init__(self, msg: Any, custom_tree: Dict[str, Any]):
    self.msg = msg
    self.custom_tree = custom_tree

  def __getattr__(self, name: str):

    base = self.msg.__getattribute__(name)  # pytype:disable=attribute-error

    if name in self.custom_tree:
      if isinstance(base, Iterable) and not isinstance(base, str):
        list_msg = []
        for o, b in zip(self.custom_tree[name], base):
          list_msg.append(TracedConfig(b, custom_tree=o))
        return list_msg
      elif not isinstance(self.custom_tree[name], dict):
        return self.custom_tree[name]
      else:
        return TracedConfig(base, self.custom_tree[name])

    return base
