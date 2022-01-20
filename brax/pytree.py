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

"""Pytree decorator for classes."""
import inspect
import jax

# Rolling our own instead of using flax.struct.dataclass because we:
# - would like our own __init__ fns / super() without too much hassle
# - want child classes to inherit from parent classes that have pytree=None
# - aren't worried about immutability / don't like dataclasses.replace()


def register(cls):
  """Registers a class to become a pytree node.

  Treats any class fields as pytree nodes unless they are added to a
  '__pytree_ignore__' class attribute.

  Args:
    cls: the class to register.

  Returns:
    the input class unchanged.
  """

  def tree_flatten(obj):
    pytree_data = []
    pytree_fields = []
    static_data = {}
    for k, v in vars(obj).items():
      static_fields = set()
      for c in inspect.getmro(cls):
        if hasattr(c, '__pytree_ignore__'):
          static_fields.update(cls.__pytree_ignore__)
      if k in static_fields:
        static_data[k] = v
      else:
        pytree_fields.append(k)
        pytree_data.append(v)
    return (pytree_data, (pytree_fields, static_data))

  def tree_unflatten(aux_data, children):
    obj = cls.__new__(cls)
    pytree_fields, static_fields = aux_data
    for k, v in zip(pytree_fields, children):
      setattr(obj, k, v)
    for k, v in static_fields.items():
      setattr(obj, k, v)
    return obj

  jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)

  return cls
