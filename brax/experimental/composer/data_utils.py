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

"""Data processing utilities."""
# pylint:disable=g-importing-member
# pylint:disable=g-doc-args
# pylint:disable=g-doc-return-or-yield
from collections import OrderedDict as odict
from typing import Any, Union, Tuple, Dict
from jax import numpy as jnp
import numpy as np


def get_indices(shape: Dict[str, Any]):
  """Check if shape is valid and return indices."""
  s = shape
  indices = s.get('indices', None)
  if indices is None:
    try:
      indices = tuple(range(s['start'], s['end']))
    except KeyError:
      raise NotImplementedError(s)
  assert np.prod(s['shape']) == s['size'], s
  assert len(indices) == s['size'], s
  return indices


def flatten_array(arr: jnp.ndarray, shape: Dict[str, Any]):
  """Flatten array."""
  s = shape
  assert arr.shape[-len(s['shape']
                       ):] == s['shape'], f'{arr.shape} incompat {s["shape"]}'
  arr = arr.reshape(arr.shape[:-len(s['shape'])] + (s['size'],))
  return arr


def check_array_shapes(array_dict: Union[Dict[str, jnp.ndarray], jnp.ndarray],
                       array_shapes: Dict[str, Dict[str, Any]]):
  """Check array_dict compatible with array_shapes."""
  leading_dims = None
  for k, s in array_shapes.items():
    assert k in array_dict, f'{k} not in {list(array_dict.keys())}'
    arr = array_dict[k]
    if leading_dims is None:
      leading_dims = arr.shape[:-len(s['shape'])]
    else:
      assert leading_dims == arr.shape[:-len(
          s['shape'])], f'{k}: {leading_dims} incompat {arr.shape}'
    assert arr.shape[-len(s['shape']
                         ):] == s['shape'], f'{k}: {arr.shape} != {s["shape"]}'


def get_array_shapes(array_dict: Union[Dict[str, jnp.ndarray], jnp.ndarray],
                     batch_shape: Tuple[int] = ()):
  """Get array dict shape.

  get_array_shape() returns shape info in the form:
    dict(key1=dict(shape=(10,), start=40, end=50), ...)
  """
  if isinstance(array_dict, jnp.ndarray):
    return array_dict.shape
  array_shapes = odict()
  i = 0
  for k, v in array_dict.items():
    v_shape = v.shape[len(batch_shape):]
    size = np.prod(v_shape)
    array_shapes[k] = dict(shape=v_shape, size=size, start=i, end=i + size)
    i += size
  return array_shapes


def fill_array(array_dict: Dict[str, jnp.ndarray],
               array: jnp.ndarray,
               array_shapes: Dict[str, Dict[str, Any]],
               assert_once: bool = False) -> jnp.ndarray:
  """Fill an array with array_dict."""
  check_array_shapes(array_dict, array_shapes)
  if assert_once:
    assert array.shape[-1] == sum([s['size'] for s in array_shapes.values()
                                  ]), f'{array.shape} incompat {array_shapes}'
    assigned = jnp.zeros(array.shape[-1])
  for k in array_shapes.keys():
    arr = array_dict[k]
    s = array_shapes[k]
    indices = get_indices(s)
    arr = flatten_array(arr, s)
    array = array.at[..., indices].set(arr)
    if assert_once:
      assigned = assigned.at[..., indices].add(1)
  if assert_once:
    assert jnp.all(jnp.equal(assigned, 1)), assigned
  return array


def concat_array(array_dict: Dict[str, jnp.ndarray],
                 array_shapes: Dict[str, Dict[str, Any]]) -> jnp.ndarray:
  """Concatenate array dictionary to a vector."""
  check_array_shapes(array_dict, array_shapes)
  return jnp.concatenate([
      arr.reshape(arr.shape[:-len(s['shape'])] + (s['size'],))
      for arr, s in zip(array_dict.values(), array_shapes.values())
  ],
                         axis=-1)


def split_array(
    array: jnp.ndarray,
    array_shapes: Dict[str, Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
  """Split array vector to a dictionary."""
  array_leading_dims = array.shape[:-1]
  array_dict = odict()
  for k, s in array_shapes.items():
    indices = get_indices(s)
    array_dict[k] = array[..., indices].reshape(array_leading_dims + s['shape'])
  return array_dict
