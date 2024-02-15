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

"""Utility functions to manipulate configuration objects.

E.g. configuration objects:
    dict(a=[2,3,4], b=1) --> sample: dict(a=2,b=1)
    [dict(a=[2,3,4], b=1), dict(a=1, b=[2,3])] --> sample: dict(a=1,b=2)

The first level can be a list (in which case, each item is expanded separately),
or a dictionary.
After the first level, all lists are expanded while the rest of structures
specified by dictionaries are kept the same.
"""
import copy
from typing import Dict, Tuple, Any
import numpy as np


def filter_configuration(config, root=True, include_keys=()):
  """Filter configuration (non-recursive)."""
  if isinstance(config, dict):
    ret = {k: v for k, v in config.items() if k in include_keys}
  elif root and isinstance(config, list):
    ret = [
        filter_configuration(c, root=False, include_keys=include_keys)
        for c in config
    ]
  else:
    ret = config
  return ret


def get_compressed_name_from_keys(config: Dict[str, Any],
                                  keys: Tuple[Tuple[str]],
                                  allow_missing: bool = True):
  """Generate a compressed name from keys wrt config."""
  assert not isinstance(config, list), config
  name = ''
  used_pre_keys_strs = []
  for pre_keys in keys:
    v = config
    pre_keys_str = ''
    missing = False
    for k in pre_keys:
      if allow_missing and k not in v:
        missing = True
        break
      v = v[k]
      k_str = ''.join([s[0] for s in k.split('_')])  # learning_rate -> lr
      pre_keys_str = f'{pre_keys_str}.{k_str}' if pre_keys_str else k_str
    if missing:
      continue
    if pre_keys_str in used_pre_keys_strs:
      i = 0
      while f'{pre_keys_str}{i}' in used_pre_keys_strs and i < 10:
        i += 1
      assert i < 10, 'too many conflicts'
      pre_keys_str = f'{pre_keys_str}{i}'
    used_pre_keys_strs.append(pre_keys_str)
    if isinstance(v, bool):
      v = str(v)[0]  # True/False -> 'T', 'F'
    elif v is None:
      v = str(v)[0]  # None -> 'N'
    else:
      v = str(v)
    pre_keys_str += '_' + v
    name = f'{name}__{pre_keys_str}' if name else pre_keys_str
  return name


def list_keys_to_expand(config, root=True, pre_keys=()):
  """List the keys corresponding to List or callable."""
  if isinstance(config, dict):
    keys = ()
    for k, v in sorted(config.items()):
      keys += list_keys_to_expand(v, root=False, pre_keys=pre_keys + (k,))
    return keys
  elif (not root and isinstance(config, list)) or callable(config):
    assert pre_keys
    return (pre_keys,)
  elif root and isinstance(config, list):
    return tuple(
        list_keys_to_expand(v, root=False, pre_keys=pre_keys) for v in config)
  else:
    return ()


def count_configuration(config, root=True, num_samples_per_dist=1):
  """Recursively count configuration."""
  count = 1
  if isinstance(config, dict):
    for _, v in sorted(config.items()):
      count *= count_configuration(
          v, root=False, num_samples_per_dist=num_samples_per_dist)
  elif callable(config):
    assert num_samples_per_dist > 0, ('callable not allowed in config with '
                                      'num_samples_per_dist < 1')
    count *= num_samples_per_dist
  elif isinstance(config, list):
    if root:
      count = ()
      for c in config:
        count += (count_configuration(
            c, root=False, num_samples_per_dist=num_samples_per_dist),)
    else:
      count *= len(config)
  return count


def index_configuration(config, index=0, root=True, count=(), return_copy=True):
  """Configuration."""
  if isinstance(config, dict):
    c = {}
    for k, v in sorted(config.items()):
      c_, index = index_configuration(v, index=index, root=False)
      c[k] = c_
    if root and return_copy:
      c = copy.deepcopy(c)
    return c, index
  elif callable(config):
    raise NotImplementedError(
        'callable not allowed; call sample_configuration_dist first')
  elif isinstance(config, list):
    if root:
      i = 0
      index_ = index
      while i < len(config) and index_ >= count[i]:
        index_ -= count[i]
        i += 1
      assert i < len(config) and index_ >= 0 and index_ <= count[
          i], f'invalid index={index} wrt count={count}'
      c, i = index_configuration(config[i], index=index_, root=False)
      if return_copy:
        c = copy.deepcopy(c)
      return c, i
    else:
      i = index % len(config)
      index_ = int(index / len(config))
      return config[i], index_
  else:
    assert not root
    return config, index


def sample_configuration_dist(config, root=True, num_samples_per_dist=1):
  """Expand configuration distribution specification."""
  if isinstance(config, dict):
    return {
        k: sample_configuration_dist(
            v, root=False, num_samples_per_dist=num_samples_per_dist)
        for k, v in sorted(config.items())
    }
  elif isinstance(config, list) and root:
    return [
        sample_configuration_dist(
            c, root=False, num_samples_per_dist=num_samples_per_dist)
        for c in config
    ]
  elif callable(config):
    return [config() for _ in range(num_samples_per_dist)]
  else:
    return config


def get_configuration_sample(config, root=True):
  """Get a sample of config."""
  if isinstance(config, dict):
    return {
        k: get_configuration_sample(v, root=False)
        for k, v in sorted(config.items())
    }
  elif isinstance(config, list):
    if root:
      return get_configuration_sample(
          config[np.random.randint(len(config))], root=False)
    else:
      return config[np.random.randint(len(config))]
  elif callable(config):
    return config()
  else:
    return config
