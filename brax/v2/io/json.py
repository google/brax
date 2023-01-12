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
"""Saves a system config and trajectory as json."""

import json
from typing import List, Text

from brax.v2.base import State, System
from etils import epath
import jax
import jax.numpy as jp
import numpy as np

# State attributes needed for the visualizer.
_STATE_ATTR = ['x', 'contact']


def _to_dict(obj):
  """Converts python object to a json encodeable object."""
  if isinstance(obj, list) or isinstance(obj, tuple):
    return [_to_dict(s) for s in obj]
  if isinstance(obj, dict):
    return {k: _to_dict(v) for k, v in obj.items()}
  if isinstance(obj, jax.Array):
    return _to_dict(obj.tolist())
  if hasattr(obj, '__dict__'):
    d = dict(obj.__dict__)
    d['name'] = obj.__class__.__name__
    return _to_dict(d)
  if isinstance(obj, np.ndarray):
    return _to_dict(obj.tolist())
  if isinstance(obj, np.floating):
    return float(obj)
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
    return str(obj)
  return obj


def _compress_contact(states: List[State]) -> List[State]:
  """Reduces the number of contacts based on penetration > 0."""
  stacked = jax.tree_map(lambda *x: jp.stack(x), *states)
  if stacked.contact is None:
    return states
  contact_mask = stacked.contact.penetration > 0
  n_contact = contact_mask.sum(axis=1).max()

  def pad(arr, n):
    r = jp.zeros(n)
    if len(arr.shape) > 1:
      r = jp.zeros((n, arr.shape[1]))
    r = r.at[: arr.shape[0]].set(arr)
    return r

  def compress(i, s):
    mask = contact_mask[i]
    contact = jax.tree_map(lambda x: pad(x[mask], n_contact), s.contact)
    return s.replace(contact=contact)

  return [compress(i, s) for i, s in enumerate(states)]


def dumps(sys: System, states: List[State]) -> Text:
  """Creates a json string of the system config.

  Args:
    sys: brax System object
    states: list of brax system states

  Returns:
    string containing json dump of system and states
  """
  d = _to_dict(sys)

  # fill in empty link names
  link_names = [n or f'link {i}' for i, n in enumerate(sys.link_names)]

  # key geoms by their link names
  link_geoms = {}
  for g in d['geoms']:
    link_name = 'world' if g['link_idx'] is None else link_names[g['link_idx']]
    link_geoms.setdefault(link_name, []).append(g)
  d['geoms'] = link_geoms

  states = _compress_contact(states)

  # stack states for the viewer
  states = _to_dict(jax.tree_map(lambda *x: jp.stack(x), *states))
  d['states'] = {k: states[k] for k in _STATE_ATTR}

  return json.dumps(_to_dict(d))


def save(path: str, sys: System, states: List[State]):
  """Saves a system config and trajectory as json."""
  with epath.Path(path).open('w') as fout:
    fout.write(dumps(sys, states))
