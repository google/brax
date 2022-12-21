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

from brax.v2.base import System, Transform
from etils import epath
import jax
import numpy as np


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


def dumps(sys: System, xs: List[Transform]) -> Text:
  """Creates a json string of the system config."""
  d = _to_dict(sys)

  # fill in empty link names
  link_names = [n or f'link {i}' for i, n in enumerate(sys.link_names)]

  # group geoms by their links
  link_geoms = {}
  for g in d['geoms']:
    link = 'world' if g['link_idx'] is None else link_names[g['link_idx']]
    link_geoms.setdefault(link, []).append(g)
  d['geoms'] = link_geoms

  d['pos'] = [x.pos for x in xs]
  d['rot'] = [x.rot for x in xs]
  d['debug'] = False  # TODO implement debugging
  return json.dumps(_to_dict(d))


def save(path: str, sys: System, xs: List[Transform]):
  """Saves a system config and trajectory as json."""
  with epath.Path(path).open('w') as fout:
    fout.write(dumps(sys, xs))
