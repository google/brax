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

"""Saves a system config and trajectory as json."""

import json
from typing import List, Optional, Text

import brax.v1 as brax
from brax.v1.io.file import File
import jax.numpy as jnp
import numpy as onp

from google.protobuf import json_format


class JaxEncoder(json.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, jnp.ndarray):
      return obj.tolist()
    if isinstance(obj, onp.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def dumps(sys: brax.System,
          qps: List[brax.QP],
          info: Optional[List[brax.Info]] = None) -> Text:
  """Creates a json string of the system config."""
  d = {
      'config': json_format.MessageToDict(sys.config, True),
      'pos': [qp.pos for qp in qps],
      'rot': [qp.rot for qp in qps],
      'debug': info is not None,
  }
  if info:
    # Add debug info for the contact points.
    max_len = max([sum(onp.array(i.contact_penetration) > 0) for i in info])

    def _pad_arr(arr):
      arr = onp.array(arr)
      padding = -onp.ones(max_len - arr.shape[0])
      if len(arr.shape) > 1:
        padding = -onp.ones((max_len - arr.shape[0], arr.shape[1]))
      return onp.concatenate([arr, padding])

    # Pad the contact points.
    d['contact_pos'] = [
        _pad_arr(i.contact_pos[i.contact_penetration > 0]) for i in info
    ]
    d['contact_normal'] = [
        _pad_arr(i.contact_normal[i.contact_penetration > 0]) for i in info
    ]
    d['contact_penetration'] = [
        _pad_arr(i.contact_penetration[i.contact_penetration > 0]) for i in info
    ]
  return json.dumps(d, cls=JaxEncoder)


def save(path: str,
         sys: brax.System,
         qps: List[brax.QP],
         info: Optional[List[brax.Info]] = None):
  """Saves a system config and trajectory as json."""
  with File(path, 'w') as fout:
    system = dumps(sys, qps, info)
    fout.write(system)
