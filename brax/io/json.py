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

"""Saves a system config and trajectory as json."""

import json
from typing import List

import brax
from brax.io.file import File
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


def save(path: str, sys: brax.System, qps: List[brax.QP]):
  with File(path, 'w') as fout:
    d = {'config': json_format.MessageToDict(sys.config, True),
         'pos': [qp.pos for qp in qps],
         'rot': [qp.rot for qp in qps],}
    fout.write(json.dumps(d, cls=JaxEncoder))
