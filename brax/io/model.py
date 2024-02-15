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

"""Loading/saving of inference functions."""

import pickle
from typing import Any
from etils import epath


def load_params(path: str) -> Any:
  with epath.Path(path).open('rb') as fin:
    buf = fin.read()
  return pickle.loads(buf)


def save_params(path: str, params: Any):
  """Saves parameters in flax format."""
  with epath.Path(path).open('wb') as fout:
    fout.write(pickle.dumps(params))
