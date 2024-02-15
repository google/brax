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

"""IO for saving objects with npy file format."""

import os
from typing import Any

from brax.v1.io.file import File
from brax.v1.io.file import MakeDirs
import jax.numpy as jnp


def save(path: str, obj: Any, make_dir: bool = False):
  """Saves object in a .npy file."""
  if make_dir and path:
    MakeDirs(os.path.dirname(path))
  with File(path, 'wb') as f_out:
    jnp.save(f_out, obj)


def load(path: str) -> Any:
  """Loads .npy file."""
  with File(path, 'rb') as f_in:
    return jnp.load(f_in, allow_pickle=True)
