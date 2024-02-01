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

"""Objects which specify the input/output spaces of an environment.


This file was taken from acme and modified to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/specs.py
"""
import dataclasses
from typing import Tuple

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Array:
  """Describes a numpy array or scalar shape and dtype.

  Similar to dm_env.specs.Array.
  """
  shape: Tuple[int, ...]
  dtype: jnp.dtype
