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

# pylint:disable=g-multiple-import
"""Base types for positional pipeline."""

from brax import base
from brax.base import Motion, Transform
from flax import struct
import jax
import jax.numpy as jp


@struct.dataclass
class State(base.State):
  """Dynamic state that changes after every step.

  Attributes:
    x_i: link center of mass in world frame
    xd_i: link center of mass motion in world frame
    j: link position in joint frame
    jd: link motion in joint frame
    a_p: joint parent anchor in world frame
    a_c: joint child anchor in world frame
    mass: link mass
  """

  x_i: Transform
  xd_i: Motion
  j: Transform
  jd: Motion
  a_p: Transform
  a_c: Transform
  mass: jax.Array
