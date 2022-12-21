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
"""Physics pipeline for fully articulated dynamics and collisiion."""
from typing import Tuple

from brax.v2.base import Motion, System, Transform
from jax import numpy as jp


def step(sys: System,
         x: Transform,
         xd: Motion,
         tau: jp.ndarray) -> Tuple[Transform, Motion]:
  """Performs a single physics step.

  Args:
    sys: system defining the kinematic tree and other properties
    x: link transform in world space
    xd: link motion in world space
    tau: joint force input vector

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  # TODO: implement

  del sys, tau

  return x, xd
