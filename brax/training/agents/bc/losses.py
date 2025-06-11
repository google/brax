# Copyright 2025 The Brax Authors.
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

"""Losses for BC."""

from typing import Any, Callable, Dict, Tuple

from brax.training.agents.bc import networks
from brax.training.types import Params
import jax.numpy as jp


# Vanilla L2 with postprocessing
def bc_loss(
    params: Params,
    normalizer_params: Any,
    data: Dict,
    make_policy: Callable[[Tuple[Any, Params]], networks.BCInferenceFn],
):
  policy = make_policy((normalizer_params, params))
  _, action_extras = policy(data['observations'], key_sample=None)  # pytype: disable=wrong-keyword-args
  actor_loss = (
      (
          (
              jp.tanh(action_extras['loc'])
              - jp.tanh(data['teacher_action_extras']['loc'])
          )
          ** 2
      )
      .sum(-1)
      .mean()
  )
  actor_loss = actor_loss.mean()
  return actor_loss, {'actor_loss': actor_loss, 'mse_loss': actor_loss}
