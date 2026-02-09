# Copyright 2026 The Brax Authors.
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

"""Optimizers for PPO."""

import enum
from typing import Tuple
import jax
from jax import numpy as jnp
import optax


class LRSchedule(enum.Enum):
  """Learning rate schedule enum."""

  NONE = 'NONE'
  ADAPTIVE_KL = 'ADAPTIVE_KL'


def adaptive_kl_learning_rate(
    optimizer_state: optax.OptState, kl_mean: jnp.ndarray, desired_kl: float
) -> Tuple[optax.OptState, jnp.ndarray]:
  """Adaptive KL learning rate schedule."""
  kl_mean = jax.lax.stop_gradient(kl_mean)

  optim_state = optimizer_state
  if isinstance(optimizer_state, tuple) and not hasattr(
      optimizer_state, 'hyperparams'
  ):
    optim_state = optimizer_state[-1]
    assert hasattr(optim_state, 'hyperparams')

  lr = optim_state.hyperparams['learning_rate']  # pytype: disable=attribute-error
  lr = jnp.where(kl_mean > desired_kl * 2.0, jnp.maximum(1e-5, lr / 1.5), lr)
  lr = jnp.where(
      (kl_mean < desired_kl / 2.0) & (kl_mean > 0.0),
      jnp.minimum(1e-2, lr * 1.5),
      lr,
  )
  optim_state.hyperparams['learning_rate'] = lr  # pytype: disable=attribute-error

  return optimizer_state, lr
