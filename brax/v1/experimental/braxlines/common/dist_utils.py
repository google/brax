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

"""Distribution utilility functions."""

from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def clipped_onehot_categorical(logits: jnp.ndarray, clip_range: float = 0):
  if clip_range:
    assert clip_range > 0.0, clip_range
    logits -= jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.clip(logits, a_min=-clip_range)
  return tfd.OneHotCategorical(logits=logits)


def clipped_bernoulli(logits: jnp.ndarray, clip_range: float = 0):
  if clip_range:
    assert clip_range > 0.0, clip_range
    logits = jnp.clip(logits, a_min=-clip_range, a_max=clip_range)
  return tfd.Bernoulli(logits=logits)
