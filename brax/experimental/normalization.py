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

"""Input normalization utils."""

from typing import Optional

from brax.training import pmap
import jax
import jax.numpy as jnp


def create_observation_normalizer(obs_size, normalize_observations=True,
                                  pmap_to_devices: Optional[int] = None,
                                  pmap_axis_name: str = 'i',
                                  num_leading_batch_dims: int = 1,
                                  apply_clipping: bool = True):
  """Observation normalization based on running statistics."""
  assert num_leading_batch_dims == 1 or num_leading_batch_dims == 2
  if normalize_observations:
    def update_fn(params, obs, mask=None):
      normalization_steps, running_mean, running_variance = params

      if mask is not None:
        mask = jnp.expand_dims(
            mask, axis=-1)  # for shape matching during multiplication
        step_increment = jnp.sum(mask)
      else:
        step_increment = obs.shape[0] * (
            obs.shape[1] if num_leading_batch_dims == 2 else 1)
      if pmap_to_devices:
        step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
      total_new_steps = normalization_steps + step_increment

      # Compute the incremental update and divide by the number of new steps.
      input_to_old_mean = obs - running_mean
      if mask is not None:
        input_to_old_mean = input_to_old_mean * mask
      mean_diff = jnp.sum(input_to_old_mean / total_new_steps,
                          axis=((0, 1) if num_leading_batch_dims == 2 else 0))
      if pmap_to_devices:
        mean_diff = jax.lax.psum(mean_diff, axis_name=pmap_axis_name)
      new_mean = running_mean + mean_diff

      # Compute difference of input to the new mean for Welford update.
      input_to_new_mean = obs - new_mean
      if mask is not None:
        input_to_new_mean = input_to_new_mean * mask
      var_diff = jnp.sum(input_to_new_mean * input_to_old_mean,
                         axis=((0, 1) if num_leading_batch_dims == 2 else 0))
      if pmap_to_devices:
        var_diff = jax.lax.psum(var_diff, axis_name=pmap_axis_name)

      return (total_new_steps, new_mean, running_variance + var_diff)

  else:
    def update_fn(params, obs, mask=None):
      if mask is not None:
        step_increment = jnp.sum(mask)
      else:
        step_increment = obs.shape[0] * (
            obs.shape[1] if num_leading_batch_dims == 2 else 1)
      if pmap_to_devices:
        step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
      return (params[0] + step_increment, params[1], params[2])
  data, apply_fn = make_data_and_apply_fn(obs_size, normalize_observations,
                                          apply_clipping)
  if pmap_to_devices:
    data = pmap.bcast_local_devices(data, pmap_to_devices)
  return data, update_fn, apply_fn


def make_data_and_apply_fn(obs_size, normalize_observations=True,
                           apply_clipping=True):
  """Creates data and an apply function for the normalizer."""
  if normalize_observations:
    data = (jnp.zeros(()), jnp.zeros((obs_size,)), jnp.ones((obs_size,)))
    def apply_fn(params, obs, std_min_value=1e-6, std_max_value=1e6):
      normalization_steps, running_mean, running_variance = params
      variance = running_variance / (normalization_steps + 1.0)
      # We clip because the running_variance can become negative,
      # presumably because of numerical instabilities.
      if apply_clipping:
        variance = jnp.clip(variance, std_min_value, std_max_value)
        return jnp.clip((obs - running_mean) / jnp.sqrt(variance), -5, 5)
      else:
        return (obs - running_mean) / jnp.sqrt(variance)
  else:
    data = (jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
    def apply_fn(params, obs):
      del params
      return obs
  return data, apply_fn
