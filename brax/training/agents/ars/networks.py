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

"""ARS networks."""

from typing import Tuple

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import jax.numpy as jnp

ARSNetwork = networks.FeedForwardNetwork


def make_policy_network(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
) -> ARSNetwork:
  """Creates a policy network."""

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.matmul(obs, policy_params)

  return ARSNetwork(
      init=lambda _: jnp.zeros((observation_size, action_size)), apply=apply)


def make_inference_fn(policy_network: ARSNetwork):
  """Creates params and inference function for the ARS agent."""

  def make_policy(params: types.PolicyParams) -> types.Policy:

    def policy(observations: types.Observation,
               unused_key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      return policy_network.apply(*params, observations), {}

    return policy

  return make_policy
