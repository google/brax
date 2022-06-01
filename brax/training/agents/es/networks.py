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

"""Evolution strategy networks."""

from typing import Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class ESNetworks:
  policy_network: networks.FeedForwardModel
  parametric_action_distribution: distribution.ParametricDistribution


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Tuple[int, ...] = (32, 32, 32, 32)
) -> networks.FeedForwardModel:
  """Creates a policy network."""
  policy_module = networks.MLP(
      layer_sizes=hidden_layer_sizes + (param_size,),
      activation=linen.relu,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return networks.FeedForwardModel(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_inference_fn(es_networks: ESNetworks):
  """Creates params and inference function for the ES agent."""

  def make_policy(params: types.PolicyParams) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = es_networks.policy_network.apply(*params, observations)
      return es_networks.parametric_action_distribution.sample(
          logits, key_sample), {}

    return policy

  return make_policy


def make_es_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Tuple[int, ...] = (32, 32, 32, 32)
) -> ESNetworks:
  """Make ES networks."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes)
  return ESNetworks(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution)
