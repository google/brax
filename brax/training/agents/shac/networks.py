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

"""SHAC networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class SHACNetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(shac_networks: SHACNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    policy_network = shac_networks.policy_network
    parametric_action_distribution = shac_networks.parametric_action_distribution

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = policy_network.apply(*params, observations)
      if deterministic:
        return shac_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions)
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions
      }


    return policy

  return make_policy


def make_shac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.elu,
    layer_norm: bool = True) -> SHACNetworks:
  """Make SHAC networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm)
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm)

  return SHACNetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)
