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

"""BC networks."""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jp


@flax.struct.dataclass
class BCNetworks:
  policy_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


BCInferenceFn = Callable[
    [types.Observation, PRNGKey], Tuple[jp.ndarray, Mapping[str, Any]]
]


def make_inference_fn(bc_networks: BCNetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      params: types.Params,
      *,
      deterministic: bool = True,
      tanh_squash: bool = False,
  ) -> BCInferenceFn:
    """Keeping unused deterministic and key_sample for API compatibility.

    (BC inference is always deterministic)
    """
    policy_network = bc_networks.policy_network
    parametric_action_distribution = bc_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: Optional[PRNGKey]
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      action_dist = parametric_action_distribution.create_dist(logits)
      act = (
          parametric_action_distribution.mode(logits)
          if tanh_squash
          else action_dist.loc
      )
      metainfo = {'loc': action_dist.loc, 'scale': action_dist.scale}
      return act, metainfo

    return policy

  return make_policy


def make_bc_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    *,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = '',
    vision: bool = False,
    latent_vision: bool = False,
) -> BCNetworks:
  """Make BC networks with preprocessor.

  Note that vision = True assumes Frozen Encoder.
  """
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )

  if latent_vision:
    make_policy_network = networks.make_policy_network_latents
  elif vision:
    make_policy_network = networks.make_policy_network_vision
  else:
    make_policy_network = networks.make_policy_network

  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )

  return BCNetworks(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution,
  )
