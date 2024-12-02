"""PPO vision networks."""

from typing import Any, Callable, Mapping, Sequence, Tuple

import jax
import jax.numpy as jp
import flax
from flax import linen
from flax.core import FrozenDict

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
from brax.training.agents.ppo.cnn_networks import VisionMLP


ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def remove_pixels(obs: FrozenDict) -> FrozenDict:
  """Remove pixel observations from the observation dict.
  FrozenDicts are used to avoid incorrect gradients."""
  pixel_keys = [k for k in obs.keys() if k.startswith('pixels/')]
  state_obs = obs
  for k in pixel_keys:
    state_obs, _ = state_obs.pop(k)
  return state_obs


def make_vision_policy_network(
  observation_size: Mapping[str, Tuple[int, ...]],
  output_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
  layer_norm: bool = False,
  state_obs_key: str = '',
  normalise_channels: bool = False) -> networks.FeedForwardNetwork:

  module = VisionMLP(
      layer_sizes=list(hidden_layer_sizes) + [output_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
      normalise_channels=normalise_channels)

  def apply(processor_params, policy_params, obs):
    if state_obs_key:
      state_obs = preprocess_observations_fn(
        remove_pixels(obs), processor_params
      )
      obs = obs.copy({state_obs_key: state_obs[state_obs_key]})
    return module.apply(policy_params, obs)

  dummy_obs = {key: jp.zeros((1,) + shape ) 
               for key, shape in observation_size.items()}
  
  return networks.FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply)


def make_vision_value_network(
  observation_size: Mapping[str, Tuple[int, ...]],
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
  state_obs_key: str = '',
  normalise_channels: bool = False) -> networks.FeedForwardNetwork:

  value_module = VisionMLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=kernel_init,
      normalise_channels=normalise_channels)

  def apply(processor_params, policy_params, obs):
    if state_obs_key:
      # Apply normaliser to state-based params.
      state_obs = preprocess_observations_fn(
        remove_pixels(obs), processor_params
      )
      obs = obs.copy({state_obs_key: state_obs[state_obs_key]})
    return jp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = {key: jp.zeros((1,) + shape ) 
               for key, shape in observation_size.items()}
  return networks.FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_vision_ppo_networks(
  # channel_size: int,
  observation_size: Mapping[str, Tuple[int, ...]],
  action_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types
  .identity_observation_preprocessor,
  policy_hidden_layer_sizes: Sequence[int] = [256, 256],
  value_hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  normalise_channels: bool = False) -> PPONetworks:
  """Make Vision PPO networks with preprocessor."""

  parametric_action_distribution = distribution.NormalTanhDistribution(
    event_size=action_size)

  policy_network = make_vision_policy_network(
    observation_size=observation_size,
    output_size=parametric_action_distribution.param_size,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=activation,
    hidden_layer_sizes=policy_hidden_layer_sizes,
    normalise_channels=normalise_channels)

  value_network = make_vision_value_network(
    observation_size=observation_size,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=activation,
    hidden_layer_sizes=value_hidden_layer_sizes,
    normalise_channels=normalise_channels)

  return PPONetworks(
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution)
