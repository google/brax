"""PPO vision networks."""
from functools import partial
from typing import Sequence, Tuple, Any, Callable, Mapping

import jax
import jax.numpy as jp
import flax
from flax import linen

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


def pure_pop(key: str, x: dict) -> Tuple[Any, dict]:
    _x = dict(x)
    v = _x.pop(key)
    return v, _x


def make_vision_policy_network(
  network_type: str,
  observation_size: Mapping[str, Tuple[int, ...]],
  output_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
  layer_norm: bool = False,
  normalise_channels: bool = False) -> networks.FeedForwardNetwork:

  if network_type == 'cnn':
    module = VisionMLP(
        layer_sizes=list(hidden_layer_sizes) + [output_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        normalise_channels=normalise_channels)
  else:
    raise ValueError(f'Unsupported network_type: {network_type}')

  def apply(processor_params, policy_params, obs):
    state_obs, _obs = pure_pop('state', obs)
    appl = {
        'state': preprocess_observations_fn(state_obs, processor_params),
        **_obs
    }

    return module.apply(policy_params, appl)

  dummy_obs = {key: jp.zeros((1,) + shape ) 
               for key, shape in observation_size.items()}
  
  return networks.FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply)


def make_vision_value_network(
  network_type: str,
  observation_size: Mapping[str, Tuple[int, ...]],
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
  normalise_channels: bool = False) -> networks.FeedForwardNetwork:

  if  network_type == 'cnn':
    value_module = VisionMLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
        normalise_channels=normalise_channels)
  else:
    raise ValueError(f'Unsupported network_type: {network_type}')

  def apply(processor_params, policy_params, obs):
    state_obs, _obs = pure_pop('state', obs)
    appl = {
        'state': preprocess_observations_fn(state_obs, processor_params),
        **_obs
    }
    return jp.squeeze(value_module.apply(policy_params, appl), axis=-1)

  dummy_obs = {key: jp.zeros((1,) + shape ) 
               for key, shape in observation_size.items()}
  return networks.FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = policy_network.apply(*params, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
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
    network_type='cnn',
    observation_size=observation_size,
    output_size=parametric_action_distribution.param_size,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=activation,
    hidden_layer_sizes=policy_hidden_layer_sizes,
    normalise_channels=normalise_channels)

  value_network = make_vision_value_network(
    network_type='cnn',
    observation_size=observation_size,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=activation,
    hidden_layer_sizes=value_hidden_layer_sizes,
    normalise_channels=normalise_channels)

  return PPONetworks(
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution)
