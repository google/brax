"""PPO vision networks."""

from functools import partial
from typing import Sequence, Tuple, Any, Callable

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey

import flax
from flax import linen
import jax
import jax.numpy as jp

from vision_cnn import NatureCNN

ModuleDef = Any
ActivationFn = Callable[[jp.ndarray], jp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_vision_policy_network(
  network_type: str,
  observation_shape: Tuple[int, int, int, int],
  output_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  policy_hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
  layer_norm: bool = False) -> networks.FeedForwardNetwork:

  if network_type == 'cnn':
    module = ManiSkillCNN(
        layer_sizes=list(policy_hidden_layer_sizes) + [output_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm)
  else:
    raise ValueError(f'Unsupported network_type: {network_type}')

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return module.apply(policy_params, obs)

  dummy_obs = jp.zeros(observation_shape)
  return networks.FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply)


def make_vision_value_network(
  network_type: str,
  observation_shape: Tuple[int, int, int, int],
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  value_hidden_layer_sizes: Sequence[int] = [256, 256],
  activation: ActivationFn = linen.swish,
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  ) -> networks.FeedForwardNetwork:

  if network_type == 'cnn':
    value_module = ManiSkillCNN(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init)
  else:
    raise ValueError(f'Unsupported network_type: {network_type}')

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jp.zeros(observation_shape)
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
  channel_size: int,
  action_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types
  .identity_observation_preprocessor,
  policy_hidden_layer_sizes: Sequence[int] = [256, 256],
  value_hidden_layer_sizes: Sequence[int] = [256, 256],
  image_dim: Tuple[int, int] = [64, 64],
  activation: ActivationFn = linen.swish) -> PPONetworks:
  """Make Vision PPO networks with preprocessor."""
  
  # Temp hack since brax only passes the last value of shape for observation size
  image_observation_shape = (1, image_dim[0], image_dim[1], channel_size)

  parametric_action_distribution = distribution.NormalTanhDistribution(
    event_size=action_size)

  policy_network = make_vision_policy_network(
    'cnn',
    image_observation_shape,
    parametric_action_distribution.param_size,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=linen.relu)

  value_network = make_vision_value_network(
    'cnn',
    image_observation_shape,
    preprocess_observations_fn=preprocess_observations_fn,
    activation=linen.relu)

  return PPONetworks(
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution)