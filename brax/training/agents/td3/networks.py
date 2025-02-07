"""TD3 networks."""

from typing import Sequence, Tuple

import jax.numpy as jnp
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer, MLP
from brax.training.types import PRNGKey
from flax import linen, struct
import jax


@struct.dataclass
class TD3Networks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork


def make_inference_fn(td3_networks: TD3Networks):
    """Creates params and inference function for the TD3 agent."""

    def make_policy(params: types.PolicyParams, exploration_noise, noise_clip) -> types.Policy:
        def policy(observations: types.Observation,
                   key_noise: PRNGKey) -> Tuple[types.Action, types.Extra]:
            actions = td3_networks.policy_network.apply(*params, observations)
            noise = (jax.random.normal(key_noise, actions.shape) * exploration_noise).clip(-noise_clip, noise_clip)
            return actions + noise, {}

        return policy

    return make_policy


def make_policy_network(
        param_size: int,
        obs_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm)

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_td3_networks(
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: networks.ActivationFn = linen.relu) -> TD3Networks:
    """Make TD3 networks."""
    policy_network = make_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation)

    return TD3Networks(
        policy_network=policy_network,
        q_network=q_network)



