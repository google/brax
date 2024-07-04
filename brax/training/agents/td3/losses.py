"""Twin Delayed Deep Deterministic Policy Gradient (TD3) losses.

See: https://arxiv.org/pdf/1802.09477.pdf
"""
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import networks as td3_networks

Transition = types.Transition


def make_losses(
        td3_network: td3_networks.TD3Networks,
        reward_scaling: float,
        discounting: float,
        smoothing: float,
        noise_clip: float,
        max_action: float = 1.0,
        bc: bool = False,
        alpha: float = 2.5):
    """Creates the TD3 losses."""
    policy_network = td3_network.policy_network
    q_network = td3_network.q_network

    def critic_loss(
            q_params: Params,
            target_q_params: Params,
            target_policy_params: Params,
            normalizer_params: Any,
            transitions: Transition,
            key: PRNGKey) -> jnp.ndarray:
        """Calculates the TD3 critic loss."""

        current_q1_q2 = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
        next_actions = policy_network.apply(normalizer_params, target_policy_params, transitions.next_observation)
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(-noise_clip, noise_clip)
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)

        next_q1_q2 = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_actions)
        target_q = jnp.min(next_q1_q2, axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discounting * target_q)

        q_error = current_q1_q2 - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss

    def actor_loss(
            policy_params: Params,
            q_params: Params,
            normalizer_params: Any,
            transitions: Transition) -> jnp.ndarray:
        """Calculates the TD3 actor loss."""

        new_actions = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        q_new_actions = q_network.apply(normalizer_params, q_params, transitions.observation, new_actions)
        q_new_actions, _ = jnp.split(q_new_actions, 2, axis=-1)
        lmbda = jax.lax.stop_gradient(bc * alpha / jnp.mean(jnp.abs(q_new_actions)) + (1 - bc))
        q_mean = jnp.mean(q_new_actions)
        return - lmbda * q_mean + bc * mean_squared_error(new_actions, transitions.action)

    def mean_squared_error(predictions, targets):
        return jnp.mean(jnp.square(predictions - targets))

    return critic_loss, actor_loss
