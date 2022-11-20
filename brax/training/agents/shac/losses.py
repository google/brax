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

"""Short-Horizon Actor Critic.

See: https://arxiv.org/pdf/2204.07137.pdf
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.shac import networks as shac_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class SHACNetworkParams:
  """Contains training state for the learner."""
  policy: Params
  value: Params


def compute_shac_policy_loss(
    policy_params: Params,
    value_params: Params,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    shac_network: shac_networks.SHACNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes SHAC critic loss.

  This implements Eq. 5 of 2204.07137.

  Args:
    policy_params: Policy network parameters
    value_params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    shac_network: SHAC networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.

  Returns:
    A scalar loss
  """

  parametric_action_distribution = shac_network.parametric_action_distribution
  policy_apply = shac_network.policy_network.apply
  value_apply = shac_network.value_network.apply

  # Put the time dimension first.
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

  # this is a redundant computation with the critic loss function
  # but there isn't a straighforward way to get these values when
  # they are used in that step
  values = value_apply(normalizer_params, value_params, data.observation)
  terminal_values = value_apply(normalizer_params, value_params, data.next_observation[-1])

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  # Append terminal values to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(terminal_values, 0)], axis=0)

  # jax implementation of https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/algorithms/shac.py#L227
  def sum_step(carry, target_t):
    gam, rew_acc = carry
    reward, termination = target_t

    # clean up gamma and rew_acc for done envs, otherwise update
    rew_acc = jnp.where(termination, 0, rew_acc + gam * reward)
    gam = jnp.where(termination, 1.0, gam * discounting)

    return (gam, rew_acc), (gam, rew_acc)

  rew_acc = jnp.zeros_like(terminal_values)
  gam = jnp.ones_like(terminal_values)
  (gam, last_rew_acc), (gam_acc, rew_acc) = jax.lax.scan(sum_step, (gam, rew_acc),
      (rewards, termination))

  policy_loss = jnp.sum(-last_rew_acc - gam * terminal_values)
  # for trials that are truncated (i.e. hit the episode length) include reward for
  # terminal state. otherwise, the trial was aborted and should receive zero additional
  policy_loss = policy_loss + jnp.sum((-rew_acc - gam_acc * jnp.where(truncation, values_t_plus_1, 0)) * termination)
  policy_loss = policy_loss / values.shape[0] / values.shape[1]


  # Entropy reward
  policy_logits = policy_apply(normalizer_params, policy_params,
                               data.observation)
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + entropy_loss

  return total_loss, {
    'policy_loss': policy_loss,
    'entropy_loss': entropy_loss
  }


def compute_shac_critic_loss(
    params: Params,
    normalizer_params: Any,
    data: types.Transition,
    shac_network: shac_networks.SHACNetworks,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    td_lambda: bool = True) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes SHAC critic loss.

  This implements Eq. 7 of 2204.07137
  https://github.com/NVlabs/DiffRL/blob/main/algorithms/shac.py#L349

  Args:
    params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    shac_network: SHAC networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.
    lambda_: Lambda for TD value updates
    td_lambda: whether to use a TD-Lambda value target

  Returns:
    A tuple (loss, metrics)
  """

  value_apply = shac_network.value_network.apply

  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

  values = value_apply(normalizer_params, params, data.observation)
  terminal_value = value_apply(normalizer_params, params, data.next_observation[-1])

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  # Append terminal values to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(terminal_value, 0)], axis=0)

  # compute target values
  if td_lambda:

    def compute_v_st(carry, target_t):
      Ai, Bi, lam = carry
      reward, vtp1, termination = target_t

      reward = reward * termination

      lam = lam * lambda_ * (1 - termination) + termination
      Ai = (1 - termination) * (lam * discounting * Ai + discounting * vtp1 + (1. - lam) / (1. - lambda_) * reward)
      Bi = discounting * (vtp1 * termination + Bi * (1.0 - termination)) + reward
      vs = (1.0 - lambda_) * Ai + lam * Bi

      return (Ai, Bi, lam), (vs)

    Ai = jnp.ones_like(terminal_value)
    Bi = jnp.zeros_like(terminal_value)
    lam = jnp.ones_like(terminal_value)
    (_, _, _), (vs) = jax.lax.scan(compute_v_st, (Ai, Bi, lam),
        (rewards, values_t_plus_1, termination),
        length=int(termination.shape[0]),
        reverse=True)

  else:
    vs = rewards + discounting * values_t_plus_1

  target_values = jax.lax.stop_gradient(vs)

  v_loss = jnp.mean((target_values - values) ** 2)

  total_loss = v_loss
  return total_loss, {
      'total_loss': total_loss,
      'policy_loss': 0,
      'v_loss': v_loss,
      'entropy_loss': 0
  }
