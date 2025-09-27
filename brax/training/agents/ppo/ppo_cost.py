# Copyright 2025 Safe-Brax Authors.
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

"""PPO-Cost: plain PPO optimizing reward - cost.

This module provides a minimal wrapper that subtracts cost from reward
and a convenience training function that reuses the standard PPO trainer.

Usage (example):

  from brax import envs
  from brax.training.agents.ppo import ppo_cost
  env = envs.get_environment('ant')
  make_policy, params, metrics = ppo_cost.train_ppo_cost(
      environment=env,
      num_timesteps=50_000_000,
      cost_weight=1.0,
  )

Assumptions:
- The environment provides a per-step cost signal via state.info['cost'].
- If 'cost' is missing, it is treated as 0.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import train as ppo_train


class RewardMinusCostWrapper(envs.Wrapper):
  """Environment wrapper that sets reward <- reward - cost_weight * cost.

  - Reads `cost` from next_state.info if present; otherwise assumes zero.
  - Keeps `cost` in info for downstream logging/analysis.
  - Adds `shaped_reward` to info for optional diagnostics.
  """

  def __init__(self, env: envs.Env, cost_weight: float = 1.0):
    super().__init__(env)
    self._cost_weight = float(cost_weight)

  def reset(self, rng: jnp.ndarray) -> envs.State:
    state = self.env.reset(rng)
    # Ensure 'cost' exists in info with correct shape
    info = state.info.copy()
    if 'cost' not in info:
      info['cost'] = jnp.zeros_like(state.reward)
    # Track the unshaped reward for debugging
    info['raw_reward'] = state.reward
    info['shaped_reward'] = state.reward - self._cost_weight * info['cost']
    return state.replace(info=info, reward=info['shaped_reward'])

  def step(self, state: envs.State, action: jnp.ndarray) -> envs.State:
    next_state = self.env.step(state, action)
    # Read cost (if not present, assume 0)
    cost = next_state.info.get('cost', jnp.zeros_like(next_state.reward))
    shaped_reward = next_state.reward - self._cost_weight * cost

    # Preserve/augment info
    info = next_state.info.copy()
    info['raw_reward'] = next_state.reward
    info['shaped_reward'] = shaped_reward

    return next_state.replace(reward=shaped_reward, info=info)


def _apply_reward_minus_cost(env: envs.Env, cost_weight: float) -> envs.Env:
  return RewardMinusCostWrapper(env, cost_weight=cost_weight)


def train_ppo_cost(
    environment: envs.Env,
    num_timesteps: int,
    cost_weight: float = 1.0,
    wrap_env: bool = True,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.97,
    unroll_length: int = 10,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 2,
    normalize_observations: bool = True,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    seed: int = 0,
    progress_fn: Callable[[int, ppo_train.Metrics], None] = lambda *args: None,
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    **kwargs,
) -> Tuple[ppo_train.InferenceParams, ppo_train.Metrics]:
  """Runs standard PPO optimizing reward - cost_weight * cost.

  Args mirror `ppo.train` with an additional `cost_weight` and `environment`.
  Returns the same outputs as `ppo.train`.
  """

  # Pre-wrap the training environment so vectorization happens after this wrapper.
  env_for_training = _apply_reward_minus_cost(environment, cost_weight=cost_weight) if wrap_env else environment

  # If an eval_env is provided, wrap it similarly so eval metrics reflect shaped reward
  eval_env = kwargs.pop('eval_env', None)
  if eval_env is not None and wrap_env:
    eval_env = _apply_reward_minus_cost(eval_env, cost_weight=cost_weight)

  make_policy, params, metrics = ppo_train.train(
      environment=env_for_training,
      num_timesteps=num_timesteps,
      wrap_env=wrap_env,
      num_envs=num_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      # Do not pass wrap_env_fn; we already applied our wrapper above
      learning_rate=learning_rate,
      entropy_cost=entropy_cost,
      discounting=discounting,
      unroll_length=unroll_length,
      batch_size=batch_size,
      num_minibatches=num_minibatches,
      num_updates_per_batch=num_updates_per_batch,
      normalize_observations=normalize_observations,
      reward_scaling=reward_scaling,
      clipping_epsilon=clipping_epsilon,
      gae_lambda=gae_lambda,
      seed=seed,
      progress_fn=progress_fn,
      save_checkpoint_path=save_checkpoint_path,
      restore_checkpoint_path=restore_checkpoint_path,
      restore_params=restore_params,
      restore_value_fn=restore_value_fn,
      eval_env=eval_env,
      **kwargs,
  )

  return make_policy, params, metrics



