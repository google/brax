# Copyright 2021 The Brax Authors.
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

"""Augmented Random Search training.

See: https://arxiv.org/pdf/1803.07055.pdf
"""

import time
from typing import Any, Callable, Dict, Optional

from absl import logging
from brax import envs
from brax.training import env
from brax.training import networks
from brax.training import normalization
import flax
import jax
import jax.numpy as jnp
import optax

Params = Any


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  key: jnp.ndarray
  normalizer_params: Params
  policy_params: Params


def make_ars_model(act_size: int, obs_size: int):
  return networks.FeedForwardModel(
      init=lambda _: jnp.zeros((obs_size, act_size)),
      apply=lambda m, o: jnp.matmul(o, m))


def get_policy_head(head_type):
  def head(params):
    if not head_type:
      return params
    if head_type == 'clip':
      return jnp.clip(params, -1, 1)
    if head_type == 'tanh':
      return jnp.tanh(params)
    assert f'policy head type {head_type} is not known'
  return head


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps: int = 100,
    log_frequency: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    num_eval_envs: int = 128,
    seed: int = 0,
    normalize_observations: bool = False,
    step_size: float = 0.015,
    max_devices_per_host: Optional[int] = None,
    number_of_directions: int = 60,
    exploration_noise_std: float = 0.025,
    top_directions: int = 20,
    head_type: str = '',
    reward_shift: float = 0.0,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """ARS."""
  # TODO: pmap it
  max_devices_per_host = 1

  xt = time.time()
  top_directions = min(top_directions, number_of_directions)
  num_envs = number_of_directions * 2  # antitethic
  epochs = 1 + num_timesteps // episode_length // num_envs
  log_frequency = min(log_frequency, epochs)

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(), process_count, process_id, local_device_count,
      local_devices_to_use)

  key = jax.random.PRNGKey(seed)
  key, key_model, key_env, key_eval = jax.random.split(key, 4)

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs // local_devices_to_use // process_count,
      episode_length=episode_length)
  first_state, step_fn = env.wrap(core_env, key_env)

  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = env.wrap(core_eval_env, key_eval)

  _, obs_size = eval_first_state.core.obs.shape

  policy_model = make_ars_model(core_env.action_size, obs_size)
  policy_head = get_policy_head(head_type)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          obs_size, normalize_observations, num_leading_batch_dims=1,
          apply_clipping=False))

  policy_params = policy_model.init(key_model)

  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params = carry
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    actions = policy_head(policy_model.apply(policy_params, obs))
    nstate = eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params), ()

  @jax.jit
  def run_eval(state, policy_params, normalizer_params) -> env.EnvState:
    (state, _, _), _ = jax.lax.scan(
        do_one_step_eval, (state, policy_params, normalizer_params), (),
        length=episode_length // action_repeat)
    return state

  @jax.vmap
  def training_inference(params, obs):
    return policy_model.apply(params, obs)

  def do_one_step(carry, unused_target_t):
    state, policy_params, cumulative_reward, normalizer_params = carry
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    actions = policy_head(training_inference(policy_params, obs))
    nstate = step_fn(state, actions)
    cumulative_reward = cumulative_reward + nstate.core.reward - reward_shift
    return (nstate, policy_params, cumulative_reward,
            normalizer_params), state.core.obs

  def run_ars_eval(state, params, normalizer_params):
    cumulative_reward = jnp.zeros(state.core.obs.shape[0])
    (state, _, cumulative_reward, _), obs = jax.lax.scan(
        do_one_step, (state, params, cumulative_reward, normalizer_params),
        (), length=episode_length // action_repeat)
    return cumulative_reward, obs, state

  def add_noise(params, key):
    noise = jax.random.normal(key, shape=params.shape, dtype=params.dtype)
    params_with_noise = params + noise * exploration_noise_std
    anit_params_with_noise = params - noise * exploration_noise_std
    return params_with_noise, anit_params_with_noise, noise

  def ars_one_epoch(carry, unused_t):
    state, training_state = carry
    params = jnp.repeat(jnp.expand_dims(training_state.policy_params, axis=0),
                        num_envs // 2, axis=0)

    key, key_petr = jax.random.split(training_state.key)
    # generate perturbations
    params_with_noise, params_with_anti_noise, noise = add_noise(
        params, key_petr)

    pparams = jnp.concatenate([params_with_noise, params_with_anti_noise],
                              axis=0)

    eval_scores, obs, state = run_ars_eval(
        state, pparams, training_state.normalizer_params)

    obs = jnp.reshape(obs, [-1] + list(obs.shape[2:]))

    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, obs)

    reward_plus, reward_minus = jnp.split(eval_scores, 2, axis=0)
    reward_max = jnp.maximum(reward_plus, reward_minus)
    reward_rank = jnp.argsort(jnp.argsort(-reward_max))
    reward_weight = jnp.where(reward_rank < top_directions, 1, 0)
    reward_weight_double = jnp.concatenate([reward_weight, reward_weight],
                                           axis=0)
    reward_std = jnp.std(eval_scores, where=reward_weight_double)

    noise = jnp.sum(jnp.transpose(jnp.transpose(noise) * reward_weight *
                                  (reward_plus - reward_minus)), axis=0)

    policy_params = (training_state.policy_params +
                     step_size / (top_directions * reward_std) * noise)

    metrics = {
        'params_norm': optax.global_norm(policy_params),
        'eval_scores_mean': jnp.mean(eval_scores),
        'eval_scores_std': jnp.std(eval_scores),
        'reward_std': reward_std,
        'weights': jnp.mean(reward_weight),
    }
    return (state,
            TrainingState(
                key=key,
                normalizer_params=normalizer_params,
                policy_params=policy_params)), metrics

  epochs_per_step = (epochs + log_frequency - 1) // log_frequency
  @jax.jit
  def run_ars(state, training_state):
    (state, training_state), metrics = jax.lax.scan(
        ars_one_epoch, (state, training_state), (), length=epochs_per_step)
    return state, training_state, jax.tree_map(jnp.mean, metrics)

  training_state = TrainingState(key=key,
                                 normalizer_params=normalizer_params,
                                 policy_params=policy_params)

  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  metrics = {}
  summary = {}
  state = first_state

  for it in range(log_frequency + 1):
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    if process_id == 0:
      eval_state = run_eval(eval_first_state,
                            training_state.policy_params,
                            training_state.normalizer_params)
      eval_state.completed_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (
          episode_length * eval_first_state.core.reward.shape[0] /
          (time.time() - t))
      avg_episode_length = (
          eval_state.completed_episodes_steps / eval_state.completed_episodes)
      metrics = dict(
          dict({
              f'eval/episode_{name}': value / eval_state.completed_episodes
              for name, value in eval_state.completed_episodes_metrics.items()
          }),
          **dict({
              f'train/{name}': value for name, value in summary.items()
          }),
          **dict({
              'eval/completed_episodes': eval_state.completed_episodes,
              'eval/episode_length': avg_episode_length,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime,
          }))
      logging.info('Step %s metrics %s',
                   int(training_state.normalizer_params[0]) * action_repeat,
                   metrics)
      if progress_fn:
        progress_fn(int(training_state.normalizer_params[0]) * action_repeat,
                    metrics)

    if it == log_frequency:
      break

    t = time.time()
    # optimization
    state, training_state, summary = run_ars(state, training_state)

    jax.tree_map(lambda x: x.block_until_ready(), training_state)
    sps = episode_length * num_envs * epochs_per_step / (
        time.time() - t)
    training_walltime += time.time() - t

  _, inference = make_params_and_inference_fn(core_env.observation_size,
                                              core_env.action_size,
                                              normalize_observations,
                                              head_type)
  params = training_state.normalizer_params, training_state.policy_params

  return (inference, params, metrics)


def make_params_and_inference_fn(observation_size, action_size,
                                 normalize_observations, head_type=None):
  """Creates params and inference function for the ES agent."""
  obs_normalizer_params, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      observation_size, normalize_observations, apply_clipping=False)
  policy_head = get_policy_head(head_type)
  policy_model = make_ars_model(action_size, observation_size)

  def inference_fn(params, obs, unused_rng):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = policy_head(policy_model.apply(policy_params, obs))
    return action

  params = (obs_normalizer_params, policy_model.init(jax.random.PRNGKey(0)))
  return params, inference_fn
