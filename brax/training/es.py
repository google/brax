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

"""Evolution strategy training.

See: https://arxiv.org/pdf/1703.03864.pdf
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from brax import envs
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  key: PRNGKey
  normalizer_params: Params
  optimizer_state: optax.OptState
  policy_params: Params


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps: int = 100,
    episode_length: int = 1000,
    action_repeat: int = 1,
    l2coeff: float = 0,
    max_devices_per_host: Optional[int] = None,
    population_size: int = 128,
    learning_rate: float = 0.001,
    fitness_shaping: int = 0,
    num_eval_envs: int = 128,
    perturbation_std: float = 0.1,
    seed: int = 0,
    normalize_observations: bool = False,
    log_frequency: int = 1,
    center_fitness: bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """ES training (from https://arxiv.org/pdf/1703.03864.pdf)."""
  num_envs = population_size * 2  # antitethic

  xt = time.time()

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
  key_envs = jax.random.split(key_env, local_devices_to_use)
  step_fn = jax.jit(core_env.step)
  reset_fn = jax.jit(jax.vmap(core_env.reset))
  first_state = reset_fn(key_envs)

  eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length,
      eval_metrics=True)
  eval_first_state = jax.jit(eval_env.reset)(key_eval)
  eval_step_fn = jax.jit(eval_env.step)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)

  _, obs_size = eval_first_state.obs.shape

  policy_model = make_es_model(parametric_action_distribution, obs_size)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          obs_size, normalize_observations, num_leading_batch_dims=2))

  optimizer = optax.adam(learning_rate=learning_rate)
  policy_params = policy_model.init(key_model)
  optimizer_state = optimizer.init(policy_params)

  key_debug = jax.random.PRNGKey(seed + 666)

  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
    logits = policy_model.apply(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params,
               normalizer_params) -> Tuple[envs.State, PRNGKey]:
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  @jax.vmap
  def training_inference(params, obs):
    return policy_model.apply(params, obs)

  def do_one_step(carry, unused_target_t):
    state, policy_params, key, cumulative_reward, active_episode, normalizer_params = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
    logits = training_inference(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = step_fn(state, actions)
    cumulative_reward = cumulative_reward + nstate.reward * active_episode
    new_active_episode = active_episode * (1 - nstate.done)
    return (nstate, policy_params, key, cumulative_reward, new_active_episode,
            normalizer_params), (state.obs, active_episode)

  def run_es_eval(state, params, key, normalizer_params):
    cumulative_reward = jnp.zeros(state.obs.shape[0])
    active_episode = jnp.ones_like(cumulative_reward)
    (state, _, key, cumulative_reward, _, _), (obs, obs_weights) = jax.lax.scan(
        do_one_step, (state, params, key, cumulative_reward, active_episode,
                      normalizer_params), (),
        length=episode_length // action_repeat)
    return cumulative_reward, obs, obs_weights, state

  def add_noise(params, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), params,
        jax.tree_unflatten(treedef, all_keys))
    params_with_noise = jax.tree_multimap(lambda g, n: g + n * perturbation_std,
                                          params, noise)
    anit_params_with_noise = jax.tree_multimap(
        lambda g, n: g - n * perturbation_std, params, noise)
    return params_with_noise, anit_params_with_noise, noise

  @jax.jit
  def es_one_epoch(state: envs.State, training_state: TrainingState):
    params = jax.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, axis=0),
                             population_size, axis=0),
        training_state.policy_params)
    key, key_petr, key_es_eval = jax.random.split(training_state.key, 3)
    # generate perturbations
    params_with_noise, params_with_anti_noise, noise = add_noise(
        params, key_petr)

    pparams = jax.tree_multimap(lambda a, b: jnp.concatenate([a, b], axis=0),
                                params_with_noise, params_with_anti_noise)

    pparams = jax.tree_map(
        lambda x: jnp.reshape(x, [local_devices_to_use, -1] + list(x.shape[1:])
                             ), pparams)

    prun_es_eval = jax.pmap(run_es_eval, in_axes=(0, 0, None, None))
    eval_scores, obs, obs_weights, state = prun_es_eval(
        state, pparams, key_es_eval,
        training_state.normalizer_params)

    obs = jnp.reshape(obs, [-1] + list(obs.shape[2:]))
    obs_weights = jnp.reshape(obs_weights, [-1] + list(obs_weights.shape[2:]))

    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, obs, obs_weights)

    weights = jnp.reshape(eval_scores, [-1])

    # aggregate results
    if fitness_shaping == 1:
      # Shaping from
      # https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
      weights = weights.shape[0] - jnp.argsort(jnp.argsort(weights))
      weights = jnp.maximum(
          0,
          jnp.log(weights.shape[0] / 2.0 + 1) - jnp.log(weights))
      weights = weights / jnp.sum(weights) - 1.0 / weights.shape[0]
    elif fitness_shaping == 2:
      # Centered rank from: https://arxiv.org/pdf/1703.03864.pdf
      weights = jnp.argsort(jnp.argsort(weights))
      weights /= (weights.shape[0] - 1)
      weights -= .5
    elif fitness_shaping == 0:
      # Original scores.
      pass
    else:
      assert 0

    if center_fitness:
      weights = (weights - jnp.mean(weights)) / (1e-6 + jnp.std(weights))

    weights1, weights2 = jnp.split(weights, 2)
    weights = weights1 - weights2

    # NOTE: a paper does "len(weights) -> len(weights) * perturbation_std,
    # but it's just a matter of a different tuning for l2_coef
    delta = jax.tree_multimap(
        lambda y: 1. /
        (len(weights)) * jnp.sum(
            y * jnp.reshape(weights, ([weights.shape[0]] + [1] *
                                      (len(y.shape) - 1))),
            axis=0), noise)
    # l2coeff controls the weight decay of the parameters of our policy network.
    # This prevents the parameters from growing very large compared to the
    # perturbations.
    delta = jax.tree_multimap(lambda d, th: d - l2coeff * th, delta,
                              training_state.policy_params)
    delta = jax.tree_map(lambda x: -x, delta)

    params_update, optimizer_state = optimizer.update(
        delta, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params,
                                        params_update)

    metrics = {
        'params_norm': optax.global_norm(policy_params),
        'eval_scores_mean': jnp.mean(eval_scores),
        'eval_scores_std': jnp.std(eval_scores),
        'weights': jnp.mean(weights),
    }
    return (TrainingState(
                key=key,
                normalizer_params=normalizer_params,
                optimizer_state=optimizer_state,
                policy_params=policy_params), metrics)

  training_state = TrainingState(
      key=key,
      normalizer_params=normalizer_params,
      optimizer_state=optimizer_state,
      policy_params=policy_params)

  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  metrics = {}
  summary = {}
  state = first_state
  it = -1

  while True:
    it += 1
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()
    num_process_env_steps = int(
        training_state.normalizer_params[0]) * action_repeat

    if process_id == 0 and it % log_frequency == 0:
      eval_state, key_debug = (
          run_eval(eval_first_state, key_debug,
                   training_state.policy_params,
                   training_state.normalizer_params))
      eval_metrics = eval_state.info['eval_metrics']
      eval_metrics.completed_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (episode_length * eval_first_state.reward.shape[0] /
                  (time.time() - t))
      avg_episode_length = (
          eval_metrics.completed_episodes_steps /
          eval_metrics.completed_episodes)
      metrics = dict(
          dict({
              f'eval/episode_{name}': value / eval_metrics.completed_episodes
              for name, value in eval_metrics.completed_episodes_metrics.items()
          }),
          **dict({
              'eval/completed_episodes': eval_metrics.completed_episodes,
              'eval/avg_episode_length': avg_episode_length,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime,
              'train/completed_episodes': it * num_envs,
              'train/params_norm': summary.get('params_norm', 0),
              'train/eval_scores_mean': summary.get('eval_scores_mean', 0),
              'train/eval_scores_std': summary.get('eval_scores_std', 0),
              'train/weights': summary.get('weights', 0),
          }))
      logging.info('Step %s metrics %s', num_process_env_steps, metrics)
      if progress_fn:
        progress_fn(num_process_env_steps, metrics)

    if num_process_env_steps > num_timesteps:
      break

    t = time.time()
    # optimization
    training_state, summary = es_one_epoch(state, training_state)
    # Don't override state with new state. For environments with variable
    # episode length we still want to start from a 'reset', not from where the
    # last run finished.
    jax.tree_map(lambda x: x.block_until_ready(), training_state)
    sps = (int(training_state.normalizer_params[0]) * action_repeat -
           num_process_env_steps) / (
               time.time() - t)
    training_walltime += time.time() - t

  inference = make_inference_fn(core_env.observation_size, core_env.action_size,
                                normalize_observations)
  params = training_state.normalizer_params, training_state.policy_params

  return (inference, params, metrics)


def make_es_model(parametric_action_distribution, obs_size):
  return networks.make_model(
      [32, 32, 32, 32, parametric_action_distribution.param_size], obs_size)


def make_inference_fn(observation_size, action_size, normalize_observations):
  """Creates params and inference function for the ES agent."""
  _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      observation_size, normalize_observations)
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_model = make_es_model(parametric_action_distribution, observation_size)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(policy_params, obs), key)
    return action

  return inference_fn
