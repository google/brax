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

"""Evolution strategy training.

See: https://arxiv.org/pdf/1703.03864.pdf
"""

import time
from typing import Any, Callable, Dict, Optional

from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax

from brax import envs
from brax.training import distribution
from brax.training import env
from brax.training import networks
from brax.training import normalization


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  key: jnp.ndarray
  normalizer_params: Any
  optimizer: Any


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps: int = 100,
    episode_length: int = 1000,
    fitness_episode_length: int = 1000,
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
  epochs = 1 + num_timesteps // fitness_episode_length // num_envs
  if epochs %  log_frequency > 0:
    epochs += log_frequency - (epochs % log_frequency)

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
  key_envs = jax.random.split(key_env, local_devices_to_use // 2)
  tmp_env_states = []
  for key in key_envs:
    first_state, step_fn = env.wrap(core_env, key)
    tmp_env_states.append(first_state)
  first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                  *tmp_env_states)

  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = env.wrap(core_eval_env, key_eval)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)

  _, obs_size = eval_first_state.core.obs.shape

  policy_model = make_es_model(parametric_action_distribution, obs_size)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          obs_size, normalize_observations, num_leading_batch_dims=2))

  optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(policy_model.init(key_model))

  key_debug = jax.random.PRNGKey(seed + 666)

  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    logits = policy_model.apply(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params, normalizer_params):
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  @jax.vmap
  def training_inference(params, obs):
    return policy_model.apply(params, obs)

  def do_one_step(carry, unused_target_t):
    state, policy_params, key, cumulative_reward, normalizer_params = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    logits = training_inference(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = step_fn(state, actions)
    cumulative_reward = cumulative_reward + nstate.core.reward
    return (nstate, policy_params, key, cumulative_reward,
            normalizer_params), state.core.obs

  def run_es_eval(state, params, key, normalizer_params):
    cumulative_reward = jnp.zeros(state.core.obs.shape[0])
    (state, _, key, cumulative_reward, _), obs = jax.lax.scan(
        do_one_step, (state, params, key, cumulative_reward, normalizer_params),
        (), length=fitness_episode_length // action_repeat)
    average_per_step_reward = cumulative_reward / fitness_episode_length
    full_episode_reward = average_per_step_reward * episode_length
    return full_episode_reward, obs, state

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

  def es_one_epoch(carry, unused_t):
    state, training_state = carry
    params = jax.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, axis=0),
                             population_size, axis=0),
        training_state.optimizer.target)
    key, key_petr, key_es_eval = jax.random.split(training_state.key, 3)
    # generate perturbations
    params_with_noise, params_with_anti_noise, noise = add_noise(
        params, key_petr)

    pstate = jax.tree_map(lambda x: jnp.concatenate([x, x], axis=0), state)
    pparams = jax.tree_multimap(lambda a, b: jnp.concatenate([a, b], axis=0),
                                params_with_noise, params_with_anti_noise)

    pparams = jax.tree_map(
        lambda x: jnp.reshape(x, [local_devices_to_use, -1] + list(x.shape[1:])
                             ), pparams)

    prun_es_eval = jax.pmap(run_es_eval, in_axes=(0, 0, None, None))
    eval_scores, obs, state = prun_es_eval(
        pstate, pparams, key_es_eval,
        training_state.normalizer_params)

    state = jax.tree_map(lambda x: jnp.split(x, 2, axis=0)[0], state)

    obs = jnp.reshape(obs, [-1] + list(obs.shape[2:]))

    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, obs)

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
                              training_state.optimizer.target)
    optimizer = training_state.optimizer
    delta = jax.tree_map(lambda x: -x, delta)
    optimizer = optimizer.apply_gradient(delta)
    new_params = optimizer.target
    metrics = {
        'params_norm': optax.global_norm(new_params),
        'eval_scores_mean': jnp.mean(eval_scores),
        'eval_scores_std': jnp.std(eval_scores),
        'weights': jnp.mean(weights),
    }
    return (state,
            TrainingState(
                key=key,
                normalizer_params=normalizer_params,
                optimizer=optimizer)), metrics

  epochs_per_step = epochs // log_frequency
  @jax.jit
  def run_es(state, training_state):
    (state, training_state), metrics = jax.lax.scan(
        es_one_epoch, (state, training_state), (), length=epochs_per_step)
    return state, training_state, jax.tree_map(jnp.mean, metrics)

  training_state = TrainingState(key=key,
                                 normalizer_params=normalizer_params,
                                 optimizer=optimizer)

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
      eval_state, key_debug = (
          run_eval(eval_first_state, key_debug,
                   training_state.optimizer.target,
                   training_state.normalizer_params))
      eval_state.total_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (episode_length * eval_first_state.core.reward.shape[0] /
                  (time.time() - t))
      metrics = dict(
          dict({f'eval/episode_{name}': value / eval_state.total_episodes
                for name, value in eval_state.total_metrics.items()}),
          **dict({
              'eval/total_episodes': eval_state.total_episodes,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime,
              'train/params_norm': summary.get('params_norm', 0),
              'train/eval_scores_mean': summary.get('eval_scores_mean', 0),
              'train/eval_scores_std': summary.get('eval_scores_std', 0),
              'train/weights': summary.get('weights', 0),
          }))
      logging.info(metrics)
      if progress_fn:
        progress_fn(int(training_state.normalizer_params[0]) * action_repeat,
                    metrics)

    if it == log_frequency:
      break

    t = time.time()
    # optimization
    state, training_state, summary = run_es(state, training_state)
    jax.tree_map(lambda x: x.block_until_ready(), training_state)
    sps = fitness_episode_length * num_envs * epochs_per_step / (
        time.time() - t)
    training_walltime += time.time() - t

  _, inference = make_params_and_inference_fn(core_env.observation_size,
                                              core_env.action_size,
                                              normalize_observations)
  params = training_state.normalizer_params, training_state.optimizer.target

  return (inference, params, metrics)


def make_es_model(parametric_action_distribution, obs_size):
  return networks.make_model(
      [32, 32, 32, 32, parametric_action_distribution.param_size], obs_size)


def make_params_and_inference_fn(observation_size, action_size,
                                 normalize_observations):
  """Creates params and inference function for the ES agent."""
  obs_normalizer_params, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      normalize_observations)
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_model = make_es_model(parametric_action_distribution,
                               observation_size)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(policy_params, obs), key)
    return action

  params = (obs_normalizer_params, policy_model.init(jax.random.PRNGKey(0)))
  return params, inference_fn
