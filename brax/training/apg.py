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

"""Analytic policy gradient training."""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from brax import envs
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training import pmap
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
from flax import linen
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
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    max_gradient_norm: float = 1e9,
    max_devices_per_host: Optional[int] = None,
    learning_rate=1e-4,
    normalize_observations=False,
    seed=0,
    log_frequency=10,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    truncation_length: Optional[int] = None,
):
  """Direct trajectory optimization training."""
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)

  key = jax.random.PRNGKey(seed)
  key, key_models, key_env = jax.random.split(key, 3)

  key_env = jax.random.split(key_env, process_count)[process_id]
  key = jax.random.split(key, process_count)[process_id]

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
  eval_step_fn = jax.jit(eval_env.step)
  eval_first_state = jax.jit(eval_env.reset)(key_env)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)

  policy_model = make_direct_optimization_model(parametric_action_distribution,
                                                core_env.observation_size)

  policy_params = policy_model.init(key_models)
  optimizer = optax.adam(learning_rate=learning_rate)
  optimizer_state = optimizer.init(policy_params)
  optimizer_state, policy_params = pmap.bcast_local_devices(
      (optimizer_state, policy_params), local_devices_to_use)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          core_env.observation_size,
          normalize_observations,
          num_leading_batch_dims=2,
          pmap_to_devices=local_devices_to_use))

  key_debug = jax.random.PRNGKey(seed + 666)

  def do_one_step_eval(carry, unused_target_t):
    state, params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    # TODO: Make this nicer ([0] comes from pmapping).
    obs = obs_normalizer_apply_fn(
        jax.tree_map(lambda x: x[0], normalizer_params), state.obs)
    print(obs.shape)
    print(jax.tree_map(lambda x: x.shape, params))
    logits = policy_model.apply(params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions)
    return (nstate, params, normalizer_params, key), ()

  @jax.jit
  def run_eval(params, state, normalizer_params,
               key) -> Tuple[envs.State, PRNGKey]:
    params = jax.tree_map(lambda x: x[0], params)
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def do_one_step(carry, step_index):
    state, params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
    logits = policy_model.apply(params, normalized_obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = step_fn(state, actions)
    if truncation_length is not None and truncation_length > 0:
      nstate = jax.lax.cond(
          jnp.mod(step_index + 1, truncation_length) == 0.,
          jax.lax.stop_gradient, lambda x: x, nstate)

    return (nstate, params, normalizer_params, key), (nstate.reward,
                                                      state.obs)

  def loss(params, normalizer_params, state, key):
    _, (rewards, obs) = jax.lax.scan(
        do_one_step, (state, params, normalizer_params, key),
        (jnp.array(range(episode_length // action_repeat))),
        length=episode_length // action_repeat)
    normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)
    return -jnp.mean(rewards), normalizer_params

  loss_grad = jax.grad(loss, has_aux=True)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    updates = jax.tree_multimap(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates)
    return updates

  def _minimize(training_state: TrainingState, state: envs.State):
    synchro = pmap.is_replicated(
        (training_state.optimizer_state, training_state.policy_params,
         training_state.normalizer_params),
        axis_name='i')
    key, key_grad = jax.random.split(training_state.key)
    grad, normalizer_params = loss_grad(training_state.policy_params,
                                        training_state.normalizer_params,
                                        state, key_grad)
    grad = clip_by_global_norm(grad)
    grad = jax.lax.pmean(grad, axis_name='i')
    params_update, optimizer_state = optimizer.update(
        grad, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params,
                                        params_update)

    metrics = {
        'grad_norm': optax.global_norm(grad),
        'params_norm': optax.global_norm(policy_params)
    }
    return TrainingState(
        key=key,
        optimizer_state=optimizer_state,
        normalizer_params=normalizer_params,
        policy_params=policy_params), metrics, synchro

  minimize = jax.pmap(_minimize, axis_name='i')

  logging.info('Available devices %s', jax.devices())
  training_walltime = 0
  sps = 0
  eval_sps = 0
  summary = {
      'params_norm':
          optax.global_norm(jax.tree_map(lambda x: x[0], policy_params))
  }
  key = jnp.stack(jax.random.split(key, local_devices_to_use))
  training_state = TrainingState(key=key,
                                 optimizer_state=optimizer_state,
                                 normalizer_params=normalizer_params,
                                 policy_params=policy_params)

  for it in range(log_frequency + 1):
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    if process_id == 0:
      eval_state, key_debug = run_eval(training_state.policy_params,
                                       eval_first_state,
                                       training_state.normalizer_params,
                                       key_debug)
      eval_metrics = eval_state.info['eval_metrics']
      eval_metrics.completed_episodes.block_until_ready()
      eval_sps = (
          episode_length * eval_first_state.reward.shape[0] /
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
              'speed/timestamp': training_walltime,
              'train/grad_norm': jnp.mean(summary.get('grad_norm', 0)),
              'train/params_norm': jnp.mean(summary.get('params_norm', 0)),
          }))

      logging.info(metrics)
      if progress_fn:
        progress_fn(it, metrics)

    if it == log_frequency:
      break

    t = time.time()
    # optimization
    training_state, summary, synchro = minimize(training_state, first_state)
    assert synchro[0], (it, training_state)
    jax.tree_map(lambda x: x.block_until_ready(), summary)
    sps = (episode_length * num_envs) / (time.time() - t)
    training_walltime += time.time() - t

  params = jax.tree_map(lambda x: x[0], training_state.policy_params)
  normalizer_params = jax.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  params = normalizer_params, params
  inference = make_inference_fn(core_env.observation_size, core_env.action_size,
                                normalize_observations)

  pmap.synchronize_hosts()
  return (inference, params, metrics)


def make_direct_optimization_model(parametric_action_distribution, obs_size):
  return networks.make_model(
      [32, 32, 32, 32, parametric_action_distribution.param_size],
      obs_size,
      activation=linen.swish)


def make_inference_fn(observation_size, action_size, normalize_observations):
  """Creates params and inference function for the direct optimization agent."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      observation_size, normalize_observations)
  policy_model = make_direct_optimization_model(parametric_action_distribution,
                                                observation_size)

  def inference_fn(params, obs, key):
    normalizer_params, params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(params, obs), key)
    return action

  return inference_fn
