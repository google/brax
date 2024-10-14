# Copyright 2024 The Brax Authors.
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

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ars import networks as ars_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  normalizer_params: running_statistics.RunningStatisticsState
  policy_params: Params
  num_env_steps: int


# TODO: Pass the network as argument.
def train(
    environment: Union[envs_v1.Env, envs.Env],
    wrap_env: bool = True,
    num_timesteps: int = 100,
    episode_length: int = 1000,
    action_repeat: int = 1,
    max_devices_per_host: Optional[int] = None,
    number_of_directions: int = 60,
    top_directions: int = 20,
    step_size: float = 0.015,
    num_eval_envs: int = 128,
    exploration_noise_std: float = 0.025,
    seed: int = 0,
    normalize_observations: bool = False,
    num_evals: int = 1,
    reward_shift: float = 0.0,
    network_factory: types.NetworkFactory[
        ars_networks.ARSNetwork
    ] = ars_networks.make_policy_network,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """ARS."""
  top_directions = min(top_directions, number_of_directions)
  num_envs = number_of_directions * 2  # noise + anti noise

  process_count = jax.process_count()
  if process_count > 1:
    raise ValueError('ES is not compatible with multiple hosts, '
                     'please use a single host device.')
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info('Local device count: %d, '
               'devices to be used count: %d', local_device_count,
               local_devices_to_use)

  num_env_steps_between_evals = num_timesteps // num_evals
  next_eval_step = num_timesteps - (num_evals - 1) * num_env_steps_between_evals

  key = jax.random.PRNGKey(seed)
  key, network_key, eval_key, rng_key = jax.random.split(key, 4)

  assert num_envs % local_devices_to_use == 0
  env = environment
  if wrap_env:
    if isinstance(env, envs.Env):
      wrap_for_training = envs.training.wrap
    else:
      wrap_for_training = envs_v1.wrappers.wrap_for_training

    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn,
          rng=jax.random.split(rng_key, num_envs // local_devices_to_use),
      )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  obs_size = env.observation_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  ars_network = network_factory(
      observation_size=obs_size,
      action_size=env.action_size,
      preprocess_observations_fn=normalize_fn)
  make_policy = ars_networks.make_inference_fn(ars_network)

  vmapped_policy = jax.vmap(ars_network.apply, in_axes=(None, 0, 0))

  def run_step(carry, unused_target_t):
    (env_state, policy_params, cumulative_reward, active_episode,
     normalizer_params) = carry
    obs = env_state.obs
    actions = vmapped_policy(normalizer_params, policy_params, obs)
    nstate = env.step(env_state, actions)
    cumulative_reward = cumulative_reward + (nstate.reward -
                                             reward_shift) * active_episode
    new_active_episode = active_episode * (1 - nstate.done)
    return (nstate, policy_params, cumulative_reward, new_active_episode,
            normalizer_params), (env_state.obs, active_episode)

  def run_episode(normalizer_params: running_statistics.NestedMeanStd,
                  params: Params, key: PRNGKey):
    reset_keys = jax.random.split(key, num_envs // local_devices_to_use)
    first_env_states = env.reset(reset_keys)
    cumulative_reward = first_env_states.reward
    active_episode = jnp.ones_like(cumulative_reward)
    (_, _, cumulative_reward, _, _), (obs, obs_weights) = jax.lax.scan(
        run_step, (first_env_states, params, cumulative_reward, active_episode,
                   normalizer_params), (),
        length=episode_length // action_repeat)
    return cumulative_reward, obs, obs_weights

  def add_noise(params: Params, key: PRNGKey) -> Tuple[Params, Params, Params]:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), params,
        jax.tree_util.tree_unflatten(treedef, all_keys))
    params_with_noise = jax.tree_util.tree_map(
        lambda g, n: g + n * exploration_noise_std, params, noise
    )
    params_with_anti_noise = jax.tree_util.tree_map(
        lambda g, n: g - n * exploration_noise_std, params, noise)
    return params_with_noise, params_with_anti_noise, noise

  prun_episode = jax.pmap(run_episode, in_axes=(None, 0, 0))

  @jax.jit
  def training_epoch(training_state: TrainingState,
                     key: PRNGKey) -> Tuple[TrainingState, Metrics]:
    params = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.expand_dims(x, axis=0), number_of_directions, axis=0
        ),
        training_state.policy_params,
    )
    key, key_noise, key_es_eval = jax.random.split(key, 3)
    # generate perturbations
    params_with_noise, params_with_anti_noise, noise = add_noise(
        params, key_noise)

    pparams = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        params_with_noise,
        params_with_anti_noise,
    )

    pparams = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (local_devices_to_use, -1) + x.shape[1:]),
        pparams)

    key_es_eval = jax.random.split(key_es_eval, local_devices_to_use)
    eval_scores, obs, obs_weights = prun_episode(
        training_state.normalizer_params, pparams, key_es_eval)

    obs = jnp.reshape(obs, (-1,) + obs.shape[2:])
    obs_weights = jnp.reshape(obs_weights, (-1,) + obs_weights.shape[2:])

    normalizer_params = running_statistics.update(
        training_state.normalizer_params, obs, weights=obs_weights)

    eval_scores = jnp.reshape(eval_scores, [-1])

    reward_plus, reward_minus = jnp.split(eval_scores, 2, axis=0)
    reward_max = jnp.maximum(reward_plus, reward_minus)
    reward_rank = jnp.argsort(jnp.argsort(-reward_max))
    reward_weight = jnp.where(reward_rank < top_directions, 1, 0)
    reward_weight_double = jnp.concatenate([reward_weight, reward_weight],
                                           axis=0)
    reward_std = jnp.std(eval_scores, where=reward_weight_double)
    reward_std += (reward_std == 0.0) * 1e-6

    noise = jax.tree_util.tree_map(
        lambda x: jnp.sum(
            jnp.transpose(
                jnp.transpose(x) * reward_weight * (reward_plus - reward_minus)
            ),
            axis=0,
        ),
        noise,
    )

    policy_params = jax.tree_util.tree_map(
        lambda x, y: x + step_size * y / (top_directions * reward_std),
        training_state.policy_params, noise)

    num_env_steps = training_state.num_env_steps + jnp.sum(
        obs_weights, dtype=jnp.int32) * action_repeat

    metrics = {
        'params_norm': optax.global_norm(policy_params),
        'eval_scores_mean': jnp.mean(eval_scores),
        'eval_scores_std': jnp.std(eval_scores),
        'weights': jnp.mean(reward_weight),
    }
    return (TrainingState(  # type: ignore  # jnp-type
        normalizer_params=normalizer_params,
        policy_params=policy_params,
        num_env_steps=num_env_steps), metrics)

  training_walltime = 0.

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(training_state: TrainingState,
                                 key: PRNGKey) -> Tuple[TrainingState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, metrics) = training_epoch(training_state, key)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_envs * episode_length) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  normalizer_params = running_statistics.init_state(
      specs.Array((obs_size,), jnp.dtype('float32')))
  policy_params = ars_network.init(network_key)
  training_state = TrainingState(
      normalizer_params=normalizer_params,
      policy_params=policy_params,
      num_env_steps=0)

  if not eval_env:
    eval_env = environment
  if wrap_env:
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
      )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  # Evaluator function
  evaluator = acting.Evaluator(
      eval_env,
      make_policy,
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  while training_state.num_env_steps < num_timesteps:
    # optimization
    key, epoch_key = jax.random.split(key)
    training_state, training_metrics = training_epoch_with_timing(
        training_state, epoch_key)

    if training_state.num_env_steps >= next_eval_step:
      # Run evals.
      metrics = evaluator.run_evaluation(
          (training_state.normalizer_params, training_state.policy_params),
          training_metrics)
      logging.info(metrics)
      progress_fn(int(training_state.num_env_steps), metrics)
      next_eval_step += num_env_steps_between_evals

  total_steps = training_state.num_env_steps
  assert total_steps >= num_timesteps

  logging.info('total steps: %s', total_steps)
  params = training_state.normalizer_params, training_state.policy_params
  return (make_policy, params, metrics)
