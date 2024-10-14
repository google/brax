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

"""Evolution strategy training.

See: https://arxiv.org/pdf/1703.03864.pdf
"""

import enum
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
from brax.training.agents.es import networks as es_networks
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
  optimizer_state: optax.OptState
  policy_params: Params
  num_env_steps: int


# Centered rank from: https://arxiv.org/pdf/1703.03864.pdf
def centered_rank(x: jnp.ndarray) -> jnp.ndarray:
  x = jnp.argsort(jnp.argsort(x))
  x /= (len(x) - 1)
  return x - .5


# Shaping from
# https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
def wierstra(x: jnp.ndarray) -> jnp.ndarray:
  x = len(x) - jnp.argsort(jnp.argsort(x))
  x = jnp.maximum(0, jnp.log(len(x) / 2.0 + 1) - jnp.log(x))
  return x / jnp.sum(x) - 1.0 / len(x)


class FitnessShaping(enum.Enum):
  ORIGINAL = functools.partial(lambda x: x)
  CENTERED_RANK = functools.partial(centered_rank)
  WIERSTRA = functools.partial(wierstra)


# TODO: Pass the network as argument.
def train(
    environment: Union[envs_v1.Env, envs.Env],
    wrap_env: bool = True,
    num_timesteps: int = 100,
    episode_length: int = 1000,
    action_repeat: int = 1,
    l2coeff: float = 0,
    max_devices_per_host: Optional[int] = None,
    population_size: int = 128,
    learning_rate: float = 1e-3,
    fitness_shaping: FitnessShaping = FitnessShaping.ORIGINAL,
    num_eval_envs: int = 128,
    perturbation_std: float = 0.1,
    seed: int = 0,
    normalize_observations: bool = False,
    num_evals: int = 1,
    center_fitness: bool = False,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        es_networks.ESNetworks
    ] = es_networks.make_es_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """ES training (from https://arxiv.org/pdf/1703.03864.pdf)."""
  num_envs = population_size * 2  # noise + anti noise

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

  num_evals_after_init = max(num_evals - 1, 1)

  num_env_steps_between_evals = num_timesteps // num_evals_after_init
  next_eval_step = num_timesteps - (num_evals_after_init -
                                    1) * num_env_steps_between_evals

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
  es_network = network_factory(
      observation_size=obs_size,
      action_size=env.action_size,
      preprocess_observations_fn=normalize_fn)
  make_policy = es_networks.make_inference_fn(es_network)

  optimizer = optax.adam(learning_rate=learning_rate)

  vmapped_policy = jax.vmap(
      es_network.policy_network.apply, in_axes=(None, 0, 0))

  def run_step(carry, unused_target_t):
    (env_state, policy_params, key, cumulative_reward, active_episode,
     normalizer_params) = carry
    key, key_sample = jax.random.split(key)
    obs = env_state.obs
    logits = vmapped_policy(normalizer_params, policy_params, obs)
    actions = es_network.parametric_action_distribution.sample(
        logits, key_sample)
    nstate = env.step(env_state, actions)
    cumulative_reward = cumulative_reward + nstate.reward * active_episode
    new_active_episode = active_episode * (1 - nstate.done)
    return (nstate, policy_params, key, cumulative_reward, new_active_episode,
            normalizer_params), (env_state.obs, active_episode)

  def run_episode(normalizer_params: running_statistics.NestedMeanStd,
                  params: Params, key: PRNGKey):
    key_scan, key_reset = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, num_envs // local_devices_to_use)
    first_env_states = env.reset(reset_keys)
    cumulative_reward = first_env_states.reward
    active_episode = jnp.ones_like(cumulative_reward)
    (_, _, key, cumulative_reward, _, _), (obs, obs_weights) = jax.lax.scan(
        run_step, (first_env_states, params, key_scan, cumulative_reward,
                   active_episode, normalizer_params), (),
        length=episode_length // action_repeat)
    return cumulative_reward, obs, obs_weights

  def add_noise(params: Params, key: PRNGKey) -> Tuple[Params, Params, Params]:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), params,
        jax.tree_util.tree_unflatten(treedef, all_keys))
    params_with_noise = jax.tree_util.tree_map(lambda g, n: g + n * perturbation_std,
                                     params, noise)
    params_with_anti_noise = jax.tree_util.tree_map(lambda g, n: g - n * perturbation_std,
                                          params, noise)
    return params_with_noise, params_with_anti_noise, noise

  prun_episode = jax.pmap(run_episode, in_axes=(None, 0, 0))

  def compute_delta(
      params: jnp.ndarray,
      noise: jnp.ndarray,
      weights: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute the delta, i.e.

    the update to be passed to the optimizer.

    Args:
      params: Policy parameter leaf.
      noise: Noise leaf, with dimensions (population_size,) + params.shape
      weights: Fitness weights, vector of length population_size.

    Returns:

    """
    # NOTE: The trick "len(weights) -> len(weights) * perturbation_std" is
    # equivalent to tuning the l2_coef.
    weights = jnp.reshape(weights, ([population_size] + [1] * (noise.ndim - 1)))
    delta = jnp.sum(noise * weights, axis=0) / population_size
    # l2coeff controls the weight decay of the parameters of our policy network.
    # This prevents the parameters from growing very large compared to the
    # perturbations.
    delta = delta - l2coeff * params
    # Return -delta because the optimizer is set up to go against the gradient.
    return -delta

  @jax.jit
  def training_epoch(training_state: TrainingState,
                     key: PRNGKey) -> Tuple[TrainingState, Metrics]:
    params = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.expand_dims(x, axis=0), population_size, axis=0),
        training_state.policy_params)
    key, key_noise, key_es_eval = jax.random.split(key, 3)
    # generate perturbations
    params_with_noise, params_with_anti_noise, noise = add_noise(
        params, key_noise)

    pparams = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0),
                           params_with_noise, params_with_anti_noise)

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

    weights = jnp.reshape(eval_scores, [-1])

    weights = fitness_shaping.value(weights)

    if center_fitness:
      weights = (weights - jnp.mean(weights)) / (1E-6 + jnp.std(weights))

    weights1, weights2 = jnp.split(weights, 2)
    weights = weights1 - weights2

    delta = jax.tree_util.tree_map(
        functools.partial(compute_delta, weights=weights),
        training_state.policy_params, noise)

    params_update, optimizer_state = optimizer.update(
        delta, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params,
                                        params_update)

    num_env_steps = training_state.num_env_steps + jnp.sum(
        obs_weights, dtype=jnp.int32) * action_repeat

    metrics = {
        'params_norm': optax.global_norm(policy_params),
        'eval_scores_mean': jnp.mean(eval_scores),
        'eval_scores_std': jnp.std(eval_scores),
        'weights': jnp.mean(weights),
    }
    return (TrainingState(  # type: ignore  # jnp-type
        normalizer_params=normalizer_params,
        optimizer_state=optimizer_state,
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
  policy_params = es_network.policy_network.init(network_key)
  optimizer_state = optimizer.init(policy_params)
  training_state = TrainingState(
      normalizer_params=normalizer_params,
      optimizer_state=optimizer_state,
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
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  if num_evals > 1:
    metrics = evaluator.run_evaluation(
        (training_state.normalizer_params, training_state.policy_params),
        training_metrics={})
    logging.info(metrics)
    progress_fn(0, metrics)

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
