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

"""Analytic policy gradient training."""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  normalizer_params: running_statistics.RunningStatisticsState
  policy_params: Params


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    episode_length: int,
    policy_updates: int,
    wrap_env: bool = True,
    horizon_length: int = 32,
    num_envs: int = 1,
    num_evals: int = 1,
    action_repeat: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    adam_b: tuple[float, float] = (0.7, 0.95),
    use_schedule: bool = True,
    use_float64: bool = False,
    schedule_decay: float = 0.997,
    seed: int = 0,
    max_gradient_norm: float = 1e9,
    normalize_observations: bool = False,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        apg_networks.APGNetworks
    ] = apg_networks.make_apg_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
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
  device_count = local_devices_to_use * process_count

  num_updates = policy_updates
  num_evals_after_init = max(num_evals - 1, 1)
  updates_per_epoch = jnp.round(num_updates / (num_evals_after_init))

  assert num_envs % device_count == 0
  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  rng, global_key = jax.random.split(global_key, 2)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, eval_key = jax.random.split(local_key)

  env = environment
  if wrap_env:
    if isinstance(env, envs.Env):
      wrap_for_training = envs.training.wrap
    else:
      wrap_for_training = envs_v1.wrappers.wrap_for_training

    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn, rng=jax.random.split(rng, num_envs // process_count)
      )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  reset_fn = jax.jit(jax.vmap(env.reset))
  step_fn = jax.jit(jax.vmap(env.step))

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  apg_network = network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=normalize)
  make_policy = apg_networks.make_inference_fn(apg_network)

  if use_schedule:
    learning_rate = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1,
        decay_rate=schedule_decay
    )

  optimizer = optax.chain(
      optax.clip(1.0),
      optax.adam(learning_rate=learning_rate, b1=adam_b[0], b2=adam_b[1])
  )

  def scramble_times(state, key):
    state.info['steps'] = jnp.round(
      jax.random.uniform(key, (local_devices_to_use, num_envs,),
                         maxval=episode_length))
    return state

  def env_step(
      carry: Tuple[Union[envs.State, envs_v1.State], PRNGKey],
      step_index: int,
      policy: types.Policy,
  ):
    env_state, key = carry
    key, key_sample = jax.random.split(key)
    actions = policy(env_state.obs, key_sample)[0]
    nstate = env.step(env_state, actions)

    return (nstate, key), (nstate.reward, env_state.obs)

  def loss(policy_params, normalizer_params, env_state, key):
    f = functools.partial(
        env_step, policy=make_policy((normalizer_params, policy_params))
    )
    (state_h, _), (rewards, obs) = jax.lax.scan(
        f, (env_state, key), (jnp.arange(horizon_length // action_repeat))
    )

    return -jnp.mean(rewards), (obs, state_h)

  loss_grad = jax.grad(loss, has_aux=True)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    return jax.tree_util.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates,
    )

  def minibatch_step(carry, epoch_step_index: int):
    (optimizer_state, normalizer_params, policy_params, key, state) = carry

    key, key_grad = jax.random.split(key)
    grad, (obs, state_h) = loss_grad(
        policy_params, normalizer_params, state, key_grad
    )

    grad = clip_by_global_norm(grad)
    grad = jax.lax.pmean(grad, axis_name='i')
    params_update, optimizer_state = optimizer.update(grad, optimizer_state)
    policy_params = optax.apply_updates(policy_params, params_update)

    normalizer_params = running_statistics.update(
        normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME
    )

    metrics = {
        'grad_norm': optax.global_norm(grad),
        'params_norm': optax.global_norm(policy_params),
    }

    return (
        optimizer_state,
        normalizer_params,
        policy_params,
        key,
        state_h,
    ), metrics

  def training_epoch(
      training_state: TrainingState,
      env_state: Union[envs.State, envs_v1.State],
      key: PRNGKey,
  ):

    (
        optimizer_state,
        normalizer_params,
        policy_params,
        key,
        state_h,
    ), metrics = jax.lax.scan(
        minibatch_step,
        (
            training_state.optimizer_state,
            training_state.normalizer_params,
            training_state.policy_params,
            key,
            env_state,
        ),
        jnp.arange(updates_per_epoch),
    )

    return (
        TrainingState(
            optimizer_state=optimizer_state,
            normalizer_params=normalizer_params,
            policy_params=policy_params,
        ),
        state_h,
        metrics,
        key,
    )

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  training_walltime = 0

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: Union[envs.State, envs_v1.State],
      key: PRNGKey,
  ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], Metrics, PRNGKey]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, metrics, key) = training_epoch(
        training_state, env_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (updates_per_epoch * num_envs * horizon_length) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, metrics, key  # pytype: disable=bad-return-type  # py311-upgrade

  # The network key should be global, so that networks are initialized the same
  # way for different processes.
  policy_params = apg_network.policy_network.init(global_key)
  del global_key

  dtype = 'float64' if use_float64 else 'float32'
  training_state = TrainingState(
      optimizer_state=optimizer.init(policy_params),
      policy_params=policy_params,
      normalizer_params=running_statistics.init_state(
          specs.Array((env.observation_size,), jnp.dtype(dtype))))
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

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

  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  metrics = {}

  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)
        ),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  init_key, scramble_key, local_key = jax.random.split(local_key, 3)
  init_key = jax.random.split(
      init_key, (local_devices_to_use, num_envs // process_count)
  )
  env_state = reset_fn(init_key)
  # TODO: this may be better off as an env wrapper
  env_state = scramble_times(env_state, scramble_key)
  env_state = step_fn(
      env_state,
      jnp.zeros(
          (local_devices_to_use, num_envs // process_count, env.action_size)
      ),
  )  # Prevent recompilation on the second epoch

  epoch_key, local_key = jax.random.split(local_key)
  epoch_key = jax.random.split(epoch_key, local_devices_to_use)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    (training_state, env_state, training_metrics, epoch_key) = (
        training_epoch_with_timing(training_state, env_state, epoch_key)
    )

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)
          ),
          training_metrics,
      )
      logging.info(metrics)
      progress_fn(it + 1, metrics)

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
