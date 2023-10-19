# Copyright 2023 The Brax Authors.
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
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    seed: int = 0,
    truncation_length: Optional[int] = None,
    max_gradient_norm: float = 1e9,
    num_evals: int = 1,
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

  if truncation_length is not None:
    assert truncation_length > 0

  num_evals_after_init = max(num_evals - 1, 1)

  assert num_envs % device_count == 0
  env = environment
  if isinstance(env, envs.Env):
    wrap_for_training = envs.training.wrap
  else:
    wrap_for_training = envs_v1.wrappers.wrap_for_training

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  rng, global_key = jax.random.split(global_key, 2)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, eval_key = jax.random.split(local_key)

  v_randomiation_fn = None
  if randomization_fn is not None:
    v_randomiation_fn = functools.partial(
        randomization_fn, rng=jax.random.split(rng, num_envs // process_count)
    )
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomiation_fn,
  )

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  apg_network = network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=normalize)
  make_policy = apg_networks.make_inference_fn(apg_network)

  optimizer = optax.adam(learning_rate=learning_rate)

  def env_step(
      carry: Tuple[Union[envs.State, envs_v1.State], PRNGKey],
      step_index: int,
      policy: types.Policy,
  ):
    env_state, key = carry
    key, key_sample = jax.random.split(key)
    actions = policy(env_state.obs, key_sample)[0]
    nstate = env.step(env_state, actions)
    if truncation_length is not None:
      nstate = jax.lax.cond(
          jnp.mod(step_index + 1, truncation_length) == 0.,
          jax.lax.stop_gradient, lambda x: x, nstate)

    return (nstate, key), (nstate.reward, env_state.obs)

  def loss(policy_params, normalizer_params, key):
    key_reset, key_scan = jax.random.split(key)
    env_state = env.reset(
        jax.random.split(key_reset, num_envs // process_count))
    f = functools.partial(
        env_step, policy=make_policy((normalizer_params, policy_params)))
    (rewards,
     obs) = jax.lax.scan(f, (env_state, key_scan),
                         (jnp.array(range(episode_length // action_repeat))))[1]
    return -jnp.mean(rewards), obs

  loss_grad = jax.grad(loss, has_aux=True)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    return jax.tree_util.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates)

  def training_epoch(training_state: TrainingState, key: PRNGKey):
    key, key_grad = jax.random.split(key)
    grad, obs = loss_grad(training_state.policy_params,
                          training_state.normalizer_params, key_grad)
    grad = clip_by_global_norm(grad)
    grad = jax.lax.pmean(grad, axis_name='i')
    params_update, optimizer_state = optimizer.update(
        grad, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params,
                                        params_update)

    normalizer_params = running_statistics.update(
        training_state.normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME)

    metrics = {
        'grad_norm': optax.global_norm(grad),
        'params_norm': optax.global_norm(policy_params)
    }
    return TrainingState(
        optimizer_state=optimizer_state,
        normalizer_params=normalizer_params,
        policy_params=policy_params), metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  training_walltime = 0

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
    sps = (episode_length * num_envs) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  # The network key should be global, so that networks are initialized the same
  # way for different processes.
  policy_params = apg_network.policy_network.init(global_key)
  del global_key

  training_state = TrainingState(
      optimizer_state=optimizer.init(policy_params),
      policy_params=policy_params,
      normalizer_params=running_statistics.init_state(
          specs.Array((env.observation_size,), jnp.dtype('float32'))))
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  if not eval_env:
    eval_env = environment
  if randomization_fn is not None:
    v_randomiation_fn = functools.partial(
        randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
    )
  eval_env = wrap_for_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomiation_fn,
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
            (training_state.normalizer_params, training_state.policy_params)),
        training_metrics={})
    logging.info(metrics)
    progress_fn(0, metrics)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    # optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state,
     training_metrics) = training_epoch_with_timing(training_state, epoch_keys)

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)),
          training_metrics)
      logging.info(metrics)
      progress_fn(it + 1, metrics)

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
