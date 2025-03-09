# Copyright 2025 The Brax Authors.
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


import functools
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from absl import logging
from etils import epath
import flax
import jax
import jax.numpy as jp
import numpy as np
import optax
from orbax import checkpoint as ocp

from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.bc import losses as bc_losses
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.ppo import train as ppo_train
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  """ 
  Optimisations:
  1) No scanning over trajectory collection. Just have a lot of parallel envs.
  2) Data never leaves the GPU. Only parameters flow in and out of it.
  """

  optimizer_state: optax.OptState
  params: Params
  normalizer_params: running_statistics.RunningStatisticsState
  dagger_it: jp.ndarray


Metrics = types.Metrics


def train(
    demo_length: int,
    teacher_inference_fn: bc_networks.BCInferenceFn,
    normalize_observations: bool = True,
    epochs: int = 50,
    num_demos: int = 1024,
    tanh_squash: bool = True,  # Greatly improves training stability.
    env: envs.Env = None,
    num_envs: int = 0,
    num_eval_envs: int = 0,
    eval_length: int = 80,
    batch_size: int = 256,
    scramble_time: int = 0,  # Maximum time to scramble the envs by.
    network_factory: types.NetworkFactory[
        bc_networks.BCNetworks
    ] = bc_networks.make_bc_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    madrona_backend: bool = False,
    seed: int = 0,
    learning_rate=4e-4,
    dagger_iterations: int = 1,
    dagger_beta_fn: Callable[[int], float] = lambda iter: jp.where(
        iter == 0, 1.0, 0.0
    ),
    num_evals=0,
    augment_pixels: bool = False,
    restore_checkpoint_path: Optional[str] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
):
  """Plain old behaviour cloning.
  Assumes your env is already wrapped.
  """
  if madrona_backend:
    if num_eval_envs and num_eval_envs != num_envs:
      raise ValueError('Madrona-MJX requires a fixed batch size')
    else:
      num_eval_envs = num_envs

  num_evals_after_init = max(num_evals - 1, 1)
  assert (
      dagger_iterations % num_evals_after_init == 0
  ), 'Dagger iterations must be divisible by num_evals - 1'
  dagger_iters_per_eval = dagger_iterations / num_evals_after_init

  key = jax.random.PRNGKey(seed)

  def scramble_times(key, state):
    state.info['steps'] = jax.random.randint(
        key, (num_envs,), minval=0, maxval=scramble_time
    )
    if '_steps' in state.info:  # Some envs have internal clocks.
      state.info['_steps'] = jp.array(state.info['steps'])
    return state

  assert num_demos == num_envs, 'Fast mode avoids scanning!'
  # Init the env state early to use for obs size calculations.
  key, key_reset, key_scramble = jax.random.split(key, 3)
  reset_fn = jax.jit(env.reset)
  env_state = reset_fn(jax.random.split(key_reset, num_envs))
  env_state = scramble_times(key_scramble, env_state)
  obs_shape = jax.tree_util.tree_map(
      lambda x: x.shape[1:], env_state.obs
  )  # Discard batch axis over envs.

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  bc_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = functools.partial(
      bc_networks.make_inference_fn(bc_network),
      deterministic=True,
      tanh_squash=tanh_squash,
  )
  key_policy, key = jax.random.split(key)
  optimizer = optax.adam(learning_rate=learning_rate)
  specs_obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jp.dtype('float32')), env_state.obs
  )
  init_params = bc_network.policy_network.init(key_policy)
  training_state = TrainingState(
      params=init_params,
      normalizer_params=running_statistics.init_state(
          ppo_train._remove_pixels(specs_obs_shape)
      ),
      optimizer_state=optimizer.init(
          init_params
      ),  # pytype: disable=wrong-arg-types  # numpy-scalars
      dagger_it=jp.array(0, dtype=int),
  )

  if (
      restore_checkpoint_path is not None
      and epath.Path(restore_checkpoint_path).exists()
  ):
    logging.info('restoring from checkpoint %s', restore_checkpoint_path)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    target = training_state.normalizer_params, init_params
    (normalizer_params, init_params) = orbax_checkpointer.restore(
        restore_checkpoint_path, item=target
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params, params=init_params
    )

  loss_fn = functools.partial(bc_losses.bc_loss, make_policy=make_policy)

  gradient_update_fn = jax.jit(
      gradients.gradient_update_fn(
          loss_fn, optimizer, pmap_axis_name=None, has_aux=True
      )
  )

  assert num_demos % num_envs == 0

  def collect_unroll(
      env_state: envs.State,
      key: PRNGKey,
      params: Tuple[Params, Params],
      beta: float,
  ) -> types.Observation:
    """Collect demo trajectory of fixed length."""

    def env_step(carry: Tuple[PRNGKey, Union[envs.State, envs_v1.State]], _):
      key, env_state = carry
      key, key_turn = jax.random.split(key)
      student_inference_fn = make_policy(params)
      teachers_turn = jax.random.bernoulli(key_turn, beta, shape=(num_envs, 1))
      actions = jp.where(
          teachers_turn,
          teacher_inference_fn(env_state.obs, None)[0],  # E x 14
          student_inference_fn(env_state.obs, None)[0],
      )
      nstate = env.step(env_state, actions)
      return (key, nstate), (env_state.obs, env_state.reward)  # E x ...

    key_step, key = jax.random.split(key)
    (_, env_state), (obs, reward) = jax.lax.scan(
        env_step, (key_step, env_state), length=demo_length
    )  # T x E x ...
    obs = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), obs)
    reward = jp.swapaxes(reward, 0, 1)
    assert list(obs.values())[0].shape[:2] == (num_envs, demo_length), obs.shape
    return env_state, obs, reward  # E x T

  def collect_data(
      env_state: envs.State,
      key: PRNGKey,
      params: Tuple[Params, Params],
      beta: float,
  ) -> types.Observation:
    """Collect a dataset for behavioural cloning.
    Builds out the dataset on host-side, assuming RAM > VRAM
    beta = probability of taking teacher action."""
    key, key_unroll = jax.random.split(key)
    env_state, all_obs, all_rewards = collect_unroll(
        env_state, key_unroll, params, beta
    )  # Tree(E x T x ...)

    for k in all_obs.keys():
      assert all_obs[k].shape[:2] == (num_demos, demo_length), (
          f'Expected shape {(num_demos, demo_length)} but got'
          f' {all_obs[k].shape[:2]}'
      )

    metrics = {
        'reward_mean': jp.mean(all_rewards, axis=0),
        'reward_std': jp.std(all_rewards, axis=0),
    }
    return env_state, all_obs, metrics

  def combine_batch_axes(tree):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), tree
    )

  assert (
      int(num_demos * demo_length) % batch_size == 0
  ), 'Demo trajectories must be evenly divisible into batches'
  num_minibatches = int(num_demos * demo_length / batch_size)

  @jax.vmap
  def random_crop(key, img):
    """Expects H W C image.
    Adapted from
    https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
    """
    padding = 4 if img.shape[-2] == 64 else 2
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jp.concatenate([crop_from, jp.zeros((1,), dtype=jp.int32)])
    padded_img = jp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode='edge'
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

  def fit_student(ts: TrainingState, X, key) -> Tuple[TrainingState, Metrics]:
    assert X['has_switched'].shape == (num_envs * demo_length, 1), (
        f'Expected shape {(num_envs * demo_length, 1)} but got'
        f' {X["has_switched"].shape}'
    )
    # Fixed through fitting process.
    normalizer_params = ts.normalizer_params

    def epoch(carry, _):
      params, optimizer_state, key = carry
      key, key_perm = jax.random.split(key)
      key, key_crop = jax.random.split(key)
      # Correlated crop between views for multi-pixel case.
      key_crop = jax.random.split(key_crop, int(num_envs * demo_length))

      shifted = X
      if augment_pixels:
        shifted = {}
        for k, v in X.items():
          if k.startswith('pixels/view'):
            shifted[k] = random_crop(key_crop, v)
        shifted = {**X, **shifted}  # shared keys are overwritten by second arg.

      def convert_data(x: jp.ndarray):
        x = jax.random.permutation(key_perm, x)
        x = jp.reshape(x, (num_minibatches, -1) + x.shape[1:])
        return x

      X_shuffled = jax.tree_util.tree_map(
          convert_data, shifted
      )  # num_minibatches, minibatch_size, ...
      assert X_shuffled['has_switched'].shape == (
          num_minibatches,
          int(num_demos * demo_length / num_minibatches),
          1,
      ), (
          'Expected shape'
          f' {(num_minibatches, int(num_demos * demo_length / num_minibatches), 1)} but'
          f' got {X_shuffled["has_switched"].shape}'
      )

      def minibatch_step(carry, X_batch):
        assert X_batch['has_switched'].shape == (
            int(num_demos * demo_length / num_minibatches),
            1,
        ), (
            'Expected shape'
            f' {(int(num_demos * demo_length / num_minibatches), 1)} but got'
            f' {X_batch["has_switched"].shape}'
        )
        params, optimizer_state = carry
        t_act, t_act_extras = teacher_inference_fn(X_batch, None)
        data = {
            'observations': X_batch,
            'teacher_action': t_act,
            'teacher_action_extras': t_act_extras,
        }
        # Apply loss grad fn
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            optimizer_state=optimizer_state,
        )
        return (params, optimizer_state), metrics

      (params, optimizer_state), metrics = jax.lax.scan(
          minibatch_step, (params, optimizer_state), X_shuffled
      )
      return (params, optimizer_state, key), metrics

    key, key_epoch = jax.random.split(key)
    (params, optimizer_state, _), metrics = jax.lax.scan(
        epoch, (ts.params, ts.optimizer_state, key_epoch), length=epochs
    )
    return ts.replace(optimizer_state=optimizer_state, params=params), metrics

  @jax.jit
  def dagger_iteration(
      carry: Tuple[PRNGKey, envs.State, TrainingState], _
  ) -> Tuple[Tuple[PRNGKey, TrainingState], Metrics]:
    # 1 dagger epoch.
    key, env_state, ts = carry
    key, key_data = jax.random.split(key)
    env_state, raw_data, data_metrics = collect_data(
        env_state,
        key_data,
        (ts.normalizer_params, ts.params),
        dagger_beta_fn(ts.dagger_it),
    )
    assert data_metrics['reward_mean'].shape == (demo_length,), (
        f'Expected shape {(demo_length,)} but got'
        f' {data_metrics["reward_mean"].shape}'
    )

    X_cur = combine_batch_axes(raw_data)
    assert X_cur['has_switched'].shape == (num_demos * demo_length, 1), (
        f'Expected shape {(num_demos * demo_length, 1)} but got'
        f' {X_cur["has_switched"].shape}'
    )
    ts = ts.replace(
        normalizer_params=running_statistics.update(
            ts.normalizer_params,
            ppo_train._remove_pixels(X_cur),
            pmap_axis_name=None,
        )
    )
    key, key_fit = jax.random.split(key)
    ts, metrics = fit_student(ts, X_cur, key_fit)
    metrics = {**data_metrics, **metrics}
    return (key, env_state, ts.replace(dagger_it=ts.dagger_it + 1)), metrics

  def evaluate(training_state: TrainingState, other_metrics: Metrics):
    inference_params = (training_state.normalizer_params, training_state.params)
    if num_evals:
      eval_metrics = evaluator.run_evaluation(
          inference_params,
          training_metrics={},
      )
      other_metrics.update(eval_metrics)
    policy_params_fn(training_state.dagger_it, make_policy, inference_params)
    progress_fn(training_state.dagger_it, other_metrics)

  if num_evals:
    eval_env = env
    Evaluator = acting.Evaluator
    key, eval_key = jax.random.split(key)
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy),
        num_eval_envs=num_eval_envs,
        episode_length=eval_length,
        action_repeat=1,
        key=eval_key,
    )
    evaluate(training_state, {})

  wall_time = 0
  key, key_dagger = jax.random.split(key)
  for i_eval in range(num_evals_after_init):
    if i_eval != 0:
      # Seed new randomness to be gradually phased in.
      key_reset, key = jax.random.split(key)
      _state = reset_fn(jax.random.split(key_reset, num_envs))
      env_state.info['first_obs'] = _state.obs
      env_state.info['first_state'] = (
          _state.data
      )  # Only playground mjx_env supported!
    t0 = time.monotonic()
    (key_dagger, env_state, training_state), b_metrics = jax.lax.scan(
        dagger_iteration,
        (key_dagger, env_state, training_state),
        length=dagger_iters_per_eval,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), b_metrics)
    t1 = time.monotonic()
    d_length = int(num_demos * demo_length)
    b_metrics['SPS'] = d_length * epochs / (t1 - t0)
    wall_time += t1 - t0
    b_metrics['walltime'] = wall_time
    evaluate(training_state, b_metrics)

  return (training_state.normalizer_params, training_state.params)
