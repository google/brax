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

"""Evaluators."""
import functools
import os
from typing import Dict, Tuple, Any
from brax.experimental.braxlines.training import utils as training_utils
from brax.experimental.braxlines.training.env import wrap
from brax.io import html
import jax
from jax import numpy as jnp


def init_run_eval(env_fn,
                  action_repeat: int,
                  batch_size: int,
                  episode_length: int = 1000,
                  seed: int = 0):
  """Initialize run_eval."""
  key_eval = jax.random.PRNGKey(seed=seed)
  core_eval_env = env_fn(
      action_repeat=action_repeat,
      batch_size=batch_size,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = wrap(core_eval_env, key_eval)
  return core_eval_env, eval_first_state, eval_step_fn, key_eval


def jit_run_eval(env_fn,
                 inference_fn,
                 action_repeat: int,
                 batch_size: int,
                 episode_length: int = 1000,
                 seed: int = 0):
  """JIT run_eval."""
  core_eval_env, eval_first_state, eval_step_fn, key_eval = init_run_eval(
      env_fn=env_fn,
      seed=seed,
      action_repeat=action_repeat,
      batch_size=batch_size,
      episode_length=episode_length)
  jit_inference_fn = jax.jit(inference_fn)

  def do_one_step_eval(carry, unused_target_t):
    state, params, key = carry
    key, key_sample = jax.random.split(key)
    actions = jit_inference_fn(params, state.core.obs, key_sample)
    nstate = eval_step_fn(state, actions, params['normalizer'], params['extra'])
    return (nstate, params, key), nstate.core

  @jax.jit
  def jit_run_eval_fn(state, key, params):
    (final_state, _, key), states = jax.lax.scan(
        do_one_step_eval, (state, params, key), (),
        length=episode_length // action_repeat)
    return final_state, key, states

  return core_eval_env, eval_first_state, key_eval, jit_run_eval_fn


def rollout_env(
    params: Dict[str, Dict[str, jnp.ndarray]],
    env_fn,
    inference_fn,
    batch_size: int = 0,
    seed: int = 0,
    reset_args: Tuple[Any] = (),
    step_args: Tuple[Any] = (),
    step_fn_name: str = 'step',
):
  """Visualize environment."""
  rng = jax.random.PRNGKey(seed=seed)
  rng, reset_key = jax.random.split(rng)
  env = env_fn(batch_size=batch_size)
  inference_fn = inference_fn or functools.partial(
      training_utils.zero_fn, action_size=env.action_size)
  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(getattr(env, step_fn_name))
  jit_inference_fn = jax.jit(inference_fn)
  states = []
  state = jit_env_reset(reset_key, *reset_args)
  while not jnp.all(state.done):
    states.append(state)
    tmp_key, rng = jax.random.split(rng)
    act = jit_inference_fn(params, state.obs, tmp_key)
    state = jit_env_step(state, act, *step_args)
  states.append(state)
  return env, states


def visualize_env(batch_size: int = 0,
                  output_path: str = None,
                  output_name: str = 'video',
                  verbose: bool = False,
                  **kwargs):
  """Visualize env."""
  env, states = rollout_env(batch_size=batch_size, **kwargs)
  if verbose:
    print(f'Collected {max(1, batch_size)} trajs of T={len(states)}')

  if output_path:
    output_name = os.path.splitext(output_name)[0]
    if batch_size:

      for i in range(batch_size):
        filename = f'{output_path}/{output_name}_eps{i:02}.html'
        html.save_html(
            filename,
            env.sys, [
                jax.tree_map(functools.partial(jnp.take, indices=i), state.qp)
                for state in states
            ],
            make_dir=True)
        if verbose:
          print(f'Saved {filename}')
    else:
      filename = f'{output_path}/{output_name}.html'
      html.save_html(
          filename, env.sys, [state.qp for state in states], make_dir=True)
      if verbose:
        print(f'Saved {filename}')

  return env, states
