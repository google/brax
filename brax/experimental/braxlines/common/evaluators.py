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

"""Evaluators."""
import functools
import os
from typing import Dict, Tuple, Any
from brax.io import html
import jax
from jax import numpy as jnp


def visualize_env(
    env_fn,
    inference_fn,
    params: Dict[str, jnp.ndarray],
    batch_size: int = 0,
    seed: int = 0,
    reset_args: Tuple[Any] = (),
    step_args: Tuple[Any] = (),
    output_path: str = None,
    output_name: str = 'video',
):
  """Visualize environment."""
  rng = jax.random.PRNGKey(seed=seed)
  rng, reset_key = jax.random.split(rng)
  if batch_size:
    reset_key = jnp.stack(jax.random.split(reset_key, batch_size))
  env = env_fn(batch_size=batch_size)
  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(inference_fn)
  qps = []
  states = []
  state = jit_env_reset(reset_key, *reset_args)
  while not jnp.all(state.done):
    qps.append(state.qp)
    states.append(state)
    tmp_key, rng = jax.random.split(rng)
    act = jit_inference_fn(params, state.obs, tmp_key)
    state = jit_env_step(state, act, *step_args)
  if output_path:
    output_name = os.path.splitext(output_name)[0]
    if batch_size:

      def index_fn(x, i):
        return x[i]

      for i in range(batch_size):
        html.save_html(
            f'{output_path}/{output_name}_eps{i:02}.html',
            env.sys,
            [jax.tree_map(functools.partial(index_fn, i=i), qp) for qp in qps],
            make_dir=True)
    else:
      html.save_html(
          f'{output_path}/{output_name}.html', env.sys, qps, make_dir=True)
  return states
