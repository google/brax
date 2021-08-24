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

"""Evaluators for VGRL."""

import functools
import itertools
from typing import Dict, Tuple, Any
from brax.io import file
from brax.io import html
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def visualize_skills(
    env_fn,
    inference_fn,
    g: Tuple[Any],
    params: Tuple[Dict[str, jnp.ndarray]],
    env_scale: float,
    algo_name: str,
    output_path: str = None,
    verbose: bool = True,
    num_samples_per_z: int = 5,
    time_subsampling: int = 10,
    time_last_n: int = 500,
    seed: int = 0,
    output_name: str = 'skills',
    save_video: bool = False,
):
  """Visualizing skills in a 2D plot."""
  env = env_fn()
  O = env.env_obs_size
  Z = env.z_size
  M = num_samples_per_z

  # Sample {D} z's [D, Z]
  if algo_name in ('fixed_gcrl',):  # D=1
    zs = jnp.ones((1, Z)) * env_scale
  elif algo_name in ('gcrl',):  # D=2^Z
    zs = jnp.array(list(itertools.product(*([[-1, 1]] * Z)))) * env_scale
  elif algo_name in ('cdiayn',):  # D=2^Z
    zs = jnp.array(list(itertools.product(*([[-1, 1]] * Z))))
  elif algo_name in ('diayn', 'diayn_full'):  # D=Z
    zs = jax.nn.one_hot(jnp.arange(0, Z), Z)
  else:
    raise NotImplementedError(algo_name)
  D = zs.shape[0]

  # Repeat each z by {M} times
  batch_z = jnp.repeat(zs, M, axis=0)  # [D*M, Z]

  # Reset and run environment
  batch_env = env_fn(batch_size=D * M)
  state = batch_env.reset(
      jnp.array([jax.random.PRNGKey(seed + i) for i in range(D * M)]),
      z=batch_z)
  states = [state]
  jit_step = jax.jit(batch_env.step)
  jit_inference_fn = jax.jit(inference_fn)
  while not state.done.all():
    act = jit_inference_fn(params, state.obs, state.rng[0])
    state = jit_step(state, act, params[0], params[-1])
    states.append(state)

  # Get env dimensions of interest
  obses = jnp.stack([state.obs for state in states],
                    axis=0)[-time_last_n:][::time_subsampling]  # [T, D*M, O+D]
  if verbose:
    print((f'Trajectory timesteps={len(states)}, '
           f'Plotted timesteps={obses.shape[0]}, O={O}, Z={Z}, D={D}, M={M}'))
  env_obses, _ = batch_env.disc.split_obs(obses)  # [T, D*M, O]
  env_vars = batch_env.disc.index_obs(env_obses)  # [T, D*M, 1 or 2]
  if env_vars.shape[-1] == 1:
    env_vars = jnp.concatenate([env_vars, jnp.zeros(env_vars.shape)], axis=-1)
  elif env_vars.shape[-1] > 2:
    env_vars = env_vars[..., :2]
  assert env_vars.shape[1:] == (D * M,
                                2), f'{env_vars.shape} incompatible {(D*M,2)}'
  env_vars = env_vars.reshape(-1, D, M,
                              2).swapaxes(1, 2).reshape(-1, D, 2)  # [T*M, D, 2]

  # Plot
  def spec(N):
    t = np.linspace(-510, 510, N)
    return np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255) / 255.

  colours = spec(D)  # [D, 3]
  _, axs = plt.subplots(ncols=1, figsize=(7, 6))
  labels = g
  if len(labels) == 1:
    labels += (None,)
  for i, (z, c) in enumerate(zip(zs, colours)):
    axs.scatter(
        x=env_vars[:, i, 0],
        y=env_vars[:, i, 1],
        c=c,
        alpha=0.3,
        label=f'z={z}')
    axs.legend()
    axs.set(xlabel=f'{labels[0]}', ylabel=f'{labels[1]}')
  if output_path:
    file.MakeDirs(output_path)
    with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
      plt.savefig(f)
    with file.File(f'{output_path}/{output_name}.npy', 'wb') as f:
      np.save(f, zs)
    if save_video:

      def index_fn(x, i):
        return x[i]

      for i in range(D):
        html.save_html(
            f'{output_path}/{output_name}_skill{i:02}.html',
            env.sys, [
                jax.tree_map(
                    functools.partial(index_fn, i=i * num_samples_per_z),
                    state.qp) for state in states
            ],
            make_dir=True)
  return states
