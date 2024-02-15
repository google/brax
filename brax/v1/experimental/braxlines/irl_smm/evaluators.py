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

"""Evaluators."""

from typing import Dict, Tuple

from brax.v1.experimental.braxlines.common import evaluators
from brax.v1.experimental.braxlines.irl_smm import utils as irl_utils
from brax.v1.io import file
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt


def visualize_disc(
    params: Dict[str, Dict[str, jnp.ndarray]],
    disc: irl_utils.IRLDiscriminator,
    num_grid=25,
    fig=None,
    axs=None,
    figsize: Tuple[float] = (3.5, 3),
    output_path: str = None,
    output_name: str = 'smm',
):
  """Visualize discriminator."""
  xgrid = jnp.linspace(-disc.obs_scale_2d[0], disc.obs_scale_2d[0], num_grid)
  ygrid = jnp.linspace(-disc.obs_scale_2d[1], disc.obs_scale_2d[1], num_grid)
  xgrid, ygrid = jnp.meshgrid(xgrid, ygrid)
  datagrid = jnp.concatenate(
      [xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=-1)

  # plot discriminator visualization
  distgrid = disc.dist(
      datagrid[..., :len(disc.obs_indices)], params=params['extra'])
  probsgrid = jax.nn.sigmoid(distgrid.logits)
  colors = jnp.clip(jnp.array([[-2, 0, 2]]) * (probsgrid - 0.5), a_min=0)
  if fig is None or axs is None:
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    axs = [ax]
  axs[-1].scatter(x=datagrid[:, 0], y=datagrid[:, 1], c=colors)
  axs[-1].set_xlim([-disc.obs_scale_2d[0], disc.obs_scale_2d[0]])
  axs[-1].set_ylim([-disc.obs_scale_2d[1], disc.obs_scale_2d[1]])
  axs[-1].set(xlabel=disc.obs_labels_2d[0], ylabel=disc.obs_labels_2d[1])
  axs[-1].set(title='discriminator output (red=0, black=0.5, blue=1)')
  fig.tight_layout()
  if output_path:
    file.MakeDirs(output_path)
    with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
      plt.savefig(f)


def estimate_energy_distance_metric(
    params: Dict[str, Dict[str, jnp.ndarray]],
    disc: irl_utils.IRLDiscriminator,
    target_data: jnp.ndarray,
    env_fn,
    inference_fn=None,
    num_samples: int = 10,
    time_subsampling: int = 10,
    time_last_n: int = 500,
    seed: int = 0,
    visualize: bool = False,
    figsize: Tuple[float] = (3.5, 3),
    output_path: str = None,
    output_name: str = 'smm_dist',
):
  """Estimate energy distance and (optimally) visualize smm."""
  batch_env, states = evaluators.rollout_env(
      env_fn=env_fn,
      params=params,
      batch_size=num_samples,
      inference_fn=inference_fn,
      step_args=(params['normalizer'], params['extra']),
      seed=seed,
  )
  # M = batch_size
  I = disc.indexed_obs_size

  # Get indices of interest
  obses_full = jnp.stack([state.obs for state in states], axis=0)
  obses = obses_full[-time_last_n:][::time_subsampling]
  env_vars = batch_env.disc.index_obs(obses)  # [T, M, I]
  target_vars = target_data
  env_vars_flat = env_vars.reshape(-1, I)
  target_vars_flat = target_vars.reshape(-1, I)

  # Compute energy distance
  norm = jnp.linalg.norm
  ee = norm(env_vars_flat[None] - env_vars_flat[:, None], axis=-1).mean()
  tt = norm(target_vars_flat[None] - target_vars_flat[:, None], axis=-1).mean()
  et = norm(target_vars_flat[None] - env_vars_flat[:, None], axis=-1).mean()
  energy_dist = 2 * et - ee - tt
  metrics = {'metrics/energy_dist': energy_dist}

  # Plot
  if visualize:
    env_vars = irl_utils.make_2d(env_vars)
    target_vars = irl_utils.make_2d(target_vars)
    fig, axs = plt.subplots(ncols=2, figsize=(figsize[0] * 2, figsize[1]))
    axs[0].set(title='agent policy')
    axs[0].set_xlim([-disc.obs_scale_2d[0], disc.obs_scale_2d[0]])
    axs[0].set_ylim([-disc.obs_scale_2d[1], disc.obs_scale_2d[1]])
    axs[0].set(xlabel=disc.obs_labels_2d[0], ylabel=disc.obs_labels_2d[1])
    axs[0].scatter(
        x=env_vars_flat[:, 0], y=env_vars_flat[:, 1], c=[1, 0, 0], alpha=0.3)
    axs[1].set(title='target')
    axs[1].set_xlim([-disc.obs_scale_2d[0], disc.obs_scale_2d[0]])
    axs[1].set_ylim([-disc.obs_scale_2d[1], disc.obs_scale_2d[1]])
    axs[1].set(xlabel=disc.obs_labels_2d[0], ylabel=disc.obs_labels_2d[1])
    axs[1].scatter(
        x=target_vars_flat[:, 0],
        y=target_vars_flat[:, 1],
        c=[0, 0, 1],
        alpha=0.3)
    fig.tight_layout()

    if output_path:
      file.MakeDirs(output_path)
      with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
        plt.savefig(f)

  return metrics
