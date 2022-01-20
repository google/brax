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

"""Evaluators for VGRL."""


import copy
import functools
import itertools
from typing import Dict, Tuple, Any
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.vgcrl.utils import Discriminator
from brax.experimental.composer import observers
from brax.io import file
from brax.io import html
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions


@jax.jit
def jit_compute_distance(env_vars: jnp.ndarray, goals: jnp.ndarray):
  dist_vector = jnp.mean(jnp.abs(env_vars - goals[None, None]), axis=(0, 1, 2))
  dist = jnp.mean(
      jnp.sqrt(jnp.mean((env_vars - goals[None, None])**2, axis=-1)))
  return dist_vector, dist


@functools.partial(jax.jit, static_argnums=(2,))
def jit_compute_mi_1d(env_vars: jnp.ndarray, obs_scale: jnp.ndarray,
                      num_1d_bins: int):
  """Use binning to compute 1d mutual information."""
  hist = jnp.histogram(
      env_vars.flatten(), bins=num_1d_bins, range=(-obs_scale, obs_scale))
  discretization_all = hist[0] / env_vars.flatten().shape[0]
  entropy_all = -jnp.sum(
      discretization_all * jnp.log(discretization_all + 1e-12))
  discretization_z = [
      jnp.histogram(env_vars[..., j].flatten(), bins=hist[1])[0] /
      env_vars[..., j].flatten().shape[0] for j in range(env_vars.shape[-1])
  ]
  entropy_z = -jnp.mean(
      jnp.array(
          [jnp.sum(p_z * jnp.log(p_z + 1e-12)) for p_z in discretization_z]))
  mi = entropy_all - entropy_z
  return mi, entropy_all, entropy_z


def estimate_latent_goal_reaching_metric(disc: Discriminator,
                                         params: Dict[str, Dict[str,
                                                                jnp.ndarray]],
                                         goals: jnp.ndarray,
                                         time_subsampling: int = 1,
                                         time_last_n: int = 500,
                                         verbose: bool = False,
                                         **kwargs):
  """Compute latent-goal reaching metric."""
  assert not disc.normalize_obs
  assert goals.shape[
      -1] == disc.indexed_obs_size, f'{goals.shape}[-1] != {disc.indexed_obs_size}'
  _, I = goals.shape

  # TODO: add stochastic q(z) sampling
  goal_obs = disc.unindex_obs(goals)
  dist_q = disc.dist_q_z_o(goal_obs, params['extra'])
  if isinstance(dist_q, tfd.MultivariateNormalDiag):
    zs = dist_q.loc
  elif isinstance(dist_q, tfd.OneHotCategorical):
    zs = jax.nn.one_hot(jnp.argmax(dist_q.logits, axis=-1), disc.z_size)
  else:
    raise NotImplementedError(dist_q)

  _, _, _, _, _, env_vars = rollout_skills(
      params=params, disc=disc, verbose=verbose, num_z=None, zs=zs, **kwargs)
  env_vars = env_vars[-time_last_n:][::time_subsampling]  # [T, M, D, I]
  dist_vector, dist = jit_compute_distance(env_vars, goals)
  metrics = {
      f'metrics/lgr_1d/{disc.obs_labels[i]}': dist_vector[i] for i in range(I)
  }
  metrics.update({
      'metrics/lgr': dist,
  })
  return metrics


def estimate_empowerment_metric(disc: Discriminator,
                                time_subsampling: int = 1,
                                time_last_n: int = 500,
                                num_1d_bins: int = 1000,
                                num_2d_bins: int = 1000,
                                verbose: bool = True,
                                include_1d: bool = True,
                                include_2d: bool = False,
                                custom_obs_indices: Tuple[Any] = (),
                                custom_obs_scale: float = None,
                                **kwargs):
  """Compute non-parametric estimate of mutual information."""
  env, _, zs, _, env_obses, _ = rollout_skills(
      disc=disc, verbose=verbose, **kwargs)

  obs_indices = copy.deepcopy(disc.obs_indices)
  obs_scale = copy.deepcopy(disc.obs_scale)
  obs_labels = copy.deepcopy(disc.obs_labels)
  if custom_obs_indices:
    custom_obs_indices, custom_obs_labels = observers.index_preprocess(
        custom_obs_indices, env)
    custom_indexed_obs_size = len(custom_obs_indices)
    custom_obs_scale = jnp.array(custom_obs_scale or
                                 1.) * jnp.ones(custom_indexed_obs_size)
    obs_labels += custom_obs_labels
    obs_indices += custom_obs_indices
    obs_scale = jnp.concatenate([obs_scale, custom_obs_scale])
  env_vars = env_obses[..., obs_indices]

  D = zs.shape[0]
  I = env_vars.shape[-1]

  env_vars = env_vars[-time_last_n:][::time_subsampling]  # [T, M, D, I]

  metrics = {}

  # compute 1D estimates
  if include_1d:
    for i in range(I):
      mi, entropy_all, entropy_z = jit_compute_mi_1d(env_vars[..., i],
                                                     obs_scale[i], num_1d_bins)
      metrics.update({
          f'metrics/mi_1d/{obs_labels[i]}': mi,
          f'metrics/entropy_all_1d/{obs_labels[i]}': entropy_all,
          f'metrics/entropy_z_1d/{obs_labels[i]}': entropy_z,
      })

  # compute 2D estimates
  # jax histogram2d has an error. Switch to np for now.
  if include_2d:
    env_vars = np.array(env_vars)
    obs_scale = np.array(obs_scale)
    for i1, i2 in itertools.combinations(range(I), 2):
      if i1 == i2:
        continue
      hist = np.histogram2d(
          env_vars[..., i1].flatten(),
          env_vars[..., i2].flatten(),
          bins=num_2d_bins,
          range=((-obs_scale[i1], obs_scale[i1]), (-obs_scale[i2],
                                                   obs_scale[i2])))
      etization_all = hist[0] / env_vars[..., i1].flatten().shape[0]
      entropy_all = -np.sum(etization_all * np.log(etization_all + 1e-12))
      etization_z = [
          np.histogram2d(
              env_vars[..., j, i1].flatten(),
              env_vars[..., j, i2].flatten(),
              bins=hist[1:])[0] / env_vars[..., j, i1].flatten().shape[0]
          for j in range(D)
      ]
      entropy_z = -np.mean(
          np.array([np.sum(p_z * np.log(p_z + 1e-12)) for p_z in etization_z]))
      mi = entropy_all - entropy_z
      metrics.update({
          f'metrics/mi_2d/{obs_labels[i1]}_{obs_labels[i2]}':
              mi,
          f'metrics/entropy_all_2d/{obs_labels[i1]}_{obs_labels[i2]}':
              entropy_all,
          f'metrics/entropy_z_2d/{obs_labels[i1]}_{obs_labels[i2]}':
              entropy_z,
      })

  return metrics


def rollout_skills(
    params: Dict[str, Dict[str, jnp.ndarray]],
    env_fn,
    disc: Discriminator,
    inference_fn=None,
    verbose: bool = True,
    num_z: int = 10,
    num_samples_per_z: int = 10,
    seed: int = 0,
    zs: jnp.ndarray = None,
):
  """Rollout skills in parallel."""
  env = env_fn()
  O = disc.env_obs_size
  I = disc.indexed_obs_size
  Z = disc.z_size
  M = num_samples_per_z

  rng = jax.random.PRNGKey(seed)
  if zs is None:
    # Sample {D} z's [D, Z]
    dist_p = disc.dist_p_z()
    if isinstance(dist_p, tfd.Deterministic):
      zs = jnp.array([dist_p.loc])  # D=1
    elif isinstance(dist_p, tfd.OneHotCategorical):
      zs = jax.nn.one_hot(jnp.arange(0, disc.z_size), disc.z_size)  # D=Z
    else:
      rng, z_key = jax.random.split(rng)
      zs = disc.sample_p_z(num_z, z_key)  # D=num_z
  D = zs.shape[0]

  # Repeat each z by {M} times
  batch_z = jnp.repeat(zs, M, axis=0)  # [D*M, Z]

  batch_env, states = evaluators.rollout_env(
      env_fn=env_fn,
      params=params,
      batch_size=D * M,
      inference_fn=inference_fn,
      reset_args=(batch_z,),
      step_args=() if params is None else
      (params['normalizer'], params['extra']),
      step_fn_name='step2' if params is None else 'step',
  )

  # Get env dimensions of interest
  obses = jnp.stack([state.obs for state in states], axis=0)  # [T, D*M, O+D]
  if verbose:
    print((f'T={len(states)}, O={O}, Z={Z}, D={D}, M={M}'))
  env_obses, _ = batch_env.disc.split_obs(obses)  # [T, D*M, O]
  env_vars = batch_env.disc.index_obs(env_obses)  # [T, D*M, I]
  env_obses = env_obses.reshape(-1, D, M, O).swapaxes(1, 2)  # [T, M, D, O]
  env_vars = env_vars.reshape(-1, D, M, I).swapaxes(1, 2)  # [T, M, D, I]
  return env, batch_env, zs, states, env_obses, env_vars


def visualize_skills(output_path: str = None,
                     time_subsampling: int = 10,
                     time_last_n: int = 500,
                     output_name: str = 'skills',
                     save_video: bool = False,
                     disc: Discriminator = None,
                     num_samples_per_z: int = 5,
                     **kwargs):
  """Visualizing skills in a 2D plot."""
  env, _, zs, states, env_obses, env_vars = rollout_skills(
      disc=disc, num_samples_per_z=num_samples_per_z, **kwargs)
  D = zs.shape[0]
  I = env_vars.shape[-1]

  env_obses = env_obses[-time_last_n:][::time_subsampling]
  env_vars = env_vars[-time_last_n:][::time_subsampling]

  labels = disc.obs_labels
  assert env_vars.shape[-1] == len(
      labels), f'{env_vars.shape}[-1] != {len(labels)}'
  if env_vars.shape[-1] % 2:
    env_vars = jnp.concatenate(
        [env_vars, jnp.zeros(env_vars.shape[:-1] + (1,))], axis=-1)
    labels = tuple(labels) + (None,)
  env_vars = env_vars.reshape(-1, D, I)  # [T*M, D, I]

  # Plot
  def spec(N):
    t = np.linspace(-510, 510, N)
    return np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255) / 255.

  colours = spec(D)  # [D, 3]
  nfigs = int(len(labels) / 2)
  _, axs = plt.subplots(ncols=nfigs, figsize=(7 * nfigs, 6))
  if nfigs == 1:
    axs = [axs]
  for i in range(nfigs):
    for j, (z, c) in enumerate(zip(zs, colours)):
      axs[i].scatter(
          x=env_vars[:, j, 2 * i],
          y=env_vars[:, j, 2 * i + 1],
          c=c,
          alpha=0.3,
          label=f'z={z}')
      axs[i].legend()
      axs[i].set(xlabel=f'{labels[2*i]}', ylabel=f'{labels[2*i+1]}')
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
