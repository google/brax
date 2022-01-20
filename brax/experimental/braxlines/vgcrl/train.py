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

"""Train VGCRL."""
import copy
import datetime
import functools
from typing import Dict, Any
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.envs.obs_indices import OBS_INDICES
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils
from brax.experimental.composer import composer
import jax
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions

TASK_KEYS = (('env_name',), ('obs_indices',), ('obs_scale',))


def train(train_job_params: Dict[str, Any],
          output_dir: str,
          return_dict: Dict[str, float] = None,
          progress_dict: Dict[str, float] = None,
          env_tag: str = None):
  """Train."""
  del env_tag
  return_dict = return_dict or {}
  progress_dict = progress_dict or {}
  logger_utils.save_config(
      f'{output_dir}/config.txt', train_job_params, verbose=True)
  config = copy.deepcopy(train_job_params)

  # @markdown **Task Parameters**
  env_name = config.pop('env_name', 'ant')
  obs_indices = config.pop('obs_indices', 'vel')
  obs_scale = config.pop('obs_scale', 5.0)
  obs_indices = OBS_INDICES[obs_indices][env_name]

  # @markdown **Experiment Parameters**
  algo_name = config.pop('algo_name', 'diayn')
  logits_clip_range = config.pop('logits_clip_range', 5.0)
  env_reward_multiplier = config.pop('env_reward_multiplier', 0.0)
  normalize_obs_for_disc = config.pop('normalize_obs_for_disc', False)
  seed = config.pop('seed', 0)
  evaluate_mi = config.pop('evaluate_mi', False)
  evaluate_lgr = config.pop('evaluate_lgr', False)
  diayn_num_skills = config.pop('diayn_num_skills', 8)
  disc_update_ratio = config.pop('disc_update_ratio', 1.0)
  spectral_norm = config.pop('spectral_norm', False)
  ppo_params = experiments.defaults.get_ppo_params(env_name, default='ant')
  ppo_params.update(config.pop('ppo_params', {}))
  assert not config, f'unused config: {config}'

  # @title Visualizing Brax environments
  # Create baseline environment to get observation specs
  base_env_fn = composer.create_fn(env_name=env_name)
  base_env = base_env_fn()
  env_obs_size = base_env.observation_size

  # Create discriminator-parameterized environment
  disc = vgcrl_utils.create_disc_fn(
      algo_name=algo_name,
      observation_size=env_obs_size,
      obs_indices=obs_indices,
      scale=obs_scale,
      diayn_num_skills=diayn_num_skills,
      logits_clip_range=logits_clip_range,
      spectral_norm=spectral_norm,
      env=base_env,
      normalize_obs=normalize_obs_for_disc)()
  extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=seed))
  env_fn = vgcrl_utils.create_fn(
      env_name=env_name,
      wrapper_params=dict(
          env_reward_multiplier=env_reward_multiplier, disc=disc))
  eval_env_fn = functools.partial(env_fn, auto_reset=False)

  # make inference function and test goals
  core_env = env_fn()
  inference_fn = ppo.make_inference_fn(
      core_env.observation_size,
      core_env.action_size,
      normalize_observations=ppo_params.get('normalize_observation', True),
      extra_params=extra_params)
  inference_fn = jax.jit(inference_fn)
  goals = tfd.Uniform(
      low=-disc.obs_scale, high=disc.obs_scale).sample(
          seed=jax.random.PRNGKey(0), sample_shape=(10,))

  # @title Training
  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)

  # We determined some reasonable hyperparameters offline and share them here.
  train_fn = functools.partial(ppo.train, **ppo_params)

  times = [datetime.datetime.now()]
  plotpatterns = [
      'eval/episode_reward', 'losses/disc_loss', 'metrics/lgr',
      'metrics/entropy_all_', 'metrics/entropy_z_', 'metrics/mi_'
  ]

  def update_metrics_fn(num_steps, metrics, params):
    del num_steps
    if evaluate_mi:
      metrics.update(
          vgcrl_evaluators.estimate_empowerment_metric(
              env_fn=eval_env_fn,
              disc=disc,
              inference_fn=inference_fn,
              params=params,
              num_z=10,
              num_samples_per_z=10,
              time_subsampling=1,
              time_last_n=500,
              num_1d_bins=1000,
              num_2d_bins=30,
              verbose=True,
              seed=0))
    if evaluate_lgr:
      metrics.update(
          vgcrl_evaluators.estimate_latent_goal_reaching_metric(
              params=params,
              env_fn=eval_env_fn,
              disc=disc,
              inference_fn=inference_fn,
              goals=goals,
              num_samples_per_z=10,
              time_subsampling=1,
              time_last_n=500,
              seed=0))

  progress, plot, _, _ = experiments.get_progress_fn(
      plotpatterns,
      times,
      tab=tab,
      max_ncols=5,
      return_dict=return_dict,
      progress_dict=progress_dict,
      xlim=[0, train_fn.keywords['num_timesteps']],
      update_metrics_fn=update_metrics_fn)

  extra_loss_fns = dict(disc_loss=disc.disc_loss_fn) if extra_params else None
  extra_loss_update_ratios = dict(
      disc_loss=disc_update_ratio) if extra_params else None
  inference_fn, params, _ = train_fn(
      environment_fn=env_fn,
      seed=seed,
      progress_fn=progress,
      extra_params=extra_params,
      extra_loss_fns=extra_loss_fns,
      extra_loss_update_ratios=extra_loss_update_ratios,
  )
  plot(output_path=output_dir)

  return_dict.update(dict(time_to_jit=times[1] - times[0]))
  return_dict.update(dict(time_to_train=times[-1] - times[1]))
  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')

  vgcrl_evaluators.visualize_skills(
      env_fn=eval_env_fn,
      inference_fn=inference_fn,
      disc=disc,
      params=params,
      output_path=output_dir,
      verbose=True,
      num_z=20,
      num_samples_per_z=5,
      time_subsampling=10,
      time_last_n=500,
      seed=seed,
      save_video=True)

  return return_dict
