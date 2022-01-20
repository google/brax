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

"""Train."""
import copy
import datetime
import functools
from typing import Dict, Any
from brax.experimental.braxlines import experiments
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.envs.obs_indices import OBS_INDICES
from brax.experimental.braxlines.irl_smm import evaluators as irl_evaluators
from brax.experimental.braxlines.irl_smm import utils as irl_utils
from brax.experimental.braxlines.training import ppo
from brax.experimental.composer import composer
import jax

TASK_KEYS = (
    ('env_name',),
    ('obs_indices',),
    ('obs_scale',),
    ('target_num_modes',),
    ('target_num_samples',),
)


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
  target_num_modes = config.pop('target_num_modes', 2)
  target_num_samples = config.pop('target_num_samples', 250)

  # @markdown **Experiment Parameters**
  reward_type = config.pop('reward_type', 'gail2')
  logits_clip_range = config.pop('logits_clip_range', 5.0)
  env_reward_multiplier = config.pop('env_reward_multiplier', 0.0)
  normalize_obs_for_disc = config.pop('normalize_obs_for_disc', False)
  balance_data_for_disc = config.pop('balance_data_for_disc', True)
  seed = config.pop('seed', 0)
  evaluate_dist = config.pop('evaluate_dist', False)
  spectral_norm = config.pop('spectral_norm', False)
  gradient_penalty_weight = config.pop('gradient_penalty_weight', 0.0)
  ppo_params = experiments.defaults.get_ppo_params(env_name, default='ant')
  ppo_params.update(config.pop('ppo_params', {}))
  assert not config, f'unused config: {config}'

  # generate target data
  rng = jax.random.PRNGKey(seed=seed)
  jit_get_dist = jax.jit(
      functools.partial(
          irl_utils.get_multimode_dist,
          indexed_obs_dim=len(obs_indices),
          num_modes=target_num_modes,
          scale=obs_scale))
  target_dist = jit_get_dist()
  target_data = target_dist.sample(seed=rng, sample_shape=(target_num_samples,))

  # make env_fn
  base_env_fn = composer.create_fn(env_name=env_name)
  base_env = base_env_fn()
  disc = irl_utils.IRLDiscriminator(
      input_size=len(obs_indices),
      obs_indices=obs_indices,
      obs_scale=obs_scale,
      include_action=False,
      arch=(32, 32),
      logits_clip_range=logits_clip_range,
      spectral_norm=spectral_norm,
      gradient_penalty_weight=gradient_penalty_weight,
      reward_type=reward_type,
      normalize_obs=normalize_obs_for_disc,
      balance_data=balance_data_for_disc,
      target_data=target_data,
      target_dist_fn=jit_get_dist,
      env=base_env)
  extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=0))
  env_fn = irl_utils.create_fn(
      env_name=env_name,
      wrapper_params=dict(
          disc=disc,
          env_reward_multiplier=env_reward_multiplier,
      ))
  eval_env_fn = functools.partial(env_fn, auto_reset=False)
  # make inference functions and goals for evaluation
  core_env = env_fn()
  inference_fn = ppo.make_inference_fn(
      core_env.observation_size,
      core_env.action_size,
      normalize_observations=ppo_params.get('normalize_observation', True))
  inference_fn = jax.jit(inference_fn)

  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)
  # We determined some reasonable hyperparameters offline and share them here.
  train_fn = functools.partial(ppo.train, **ppo_params)

  times = [datetime.datetime.now()]
  plotpatterns = [
      'eval/episode_reward', 'losses/disc_loss', 'losses/total_loss',
      'losses/policy_loss', 'losses/value_loss', 'losses/entropy_loss',
      'metrics/energy_dist'
  ]

  def update_metrics_fn(num_steps, metrics, params):
    del num_steps
    if evaluate_dist:
      metrics.update(
          irl_evaluators.estimate_energy_distance_metric(
              params=params,
              disc=disc,
              target_data=target_data,
              env_fn=eval_env_fn,
              inference_fn=inference_fn,
              num_samples=10,
              time_subsampling=10,
              time_last_n=500,
              visualize=False,
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

  extra_loss_fns = dict(disc_loss=disc.disc_loss_fn)
  inference_fn, params, _ = train_fn(
      environment_fn=env_fn,
      seed=seed,
      progress_fn=progress,
      extra_params=extra_params,
      extra_loss_fns=extra_loss_fns)
  plot(output_path=output_dir)
  irl_evaluators.visualize_disc(
      params=params, disc=disc, num_grid=25, output_path=output_dir)

  return_dict.update(dict(time_to_jit=times[1] - times[0]))
  return_dict.update(dict(time_to_train=times[-1] - times[1]))
  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')

  for i in range(3):
    evaluators.visualize_env(
        env_fn=eval_env_fn,
        inference_fn=inference_fn,
        params=params,
        batch_size=0,
        step_args=(params['normalizer'], params['extra']),
        output_path=output_dir,
        output_name=f'video_eps{i}',
    )

  metrics = irl_evaluators.estimate_energy_distance_metric(
      params=params,
      disc=disc,
      target_data=target_data,
      env_fn=eval_env_fn,
      inference_fn=inference_fn,
      num_samples=10,
      time_subsampling=10,
      time_last_n=500,
      visualize=True,
      output_path=output_dir,
      seed=0)
  return_dict.update(metrics)

  return return_dict
