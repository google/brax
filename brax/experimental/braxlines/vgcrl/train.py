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

"""Train VGCRL."""
import copy
import datetime
import functools
from typing import Dict, Any
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils
from brax.experimental.composer import composer
from brax.io import file
import jax
import matplotlib.pyplot as plt

TASK_KEYS = (('env_name',), ('env_space',), ('env_scale',))


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
  env_space = config.pop('env_space', 'vel')
  env_scale = config.pop('env_scale', 5.0)
  obs_indices = {
      'vel': {
          'ant': (13, 14),
          'humanoid': (22, 23),
          'halfcheetah': (11,),
          'uni_ant': (('vel:torso_ant1', 0), ('vel:torso_ant1', 1)),
          'bi_ant': (('vel:torso_ant1', 0), ('vel:torso_ant2', 0)),
      },
      'ang': {
          'ant': (17,),
          'uni_ant': (('ang:torso_ant1', 2),),
      },
  }[env_space][env_name]

  # @markdown **Experiment Parameters**
  algo_name = config.pop('algo_name', 'diayn')
  logits_clip_range = config.pop('logits_clip_range', 5.0)
  normalize_obs_for_disc = config.pop('normalize_obs_for_disc', True)
  seed = config.pop('seed', 0)
  diayn_num_skills = config.pop('diayn_num_skills', 8)
  disc_update_ratio = config.pop('disc_update_ratio', 1.0)
  spectral_norm = config.pop('spectral_norm', False)
  ppo_params = dict(
      num_timesteps=int(5e7),
      reward_scaling=10,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=5,
      num_minibatches=32,
      num_update_epochs=4,
      discounting=0.95,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=2048,
      batch_size=1024)
  ppo_params.update(config.pop('ppo_params', {}))
  assert not config, f'unused config: {config}'

  # @title Visualizing Brax environments
  # Create baseline environment to get observation specs
  base_env_fn = composer.create_fn(env_name=env_name)
  base_env = base_env_fn()
  env_obs_size = base_env.observation_size

  # Create discriminator-parameterized environment
  disc_fn = vgcrl_utils.create_disc_fn(
      algo_name=algo_name,
      observation_size=env_obs_size,
      obs_indices=obs_indices,
      scale=env_scale,
      diayn_num_skills=diayn_num_skills,
      logits_clip_range=logits_clip_range,
      spectral_norm=spectral_norm)
  disc = disc_fn(env=base_env, normalize_obs=normalize_obs_for_disc)
  extra_params = disc.init_model(rng=jax.random.PRNGKey(seed=seed))
  env_fn = vgcrl_utils.create_fn(env_name=env_name, disc=disc)

  # @title Training
  tab = logger_utils.Tabulator(
      output_path=f'{output_dir}/training_curves.csv', append=False)

  # We determined some reasonable hyperparameters offline and share them here.
  train_fn = functools.partial(ppo.train, log_frequency=20, **ppo_params)

  times = [datetime.datetime.now()]
  plotdata = {}
  plotkeys = ['eval/episode_reward', 'losses/disc_loss']

  def plot(output_path: str = None, output_name: str = 'training_curves'):
    num_figs = len(plotkeys)
    fig, axs = plt.subplots(ncols=num_figs, figsize=(3.5 * num_figs, 3))
    for i, key in enumerate(plotkeys):
      if key in plotdata:
        axs[i].plot(plotdata[key]['x'], plotdata[key]['y'])
      axs[i].set(xlabel='# environment steps', ylabel=key)
      axs[i].set_xlim([0, train_fn.keywords['num_timesteps']])
    fig.tight_layout()
    if output_path:
      with file.File(f'{output_path}/{output_name}.png', 'wb') as f:
        plt.savefig(f)

  def progress(num_steps, metrics, _):
    times.append(datetime.datetime.now())
    for key, v in metrics.items():
      plotdata[key] = plotdata.get(key, dict(x=[], y=[]))
      plotdata[key]['x'] += [num_steps]
      plotdata[key]['y'] += [v]
    # the first step does not include losses
    if num_steps > 0:
      tab.add(num_steps=num_steps, **metrics)
      tab.dump()
      return_dict.update(dict(num_steps=num_steps, **metrics))
      progress_dict.update(dict(num_steps=num_steps, **metrics))

  extra_loss_fns = dict(disc_loss=disc.disc_loss_fn) if extra_params else None
  extra_loss_update_ratios = dict(
      disc_loss=disc_update_ratio) if extra_params else None
  inference_fn, params, _ = train_fn(
      environment_fn=env_fn,
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
      env_fn,
      inference_fn,
      obs_indices,
      params,
      env_scale,
      algo_name,
      output_dir,
      verbose=True,
      num_samples_per_z=5,
      time_subsampling=10,
      time_last_n=500,
      seed=seed,
      save_video=True)

  return return_dict
