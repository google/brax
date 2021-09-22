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

"""Train a ComposerEnv."""
import copy
import datetime
from typing import Dict, Any
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.training import ppo
from brax.experimental.composer import composer
from brax.experimental.composer import register_default_components
import jax.numpy as jnp
import matplotlib.pyplot as plt
register_default_components()

TASK_KEYS = (('env_name',),)


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
  output_path = output_dir

  # extra parameters
  env_name = config.pop('env_name', 'ant_run')
  desc_edits = config.pop('desc_edits', {})
  eval_seed = config.pop('eval_seed', 0)
  ppo_params = dict(
      num_timesteps=int(5e7),
      log_frequency=20,
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
      extra_step_kwargs=False,
      batch_size=1024)
  ppo_params.update(config.pop('ppo_params', {}))
  assert not config, f'unused config: {config}'

  env_fn = composer.create_fn(env_name=env_name, desc_edits=desc_edits)

  # @title Training the custom env
  log_path = output_path
  if log_path:
    log_path = f'{log_path}/training_curves.csv'
  tab = logger_utils.Tabulator(output_path=log_path, append=False)

  # We determined some reasonable hyperparameters offline and share them here.
  times = [datetime.datetime.now()]
  plotdata = {}
  plotpatterns = ['eval/episode_reward', 'eval/episode_score']

  def progress(num_steps, metrics, params):
    del params
    times.append(datetime.datetime.now())
    plotkeys = []
    for key, v in metrics.items():
      assert not jnp.isnan(v), f'{key} {num_steps} NaN'
      plotdata[key] = plotdata.get(key, dict(x=[], y=[]))
      plotdata[key]['x'] += [num_steps]
      plotdata[key]['y'] += [v]
      if any(x in key for x in plotpatterns):
        plotkeys += [key]
    if num_steps > 0:
      tab.add(num_steps=num_steps, **metrics)
      tab.dump()
      return_dict.update(dict(num_steps=num_steps, **metrics))
      progress_dict.update(dict(num_steps=num_steps, **metrics))
    num_figs = max(len(plotkeys), 2)
    fig, axs = plt.subplots(ncols=num_figs, figsize=(3.5 * num_figs, 3))
    for i, key in enumerate(plotkeys):
      if key in plotdata:
        axs[i].plot(plotdata[key]['x'], plotdata[key]['y'])
      axs[i].set(xlabel='# environment steps', ylabel=key)
      axs[i].set_xlim([0, ppo_params['num_timesteps']])
    fig.tight_layout()

  inference_fn, params, _ = ppo.train(
      environment_fn=env_fn, progress_fn=progress, **ppo_params)
  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')
  print(f'Saved logs to {log_path}')

  evaluators.visualize_env(
      env_fn=env_fn,
      inference_fn=inference_fn,
      params=params,
      batch_size=0,
      seed=eval_seed,
      output_path=output_path,
      verbose=True)
  return return_dict
