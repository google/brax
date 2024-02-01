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

"""Train a ComposerEnv."""
import copy
import datetime
from typing import Dict, Any
from brax.v1.experimental.braxlines import experiments
from brax.v1.experimental.braxlines.common import evaluators
from brax.v1.experimental.braxlines.common import logger_utils
from brax.v1.experimental.braxlines.training import ppo
from brax.v1.experimental.composer import composer
from brax.v1.experimental.composer.training import mappo

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
  env_params = config.pop('env_params', {})
  desc_edits = config.pop('desc_edits', {})
  seed = config.pop('seed', 0)
  eval_seed = config.pop('eval_seed', 0)
  ppo_params = experiments.defaults.get_ppo_params(env_name, default='ant')
  ppo_params.update(config.pop('ppo_params', {}))
  assert not config, f'unused config: {config}'

  env_fn = composer.create_fn(
      env_name=env_name, desc_edits=desc_edits, **env_params)

  # @title Training the custom env
  log_path = output_path
  if log_path:
    log_path = f'{log_path}/training_curves.csv'
  tab = logger_utils.Tabulator(output_path=log_path, append=False)

  # We determined some reasonable hyperparameters offline and share them here.
  times = [datetime.datetime.now()]
  plotpatterns = ['eval/episode_reward', 'eval/episode_score']

  progress, _, _, _ = experiments.get_progress_fn(
      plotpatterns,
      times,
      tab=tab,
      max_ncols=5,
      xlim=[0, ppo_params['num_timesteps']],
      return_dict=return_dict,
      progress_dict=progress_dict)

  ppo_lib = mappo if env_fn().metadata.agent_groups else ppo
  inference_fn, params, _ = ppo_lib.train(
      environment_fn=env_fn,
      progress_fn=progress,
      seed=seed,
      extra_step_kwargs=False,
      **ppo_params)
  time_to_jit = times[1] - times[0]
  time_to_train = times[-1] - times[1]
  print(f'time to jit: {time_to_jit}')
  print(f'time to train: {time_to_train}')
  print(f'Saved logs to {log_path}')
  return_dict.update(dict(time_to_jit=time_to_jit, time_to_train=time_to_train))

  evaluators.visualize_env(
      env_fn=env_fn,
      inference_fn=inference_fn,
      params=params,
      batch_size=0,
      seed=eval_seed,
      output_path=output_path,
      verbose=True)
  return return_dict
