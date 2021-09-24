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

"""Experiment configuration loader and runner."""
# pylint:disable=broad-except
import importlib
import pprint
from typing import Dict, Any
from brax.experimental.braxlines.common import config_utils

DEFAULT_LIB_PATH_TEMPLATE = 'brax.experimental.braxlines.experiments'


def load_experiment(experiment_name: str):
  load_path = f'{DEFAULT_LIB_PATH_TEMPLATE}.{experiment_name}'
  experiment_lib = importlib.import_module(load_path)
  return experiment_lib.AGENT_MODULE, experiment_lib.CONFIG


def run_experiment(experiment_name: str = None,
                   output_path: str = '/tmp/sweep',
                   start_count: int = 0,
                   end_count: int = int(1e6),
                   ignore_errors: bool = False,
                   agent_module: str = None,
                   config: Dict[str, Any] = None):
  """Run experiments defined by `config` serially."""
  if not agent_module and not config:
    agent_module, config = load_experiment(experiment_name)
  if isinstance(agent_module, str):
    agent_module = importlib.import_module(agent_module)

  prefix_keys = config_utils.list_keys_to_expand(config)
  for c, p in zip(config, prefix_keys):
    c.update(dict(prefix_keys=p))
  config_count = config_utils.count_configuration(config)
  start_count = max(start_count, 0)
  end_count = min(end_count, sum(config_count))
  print(f'Loaded experiment_name={experiment_name}')
  print(f'Loaded {sum(config_count)}({config_count}) experiment configurations')
  print(f'Set start_count={start_count}, end_count={end_count}')
  print(f'Set prefix_keys={prefix_keys}')
  print(f'Set output_dir={output_path}')

  # @title Launch experiments
  for i in range(start_count, end_count):
    c, _ = config_utils.index_configuration(config, index=i, count=config_count)
    task_name = config_utils.get_compressed_name_from_keys(
        c, agent_module.TASK_KEYS)
    experiment_name = config_utils.get_compressed_name_from_keys(
        c, c.pop('prefix_keys'))
    output_dir = f'{output_path}/{task_name}/{experiment_name}'
    print(f'[{i+1}/{sum(config_count)}] Starting experiment...')
    print(f'\t config: {pprint.pformat(c, indent=2)}')
    print(f'\t output_dir={output_dir}')
    return_dict = {}
    if ignore_errors:
      try:
        agent_module.train(c, output_dir=output_dir, return_dict=return_dict)
      except Exception as e:
        print(
            f'[{i+1}/{sum(config_count)}] FAILED experiment {e.__class__.__name__}: {e.message}'
        )
    else:
      agent_module.train(c, output_dir=output_dir, return_dict=return_dict)
    print(f'\t time_to_jit={return_dict.get("time_to_train", None)}')
    print(f'\t time_to_train={return_dict.get("time_to_jit", None)}')
