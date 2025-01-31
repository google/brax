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

"""Checkpointing for PPO."""

import inspect
import json
import logging
from typing import Any, Dict, Tuple, Union

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from flax import linen
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp

_CONFIG_FNAME = 'config.json'


def _get_default_kwargs(func: Any) -> Dict[str, Any]:
  """Returns the default kwargs of a function."""
  return {
      p.name: p.default
      for p in inspect.signature(func).parameters.values()
      if p.default is not inspect.Parameter.empty
  }


def ppo_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating PPO params from a checkpoint."""
  config = config_dict.ConfigDict()
  kwargs = _get_default_kwargs(network_factory)

  if (
      kwargs.get('preprocess_observations_fn')
      != types.identity_observation_preprocessor
  ):
    raise ValueError(
        'preprocess_observations_fn must be identity_observation_preprocessor'
    )
  del kwargs['preprocess_observations_fn']
  if kwargs.get('activation') != linen.swish:
    raise ValueError('activation must be swish')
  del kwargs['activation']

  config.network_factory_kwargs = kwargs
  config.normalize_observations = normalize_observations
  config.observation_size = observation_size
  config.action_size = action_size
  return config


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Tuple[Any, ...],
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  ckpt_path = epath.Path(path) / f'{step:012d}'
  logging.info('saving checkpoint to %s', ckpt_path.as_posix())

  if not ckpt_path.exists():
    ckpt_path.mkdir(parents=True)

  config_path = epath.Path(path) / _CONFIG_FNAME
  if not config_path.exists():
    config_path.write_text(config.to_json())

  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  orbax_checkpointer.save(ckpt_path, params, force=True, save_args=save_args)


def load(
    path: Union[str, epath.Path],
):
  """Loads PPO checkpoint."""
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'PPO checkpoint path does not exist: {path.as_posix()}')

  logging.info('restoring from checkpoint %s', path.as_posix())

  orbax_checkpointer = ocp.PyTreeCheckpointer()
  target = orbax_checkpointer.restore(path, item=None)
  target[0] = running_statistics.RunningStatisticsState(**target[0])

  return target


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks],
) -> ppo_networks.PPONetworks:
  """Generates a PPO network given config."""
  normalize = lambda x, y: x
  if config.normalize_observations:
    normalize = running_statistics.normalize
  ppo_network = network_factory(
      config.to_dict()['observation_size'],
      config.action_size,
      preprocess_observations_fn=normalize,
      **config.network_factory_kwargs,
  )
  return ppo_network


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from PPO checkpoint."""
  path = epath.Path(path)

  config_path = path.parent / _CONFIG_FNAME
  if not config_path.exists():
    raise ValueError(f'PPO config file not found at {config_path.as_posix()}')

  config = config_dict.create(**json.loads(config_path.read_text()))

  params = load(path)
  ppo_network = _get_network(config, network_factory)
  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

  return make_inference_fn(params, deterministic=deterministic)
