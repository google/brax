# Copyright 2025 The Brax Authors.
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

"""Checkpointing functions."""

import inspect
import json
import logging
from typing import Any, Dict, Tuple, Union

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from etils import epath
from flax import linen as nn
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint as ocp


_ACTIVATION_REGISTRY = {
    'celu': nn.activation.celu,
    'compact': nn.activation.compact,
    'elu': nn.activation.elu,
    'gelu': nn.activation.gelu,
    'glu': nn.activation.glu,
    'hard_sigmoid': nn.activation.hard_sigmoid,
    'hard_silu': nn.activation.hard_silu,
    'hard_swish': nn.activation.hard_swish,
    'hard_tanh': nn.activation.hard_tanh,
    'leaky_relu': nn.activation.leaky_relu,
    'linear': lambda x: x,
    'log_sigmoid': nn.activation.log_sigmoid,
    'log_softmax': nn.activation.log_softmax,
    'logsumexp': nn.activation.logsumexp,
    'normalize': nn.activation.normalize,
    'one_hot': nn.activation.one_hot,
    'relu': nn.activation.relu,
    'relu6': nn.activation.relu6,
    'selu': nn.activation.selu,
    'sigmoid': nn.activation.sigmoid,
    'silu': nn.activation.silu,
    'soft_sign': nn.activation.soft_sign,
    'softmax': nn.activation.softmax,
    'softplus': nn.activation.softplus,
    'standardize': nn.activation.standardize,
    'swish': nn.activation.swish,
    'tanh': nn.activation.tanh,
}
_KERNEL_INIT_REGISTRY = {
    'constant': jax.nn.initializers.constant,
    'delta_orthogonal': jax.nn.initializers.delta_orthogonal,
    'glorot_normal': jax.nn.initializers.glorot_normal,
    'glorot_uniform': jax.nn.initializers.glorot_uniform,
    'he_normal': jax.nn.initializers.he_normal,
    'he_uniform': jax.nn.initializers.he_uniform,
    'kaiming_normal': jax.nn.initializers.kaiming_normal,
    'kaiming_uniform': jax.nn.initializers.kaiming_uniform,
    'lecun_normal': jax.nn.initializers.lecun_normal,
    'lecun_uniform': jax.nn.initializers.lecun_uniform,
    'normal': jax.nn.initializers.normal,
    'ones': jax.nn.initializers.ones,
    'orthogonal': jax.nn.initializers.orthogonal,
    'truncated_normal': jax.nn.initializers.truncated_normal,
    'uniform': jax.nn.initializers.uniform,
    'variance_scaling': jax.nn.initializers.variance_scaling,
    'xavier_normal': jax.nn.initializers.xavier_normal,
    'xavier_uniform': jax.nn.initializers.xavier_uniform,
    'zeros': jax.nn.initializers.zeros,
}
_KERNEL_INIT_FN_KEYWORDS = (
    'policy_network_kernel_init_fn',
    'value_network_kernel_init_fn',
    'q_network_kernel_init_fn',
)


def _get_function_kwargs(func: Any) -> Dict[str, Any]:
  """Gets kwargs of a function."""
  return {
      p.name: p.default
      for p in inspect.signature(func).parameters.values()
      if p.default is not inspect.Parameter.empty
  }


def _get_function_defaults(func: Any) -> Dict[str, Any]:
  """Gets default kwargs of a function potentially wrapped in partials."""
  kwargs = _get_function_kwargs(func)
  if hasattr(func, 'func'):
    kwargs.update(_get_function_defaults(func.func))
  return kwargs


def _np_jp_to_python_types(data: Any) -> Any:
  if isinstance(data, (np.ndarray, jp.ndarray)):
    return data.item() if data.ndim == 0 else data.tolist()
  if isinstance(data, dict):
    return {key: _np_jp_to_python_types(value) for key, value in data.items()}
  if isinstance(data, (list, tuple)):
    return type(data)(_np_jp_to_python_types(item) for item in data)
  return data


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[
        Union[
            bc_networks.BCNetworks,
            ppo_networks.PPONetworks,
            sac_networks.SACNetworks,
        ]
    ],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  config = config_dict.ConfigDict()
  kwargs = _get_function_kwargs(network_factory)
  defaults = _get_function_defaults(network_factory)

  if 'preprocess_observations_fn' in kwargs:
    if (
        kwargs['preprocess_observations_fn']
        != defaults['preprocess_observations_fn']
    ):
      raise ValueError(
          'checkpointing only supports identity_observation_preprocessor as the'
          ' preprocess_observations_fn'
      )
    del kwargs['preprocess_observations_fn']

  config.network_factory_kwargs = kwargs
  config.normalize_observations = normalize_observations
  config.observation_size = observation_size
  config.action_size = action_size
  return config


def get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[
        Union[
            bc_networks.BCNetworks,
            ppo_networks.PPONetworks,
            sac_networks.SACNetworks,
        ]
    ],
) -> Union[
    bc_networks.BCNetworks, ppo_networks.PPONetworks, sac_networks.SACNetworks
]:
  """Generates a network given config."""
  normalize = lambda x, y: x
  if config.normalize_observations:
    normalize = running_statistics.normalize
  network = network_factory(
      config.to_dict()['observation_size'],
      config.action_size,
      preprocess_observations_fn=normalize,
      **config.network_factory_kwargs,
  )
  return network


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Tuple[Any, ...],
    config: config_dict.ConfigDict,
    config_fname: str = 'config.json',
):
  """Saves a checkpoint."""
  ckpt_path = epath.Path(path) / f'{step:012d}'
  logging.info('saving checkpoint to %s', ckpt_path.as_posix())

  if not ckpt_path.exists():
    ckpt_path.mkdir(parents=True)

  # Save the network params.
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  orbax_checkpointer.save(ckpt_path, params, force=True, save_args=save_args)

  config_cp_dict = config.to_dict()
  # Convert activation functions to registered names.
  if 'activation' in config_cp_dict['network_factory_kwargs'] and callable(
      config_cp_dict['network_factory_kwargs']['activation']
  ):
    name_ = config_cp_dict['network_factory_kwargs']['activation'].__name__
    if name_ not in _ACTIVATION_REGISTRY:
      raise ValueError(
          f'Activation function {name_} not registered for checkpointing.'
      )
    config_cp_dict['network_factory_kwargs']['activation'] = name_
  # Convert kernel init functions to registered names.
  for init_fn_name in _KERNEL_INIT_FN_KEYWORDS:
    if init_fn_name not in config_cp_dict['network_factory_kwargs']:
      continue
    name_ = config_cp_dict['network_factory_kwargs'][init_fn_name].__name__
    if name_ not in _KERNEL_INIT_REGISTRY:
      raise ValueError(
          f'Kernel init function {name_} not registered for checkpointing.'
      )
    config_cp_dict['network_factory_kwargs'][init_fn_name] = name_
  config_cp_dict = _np_jp_to_python_types(config_cp_dict)
  config = config_dict.ConfigDict(config_cp_dict)

  # Save the config.
  config_path = ckpt_path / config_fname
  config_path.write_text(config.to_json_best_effort())


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'checkpoint path does not exist: {path.as_posix()}')

  logging.info('restoring from checkpoint %s', path.as_posix())

  metadata = ocp.PyTreeCheckpointer().metadata(path).item_metadata
  restore_args = jax.tree.map(
      lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
  )
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  target = orbax_checkpointer.restore(
      path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
  )
  target[0] = running_statistics.RunningStatisticsState(**target[0])

  return target


def load_config(
    config_path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads config from config path."""
  config_path = epath.Path(config_path)
  if not config_path.exists():
    raise ValueError(f'Config file not found at {config_path.as_posix()}')

  loaded_dict = json.loads(config_path.read_text())

  if 'activation' in loaded_dict['network_factory_kwargs']:
    activation_name = loaded_dict['network_factory_kwargs']['activation']
    loaded_dict['network_factory_kwargs']['activation'] = _ACTIVATION_REGISTRY[
        activation_name
    ]
  for init_fn_name in _KERNEL_INIT_FN_KEYWORDS:
    if init_fn_name not in loaded_dict['network_factory_kwargs']:
      continue
    init_fn_name_ = loaded_dict['network_factory_kwargs'][init_fn_name]
    loaded_dict['network_factory_kwargs'][init_fn_name] = (
        _KERNEL_INIT_REGISTRY[init_fn_name_]
    )

  return config_dict.create(**loaded_dict)
