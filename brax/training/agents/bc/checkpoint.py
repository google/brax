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

"""Checkpointing for BC."""

from typing import Any, Optional, Union

from brax.training import checkpoint
from brax.training import types
from brax.training.agents.bc import networks as bc_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'bc_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[bc_networks.BCNetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_bc_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[bc_networks.BCNetworks],
) -> bc_networks.BCNetworks:
  """Generates a BC network given config."""
  return checkpoint.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_config(
    path: Union[str, epath.Path],
    config_fname: str = _CONFIG_FNAME
) -> config_dict.ConfigDict:
  """Loads BC config from checkpoint."""
  path = epath.Path(path)
  config_path = path / config_fname
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[bc_networks.BCNetworks],
    deterministic: bool = True,
    config_fname: Optional[Union[str, epath.Path]] = _CONFIG_FNAME,
):
  """Loads policy inference function from BC checkpoint.

  The policy is always deterministic.
  """
  path = epath.Path(path)
  config = load_config(path, config_fname=config_fname)
  params = load(path)
  bc_network = _get_bc_network(config, network_factory)
  make_inference_fn = bc_networks.make_inference_fn(bc_network)

  return make_inference_fn(params, deterministic=deterministic)
