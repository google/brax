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

"""Environments.

In Braxlines Composer, all environments are defined through
a descriotion dictionary `env_desc`.

composer.py loads from `ENV_DESCS` with `env_name`,
where each entry can be a `env_desc` or a function that returns `env_desc`.
"""
import copy
import importlib
import inspect
from typing import Any

ENV_DESCS = {}
DEFAULT_REGISTER_LIBS = ('ant_descs', 'ma_descs')


def register_env(env_name: str, env_desc: Any, override: bool = True):
  """Register env_name and return env_descs."""
  global ENV_DESCS
  if not override and env_name in ENV_DESCS:
    return copy.deepcopy(ENV_DESCS[env_name])
  else:
    ENV_DESCS[env_name] = env_desc
    return copy.deepcopy(ENV_DESCS[env_name])


def register_lib(load_path: str):
  """Register a library of env_names with env_descs."""
  global ENV_DESCS
  if '.' not in load_path:
    load_path = f'brax.experimental.composer.envs.{load_path}'
  env_lib = importlib.import_module(load_path)
  ENV_DESCS.update(env_lib.ENV_DESCS)


def register_default_libs():
  """Register all default env."""
  for load_path in DEFAULT_REGISTER_LIBS:
    register_lib(load_path)


def list_env():
  """List registered envs."""
  return sorted(ENV_DESCS)


def inspect_env(env_name: str):
  """Inspect parameters of the env."""
  desc = env_name
  if isinstance(desc, str):
    desc = ENV_DESCS[desc]
  assert callable(desc) or isinstance(desc, dict), desc
  if not callable(desc):
    return {}, False
  fn_params = inspect.signature(desc).parameters
  supported_params = {k: v.default for k, v in fn_params.items()}
  support_kwargs = 'kwargs' in supported_params
  supported_params.pop('kwargs', None)
  return supported_params, support_kwargs
