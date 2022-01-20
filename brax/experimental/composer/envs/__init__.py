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

"""Environments.

In Braxlines Composer, all environments are defined through
a descriotion dictionary `env_desc`.

composer.py loads from `ENV_DESCS` with `env_name`,
where each entry can be a `env_desc` or a function that returns `env_desc`.
"""
# pylint:disable=protected-access
# pylint:disable=g-complex-comprehension
import copy
import functools
import importlib
import inspect
from typing import Any, Dict

ENV_DESCS = {}
DEFAULT_REGISTER_LIBS = ('sa_descs', 'ma_descs')


def is_env_desc(env_desc: Any):
  """Check if it is appropriate env_desc object."""
  return isinstance(env_desc, dict) or callable(env_desc)


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
  """List registered environments."""
  return sorted(ENV_DESCS)


def exists(env_name: str):
  """If environment is registered."""
  return env_name in ENV_DESCS


def get_func_kwargs(func):
  """Get keyword args of a function."""
  # first, unwrap functools.partial. only extra keyword arguments.
  partial_supported_params = {}
  while isinstance(func, functools.partial):
    partial_supported_params.update(func.keywords)
    func = func.func
  # secondly, inspect the original function for keyword arguments.
  fn_params = inspect.signature(func).parameters
  support_kwargs = any(
      v.kind == inspect.Parameter.VAR_KEYWORD for v in fn_params.values())
  supported_params = {
      k: v.default
      for k, v in fn_params.items()
      if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
      v.default != inspect._empty
  }
  supported_params.update(partial_supported_params)
  return supported_params, support_kwargs


def inspect_env(env_name: str):
  """Inspect parameters of the env."""
  desc = env_name
  if isinstance(desc, str):
    desc = ENV_DESCS[desc]
  assert callable(desc) or isinstance(desc, dict), desc
  if not callable(desc):
    return {}, False
  return get_func_kwargs(desc)


def assert_env_params(env_name: str,
                      env_params: Dict[str, Any],
                      ignore_kwargs: bool = False):
  """Assert env_params are valid parameters for env_name."""
  assert isinstance(env_params, dict), env_params
  supported_params, support_kwargs = inspect_env(env_name)
  # if unnamed **kwargs, then always assert True
  if support_kwargs and not ignore_kwargs:
    return
  assert all(
      k in supported_params
      for k in env_params), f'invalid {env_params} for {supported_params}'
