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

"""BIG-Gym: crowd-sourced environments and behaviors."""
# pylint:disable=protected-access
import functools
import importlib
from typing import Any, Union, Dict, Callable, Optional
from brax import envs as brax_envs
from brax.envs import Env
from brax.envs import wrappers
from brax.experimental.biggym.tasks import TASKS
from brax.experimental.braxlines.envs import obs_indices
from brax.experimental.composer import components as composer_components
from brax.experimental.composer import composer
from brax.experimental.composer import envs as composer_envs

is_multiagent = composer.is_multiagent
unwrap = composer.unwrap

ROOT_PATH = 'brax.experimental.biggym.registry'
ENVS = {}


def inspect_env(env_name: str):
  """Inspect env_params of an env (ComposerEnv only)."""
  if composer_envs.exists(env_name):
    return composer_envs.inspect_env(env_name)
  else:
    return {}, False


def assert_env_params(env_name: str,
                      env_params: Dict[str, Any],
                      ignore_kwargs: bool = True):
  """Inspect env_params of an env (ComposerEnv only)."""
  if composer_envs.exists(env_name):
    composer_envs.assert_env_params(env_name, env_params, ignore_kwargs)
  else:
    assert not env_params, env_params


def list_env():
  """List registered environments."""
  return sorted(brax_envs._envs) + composer_envs.list_env() + sorted(ENVS)


def exists(env_name: str):
  """If environment is registered."""
  return env_name in list_env()


def register(registry_name: str, assert_override: bool = True):
  """Register all envs and components."""
  global ENVS
  lib = importlib.import_module(f'{ROOT_PATH}.{registry_name}')
  envs = lib.ENVS or {}
  components = lib.COMPONENTS or {}
  task_envs = []

  # register environments
  for env_name, env_info in sorted(envs.items()):
    if assert_override:
      assert not exists(env_name), f'{list_env()} contains {env_name}'
    assert 'module' in env_info, env_info
    env_module = env_info['module']
    if isinstance(env_module, str):  # e.g. jump_cheetah:JumpCheetah
      env_lib_path, env_mod_name = env_module.split(':', 1)
      env_lib = importlib.import_module(
          f'{ROOT_PATH}.{registry_name}.envs.{env_lib_path}')
      env_module = getattr(env_lib, env_mod_name)
    if composer_envs.is_env_desc(env_module):
      # register a ComposerEnv
      composer_envs.register_env(env_name, env_module, override=True)
    else:
      # register a standard Env
      ENVS[env_name] = env_module
    if 'mimax' in env_info.get('tracks', []):
      # (MI-Max only) register obs_indices
      for indices_type, indices in env_info.get('obs_indices', {}).items():
        obs_indices.register_indices(env_name, indices_type, indices)

  # register components
  for comp_name, comp_info in sorted(components.items()):
    if assert_override:
      assert not composer_components.exists(
          comp_name
      ), f'{composer_components.list_components()} contains {comp_name}'
    comp_module = comp_info['module']
    composer_components.register_component(
        comp_name,
        load_path=f'{ROOT_PATH}.{registry_name}.components.{comp_module}',
        override=True)
    for track in comp_info.get('tracks', []):
      assert track in TASKS, f'{track} not in {sorted(TASKS)}'
      track_env_name = f'{track}_{registry_name}_{comp_name}'
      track_env_module = TASKS[track](
          component=comp_module,
          component_params=comp_info.get('component_params', {}))
      if assert_override:
        assert not exists(
            track_env_name), f'{list_env()} contains {track_env_name}'
      # register a ComposerEnv
      composer_envs.register_env(
          track_env_name, track_env_module, override=True)
      task_envs += [track_env_name]

  return sorted(envs), sorted(components), sorted(task_envs)


def create(env_name: str = None,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  if env_name in ENVS:
    env = ENVS[env_name](**kwargs)
    if episode_length is not None:
      env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
      env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
      env = wrappers.AutoResetWrapper(env)
    return env
  else:
    return composer.create(
        env_name,
        episode_length=episode_length,
        action_repeat=action_repeat,
        auto_reset=auto_reset,
        batch_size=batch_size,
        **kwargs)


def create_fn(env_name: str = None, **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name=env_name, **kwargs)
