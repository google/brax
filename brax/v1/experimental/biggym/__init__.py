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

"""BIG-Gym: crowd-sourced environments and behaviors."""
# pylint:disable=protected-access
# pylint:disable=g-complex-comprehension
import difflib
import functools
import importlib
import inspect
import itertools
from typing import Any, Union, Dict, Callable, Optional
from brax.v1 import envs as brax_envs
from brax.v1.envs import Env
from brax.v1.envs import wrappers
from brax.v1.experimental.biggym import registry
from brax.v1.experimental.biggym import tasks
from brax.v1.experimental.braxlines.envs import obs_indices
from brax.v1.experimental.composer import components as composer_components
from brax.v1.experimental.composer import composer
from brax.v1.experimental.composer import envs as composer_envs

is_multiagent = composer.is_multiagent
unwrap = composer.unwrap

ROOT_PATH = 'brax.experimental.biggym.registry'
ENVS = {}
REGISTRIES = {}
OPEN_ENDED_TRACKS = ('rl', 'mimax')
GOAL_ORIENTED_TRACKS = sorted(tasks.TASKS)
ENVS_BY_TRACKS = dict(
    open_ended={k: () for k in OPEN_ENDED_TRACKS},
    goal_oriented={k: () for k in GOAL_ORIENTED_TRACKS},
    goal_oriented_matches={
        k: () for k in tasks.SYMMETRIC_MA_TASKS + tasks.ASYMMETRIC_MA_TASKS
    },
)
COMPONENTS_BY_TRACKS = {k: () for k in tasks.TASKS}


def inspect_env(env_name: str):
  """Inspect env_params of an env (ComposerEnv only)."""
  assert_exists(env_name)
  if composer_envs.exists(env_name):
    return composer_envs.inspect_env(env_name)
  else:
    return {}, False


def assert_env_params(env_name: str,
                      env_params: Dict[str, Any],
                      ignore_kwargs: bool = True):
  """Inspect env_params of an env (ComposerEnv only)."""
  assert_exists(env_name)
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


def assert_exists(env_name: str):
  """Assert if an environment is registered."""
  exists_ = exists(env_name)
  if not exists_:
    closest = difflib.get_close_matches(env_name, list_env(), n=3)
    assert 0, f'{env_name} not found. Closest={closest}'


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


def register_all(verbose: bool = False, **kwargs):
  """Register all registries."""
  for registry_name in registry.REGISTRIES:
    env_names, comp_names, task_env_names, _ = register(registry_name, **kwargs)
    if verbose:
      print((f'Registered {registry_name}: '
             f'{len(env_names)} envs, '
             f'{len(comp_names)} comps, '
             f'{len(task_env_names)} task_envs, '))


def register(registry_name: str,
             assert_override: bool = True,
             assert_format: bool = True,
             optional: bool = True):
  """Register all envs and components."""
  global ENVS, REGISTRIES, ENVS_BY_TRACKS

  assert (optional or registry_name not in REGISTRIES
         ), f'non-optional register() conflicts: {registry_name}'
  if registry_name in REGISTRIES:
    return REGISTRIES[registry_name]

  lib = importlib.import_module(f'{ROOT_PATH}.{registry_name}')
  metadata = {}
  if assert_format:  # assert submission format
    for attr in ('AUTHORS', 'CONTACTS', 'AFFILIATIONS', 'DESCRIPTIONS'):
      assert hasattr(lib, attr), f'{attr} must be defined as a tuple of strs'
      values = getattr(lib, attr)
      assert isinstance(
          values, tuple), f'{attr}={values} must be defined as a tuple of strs'
      assert all(isinstance(v, str) for v in values
                ), f'{attr}={values} must be defined as a tuple of strs'
      metadata[attr] = values

  envs = lib.ENVS or {}
  components = lib.COMPONENTS or {}
  envs = {registry.get_env_name(registry_name, k): v for k, v in envs.items()}
  components = {
      registry.get_comp_name(registry_name, k): v
      for k, v in components.items()
  }
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
    tracks = env_info.get('tracks', ['rl'])
    for track in tracks:
      assert track in OPEN_ENDED_TRACKS, f'{track} not in {OPEN_ENDED_TRACKS}'
      ENVS_BY_TRACKS['open_ended'][track] += (env_name,)
    if 'mimax' in tracks:
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
    comp_lib = composer_components.register_component(
        comp_name,
        load_path=f'{ROOT_PATH}.{registry_name}.components.{comp_module}',
        override=True)
    component_params = get_func_kwargs(comp_lib.get_specs)[0]
    for track in comp_info.get('tracks', []):
      assert (track
              in GOAL_ORIENTED_TRACKS), f'{track} not in {GOAL_ORIENTED_TRACKS}'
      track_env_name = tasks.get_task_env_name(track, comp_name)
      if assert_override:
        assert not exists(
            track_env_name), f'{list_env()} contains {track_env_name}'
      track_env_module = functools.partial(tasks.TASKS[track], comp_name,
                                           **component_params)
      # register a ComposerEnv
      composer_envs.register_env(
          track_env_name, track_env_module, override=True)
      task_envs += [track_env_name]
      ENVS_BY_TRACKS['goal_oriented'][track] += (track_env_name,)
      COMPONENTS_BY_TRACKS[track] += (comp_name,)

  assert envs or task_envs, 'no envs registered'
  REGISTRIES[registry_name] = (sorted(envs), sorted(components),
                               sorted(task_envs), metadata)
  return REGISTRIES[registry_name]


def register_match(track: str, comp1: str, comp2: str, assert_override=True):
  """Register a match."""
  comp1_lib = composer_components.register_component(comp1)
  comp1_params = get_func_kwargs(comp1_lib.get_specs)[0]
  comp2_lib = composer_components.register_component(comp2)
  comp2_params = get_func_kwargs(comp2_lib.get_specs)[0]
  track_env_name = tasks.get_match_env_name(track, comp1, comp2)
  if assert_override:
    assert not exists(track_env_name), f'{list_env()} contains {track_env_name}'
  track_env_module = functools.partial(
      tasks.TASKS[track],
      comp1,
      opponent=comp2,
      opponent_params=comp2_params,
      **comp1_params)
  # register a ComposerEnv
  composer_envs.register_env(track_env_name, track_env_module, override=True)
  ENVS_BY_TRACKS['goal_oriented_matches'][track] += (track_env_name,)
  return track_env_name


def register_matches(assert_override=True):
  """Register components by goal-oriented track to fight each other."""
  task_envs = []
  for track in tasks.SYMMETRIC_MA_TASKS:
    for comp1, comp2 in itertools.combinations(COMPONENTS_BY_TRACKS[track], 2):
      task_envs += [
          register_match(track, comp1, comp2, assert_override=assert_override)
      ]
  for track in tasks.ASYMMETRIC_MA_TASKS:
    for comp1, comp2 in itertools.product(COMPONENTS_BY_TRACKS[track],
                                          COMPONENTS_BY_TRACKS[track]):
      task_envs += [
          register_match(track, comp1, comp2, assert_override=assert_override)
      ]
  return task_envs


def create(env_name: str = None,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  assert_exists(env_name)
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
