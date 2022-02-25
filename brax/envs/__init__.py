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

# pylint:disable=g-multiple-import
"""Some example environments to help get started quickly with brax."""

import functools
from typing import Callable, Optional, Union, overload

import brax
from brax.envs import acrobot
from brax.envs import ant
from brax.envs import fast
from brax.envs import fetch
from brax.envs import grasp
from brax.envs import halfcheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import humanoid_standup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import reacher
from brax.envs import reacherangle
from brax.envs import swimmer
from brax.envs import ur5e
from brax.envs import walker2d
from brax.envs import wrappers
from brax.envs.env import Env, State, Wrapper
import gym


_envs = {
    'acrobot': acrobot.Acrobot,
    'ant': ant.Ant,
    'fast': fast.Fast,
    'fetch': fetch.Fetch,
    'grasp': grasp.Grasp,
    'halfcheetah': halfcheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoidstandup': humanoid_standup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'reacher': reacher.Reacher,
    'reacherangle': reacherangle.ReacherAngle,
    'swimmer': swimmer.Swimmer,
    'ur5e': ur5e.Ur5e,
    'walker2d': walker2d.Walker2d,
}


def get_environment(env_name, **kwargs):
  return _envs[env_name](**kwargs)


def create(env_name: str,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           eval_metrics: bool = False,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = _envs[env_name](**kwargs)
  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)
  if eval_metrics:
    env = wrappers.EvalWrapper(env)

  return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, **kwargs)


@overload
def create_gym_env(env_name: str,
                   batch_size: None = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.Env:
  ...


@overload
def create_gym_env(env_name: str,
                   batch_size: int,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.vector.VectorEnv:
  ...


def create_gym_env(env_name: str,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
  """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
  environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
  if batch_size is None:
    return wrappers.GymWrapper(environment, seed=seed, backend=backend)
  if batch_size <= 0:
    raise ValueError(
        '`batch_size` should either be None or a positive integer.')
  return wrappers.VectorGymWrapper(environment, seed=seed, backend=backend)
