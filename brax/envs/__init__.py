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

"""Some example environments to help get started quickly with brax."""

import functools
from typing import Callable

import gym
import brax
from brax.envs import ant
from brax.envs import env
from brax.envs import fetch
from brax.envs import grasp
from brax.envs import halfcheetah
from brax.envs import humanoid
from brax.envs import wrappers

_envs = {
    'fetch': fetch.Fetch,
    'ant': ant.Ant,
    'grasp': grasp.Grasp,
    'halfcheetah': halfcheetah.Halfcheetah,
    'humanoid': humanoid.Humanoid
}
State = env.State
Env = env.Env


def create(env_name: str, **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  return _envs[env_name](**kwargs)


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, **kwargs)


def create_gym_env(env_name: str, **kwargs) -> gym.Env:
  """Creates a Gym Env with a specified brax system."""
  return wrappers.GymWrapper(create(env_name, **kwargs))
