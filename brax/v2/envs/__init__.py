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
"""Environments for training and evaluating policies."""

from typing import Optional, Type

from brax.v2.envs import ant
from brax.v2.envs import half_cheetah
from brax.v2.envs import hopper
from brax.v2.envs import humanoid
from brax.v2.envs import humanoidstandup
from brax.v2.envs import inverted_double_pendulum
from brax.v2.envs import inverted_pendulum
from brax.v2.envs import reacher
from brax.v2.envs import walker2d
from brax.v2.envs import wrapper
from brax.v2.envs.env import Env


_envs = {
    'ant': ant.Ant,
    'halfcheetah': half_cheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoidstandup': humanoidstandup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'reacher': reacher.Reacher,
    'walker2d': walker2d.Walker2d,
}


def get_environment(env_name: str, **kwargs) -> Env:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
  """Adds an environment to the registry.

  Args:
    env_name: environment name string
    env_class: the Env class to add to the registry
  """
  _envs[env_name] = env_class


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword arugments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  env = _envs[env_name](**kwargs)

  if episode_length is not None:
    env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrapper.VmapWrapper(env, batch_size)
  if auto_reset:
    env = wrapper.AutoResetWrapper(env)

  return env  # type: ignore
