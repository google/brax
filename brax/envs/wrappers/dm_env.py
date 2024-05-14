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

"""Wrappers to convert brax envs to DM Env envs."""
from typing import Optional

from brax.envs.base import PipelineEnv
from brax.io import image
import dm_env
from dm_env import specs
import jax
from jax import numpy as jp
import numpy as np


class DmEnvWrapper(dm_env.Environment):
  """A wrapper that converts Brax Env to one that follows Dm Env API."""

  def __init__(self,
               env: PipelineEnv,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.seed(seed)
    self.backend = backend
    self._state = None

    if hasattr(self._env, 'observation_spec'):
      self._observation_spec = self._env.observation_spec()
    else:
      obs_high = jp.inf * jp.ones(self._env.observation_size, dtype='float32')
      self._observation_spec = specs.BoundedArray((self._env.observation_size,),
                                                  minimum=-obs_high,
                                                  maximum=obs_high,
                                                  dtype='float32',
                                                  name='observation')

    if hasattr(self._env, 'action_spec'):
      self._action_spec = self._env.action_spec()
    else:
      action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
      self._action_spec = specs.BoundedArray((self._env.action_size,),
                                             minimum=action[:, 0],
                                             maximum=action[:, 1],
                                             dtype='float32',
                                             name='action')

    self._reward_spec = specs.Array(shape=(), dtype=jp.dtype('float32'), name='reward')
    self._discount_spec = specs.BoundedArray(
        shape=(), dtype='float32', minimum=0., maximum=1., name='discount')
    if hasattr(self._env, 'discount_spec'):
      self._discount_spec = self._env.discount_spec()

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=jp.float32(1.),
        observation=obs)

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    del info
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID if not done else dm_env.StepType.LAST,
        reward=reward,
        discount=jp.float32(1.),
        observation=obs)

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def reward_spec(self):
    return self._reward_spec

  def discount_spec(self):
    return self._discount_spec

  def render(self):
    sys, state = self._env.sys, self._state
    if state is None:
      raise RuntimeError('must call reset or step before rendering')
    return image.render_array(sys, state.pipeline_state, 256, 256)
