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

"""Wrappers for Brax and Gym env."""

from typing import ClassVar, Optional

from brax import jumpy as jp
from brax.envs import env as brax_env
import gym
from gym import spaces
from gym.vector import utils
import jax


class VectorWrapper(brax_env.Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: brax_env.Env, batch_size: int):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    rng = jp.random_split(rng, self.batch_size)
    return jp.vmap(self.env.reset)(rng)

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    return jp.vmap(self.env.step)(state, action)


class EpisodeWrapper(brax_env.Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: brax_env.Env, episode_length: int,
               action_repeat: int):
    super().__init__(env)
    if hasattr(self.unwrapped, 'sys'):
      self.unwrapped.sys.config.dt *= action_repeat
      self.unwrapped.sys.config.substeps *= action_repeat
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(())
    state.info['truncation'] = jp.zeros(())
    return state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    state = self.env.step(state, action)
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    done = jp.where(steps >= self.episode_length, one, state.done)
    state.info['truncation'] = jp.where(steps >= self.episode_length,
                                        1 - state.done, zero)
    state.info['steps'] = steps
    return state.replace(done=done)


class AutoResetWrapper(brax_env.Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    state = self.env.reset(rng)
    state.info['first_qp'] = state.qp
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    qp = jp.tree_map(where_done, state.info['first_qp'], state.qp)
    obs = where_done(state.info['first_obs'], state.obs)
    return state.replace(qp=qp, obs=obs)


class GymWrapper(gym.Env):
  """A wrapper that converts Brax Env to one that follows Gym API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: brax_env.Env,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.sys.config.dt
    }
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs_high = jp.inf * jp.ones(self._env.observation_size, dtype='float32')
    self.observation_space = spaces.Box(-obs_high, obs_high, dtype='float32')

    action_high = jp.ones(self._env.action_size, dtype='float32')
    self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

    def reset(key):
      key1, key2 = jp.random_split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      return state, state.obs, state.reward, state.done, state.info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='human'):
    # pylint:disable=g-import-not-at-top
    from brax.io import image
    if mode == 'rgb_array':
      sys, qp = self._env.sys, self._state.qp
      return image.render_array(sys, qp, 256, 256)
    else:
      return super().render(mode=mode)  # just raise an exception


class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: brax_env.Env,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.sys.config.dt
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs_high = jp.inf * jp.ones(self._env.observation_size, dtype='float32')
    self.single_observation_space = spaces.Box(
        -obs_high, obs_high, dtype='float32')
    self.observation_space = utils.batch_space(self.single_observation_space,
                                               self.num_envs)

    action_high = jp.ones(self._env.action_size, dtype='float32')
    self.single_action_space = spaces.Box(
        -action_high, action_high, dtype='float32')
    self.action_space = utils.batch_space(self.single_action_space,
                                          self.num_envs)

    def reset(key):
      key1, key2 = jp.random_split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      return state, state.obs, state.reward, state.done, state.info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='human'):
    # pylint:disable=g-import-not-at-top
    from brax.io import image
    if mode == 'rgb_array':
      sys = self._env.sys
      qp = jp.take(self._state.qp, 0)
      return image.render_array(sys, qp, 256, 256)
    else:
      return super().render(mode=mode)  # just raise an exception
