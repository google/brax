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

"""Wrappers for Brax and Gym env."""

import copy
from typing import ClassVar, Dict, Optional

from brax.v1 import jumpy as jp
from brax.v1.envs import env as brax_env
import dm_env
from dm_env import specs
import flax
import gym
from gym import spaces
from gym.vector import utils
import jax
import jax.numpy as jnp


def wrap_for_training(env: brax_env.Env,
                      episode_length: int = 1000,
                      action_repeat: int = 1) -> brax_env.Wrapper:
  """Common wrapper pattern for all training agents.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  env = EpisodeWrapper(env, episode_length, action_repeat)
  batched = False
  if hasattr(env, 'custom_tree_in_axes'):
    batch_indices, _ = jax.tree_util.tree_flatten(env.custom_tree_in_axes)
    if 0 in batch_indices:
      batched = True
  if not batched:
    env = VmapWrapper(env)
  env = AutoResetWrapper(env)
  return env


class VectorWrapper(brax_env.Wrapper):
  """DEPRECATED Vectorizes Brax env. Use VmapWrapper instead."""

  def __init__(self, env: brax_env.Env, batch_size: int):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    rng = jp.random_split(rng, self.batch_size)
    return jp.vmap(self.env.reset)(rng)

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    return jp.vmap(self.env.step)(state, action)


class VmapWrapper(brax_env.Wrapper):
  """Vectorizes Brax env."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    return jp.vmap(self.env.reset)(rng)

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    return jp.vmap(self.env.step)(state, action)


class EpisodeWrapper(brax_env.Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: brax_env.Env, episode_length: int,
               action_repeat: int):
    super().__init__(env)
    # For proper video speed.
    # TODO: Fix dt book keeping with action repeats so there isn't
    # async between sys and env steps.
    if hasattr(env, 'sys'):
      self.sys = copy.deepcopy(env.sys)
      self.sys.config.dt *= action_repeat
      self.sys.config.substeps *= action_repeat
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    return state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jp.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(steps >= episode_length,
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


@flax.struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

    Args:
        episode_metrics: Aggregated episode metrics since the beginning of the
          episode.
        active_episodes: Boolean vector tracking which episodes are not done
          yet.
        episode_steps: Integer vector tracking the number of steps in
          the episode.
  """
  episode_metrics: Dict[str, jp.ndarray]
  active_episodes: jp.ndarray
  episode_steps: jp.ndarray


class EvalWrapper(brax_env.Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jp.ndarray) -> brax_env.State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(jp.zeros_like,
                                               reset_state.metrics),
        active_episodes=jp.ones_like(reset_state.reward),
        episode_steps=jp.zeros_like(reset_state.reward))
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}')
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['reward'] = nstate.reward
    episode_steps = jp.where(state_metrics.active_episodes,
                             nstate.info['steps'], state_metrics.episode_steps)
    episode_metrics = jax.tree_util.tree_map(
        lambda a, b: a + b * state_metrics.active_episodes,
        state_metrics.episode_metrics, nstate.metrics)
    active_episodes = state_metrics.active_episodes * (1 - nstate.done)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps)
    nstate.info['eval_metrics'] = eval_metrics
    return nstate


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
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    # We return device arrays for pytorch users.
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='human'):
    # pylint:disable=g-import-not-at-top
    from brax.v1.io import image
    if mode == 'rgb_array':
      assert self._state is not None
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
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

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
    from brax.v1.io import image
    if mode == 'rgb_array':
      sys = self._env.sys
      qp = jp.take(self._state.qp, 0)  # pytype: disable=wrong-arg-types  # jax-ndarray
      return image.render_array(sys, qp, 256, 256)
    else:
      return super().render(mode=mode)  # just raise an exception


class DmEnvWrapper(dm_env.Environment):
  """A wrapper that converts Brax Env to one that follows Dm Env API."""

  def __init__(self,
               env: brax_env.Env,
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
      action_high = jp.ones(self._env.action_size, dtype='float32')
      self._action_spec = specs.BoundedArray((self._env.action_size,),
                                             minimum=-action_high,
                                             maximum=action_high,
                                             dtype='float32',
                                             name='action')

    self._reward_spec = specs.Array(
        shape=(), dtype=jnp.dtype('float32'), name='reward')
    self._discount_spec = specs.BoundedArray(
        shape=(), dtype='float32', minimum=0., maximum=1., name='discount')
    if hasattr(self._env, 'discount_spec'):
      self._discount_spec = self._env.discount_spec()

    def reset(key):
      key1, key2 = jp.random_split(key)
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
    # pylint:disable=g-import-not-at-top
    from brax.v1.io import image
    assert self._state is not None
    sys, qp = self._env.sys, self._state.qp
    return image.render_array(sys, qp, 256, 256)
