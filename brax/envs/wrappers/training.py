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

# pylint:disable=g-multiple-import, g-importing-member
"""Wrappers to support Brax training."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all training agents.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized system
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if randomization_fn is None:
    env = VmapWrapper(env)
  else:
    env = DomainRandomizationVmapWrapper(env, randomization_fn)
  env = EpisodeWrapper(env, episode_length, action_repeat)
  env = AutoResetWrapper(env)
  return env


class VmapWrapper(Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: Env, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jax.Array) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)


class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: State, action: jax.Array) -> State:
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

    pipeline_state = jax.tree.map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(pipeline_state=pipeline_state, obs=obs)


@struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

  Attributes:
      episode_metrics: Aggregated episode metrics since the beginning of the
        episode.
      active_episodes: Boolean vector tracking which episodes are not done yet.
      episode_steps: Integer vector tracking the number of steps in the episode.
  """

  episode_metrics: Dict[str, jax.Array]
  active_episodes: jax.Array
  episode_steps: jax.Array


class EvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jax.Array) -> State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(
            jp.zeros_like, reset_state.metrics
        ),
        active_episodes=jp.ones_like(reset_state.reward),
        episode_steps=jp.zeros_like(reset_state.reward),
    )
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: State, action: jax.Array) -> State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}'
      )
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['reward'] = nstate.reward
    episode_steps = jp.where(
        state_metrics.active_episodes,
        nstate.info['steps'],
        state_metrics.episode_steps,
    )
    episode_metrics = jax.tree_util.tree_map(
        lambda a, b: a + b * state_metrics.active_episodes,
        state_metrics.episode_metrics,
        nstate.metrics,
    )
    active_episodes = state_metrics.active_episodes * (1 - nstate.done)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps,
    )
    nstate.info['eval_metrics'] = eval_metrics
    return nstate


class DomainRandomizationVmapWrapper(Wrapper):
  """Wrapper for domain randomization."""

  def __init__(
      self,
      env: Env,
      randomization_fn: Callable[[System], Tuple[System, System]],
  ):
    super().__init__(env)
    self._sys_v, self._in_axes = randomization_fn(self.sys)

  def _env_fn(self, sys: System) -> Env:
    env = self.env
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._sys_v, state, action
    )
    return res


class RunningMeanStd:
    """Computes and updates running mean and variance for online normalization."""

    def __init__(self, shape: Tuple[int], epsilon: float = 1e-8):
        self.mean = jp.zeros(shape)
        self.var = jp.ones(shape)
        self.count = 0.0
        self.epsilon = epsilon

    def update(self, x: jp.ndarray):
        batch_mean = jp.mean(x, axis=0)
        batch_var = jp.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1.0

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ClipVecAction(Wrapper):
    """Wrapper that clips continuous actions to be within the environment's valid range."""

    def __init__(self, env: Env, low: float = -10.0, high: float = 10.0):
        super().__init__(env)
        self.low = jp.array(low)
        self.high = jp.array(high)

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        clipped_action = jp.clip(action, self.low, self.high)
        return self.env.step(state, clipped_action)


class NormalizeVecObservation(Wrapper):
    """Wrapper that normalizes observations using a running mean and standard deviation."""

    def __init__(self, env: Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_rms = RunningMeanStd(shape=self.env.observation_size, epsilon=epsilon)
        self.update_running_mean = True

    def _normalize(self, obs: jp.ndarray) -> jp.ndarray:
        return (obs - self.obs_rms.mean) / jp.sqrt(self.obs_rms.var + self.epsilon)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        if self.update_running_mean:
            self.obs_rms.update(state.obs)
        normalized_obs = self._normalize(state.obs)
        return state.replace(obs=normalized_obs)

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        if self.update_running_mean:
            self.obs_rms.update(state.obs)
        normalized_obs = self._normalize(state.obs)
        return state.replace(obs=normalized_obs)


class NormalizeVecReward(Wrapper):
    """Wrapper that normalizes rewards using a running mean and standard deviation of the accumulated (discounted) rewards."""
    
    def __init__(self, env: Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=(), epsilon=epsilon)
        self.accumulated_reward = None
        self.update_running_mean = True

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        self.accumulated_reward = jp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        done_float = state.done.astype(jp.float32)
        self.accumulated_reward = self.accumulated_reward * self.gamma * (1 - done_float) + state.reward
        if self.update_running_mean:
            self.return_rms.update(self.accumulated_reward[jp.newaxis])
        normalized_reward = state.reward / jp.sqrt(self.return_rms.var + self.epsilon)
        return state.replace(reward=normalized_reward)
