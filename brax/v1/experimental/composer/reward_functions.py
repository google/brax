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

"""Reward functions.

Each function has the form:
    reward, done = reward_fn(action, obs_dict, ...)
    If action.shape = (...,action_dim), reward/done shapes are (...,reward_dim)
"""
# pylint:disable=g-long-lambda
import functools
import inspect
from typing import Any, Dict, Union
from brax.v1.experimental.composer.observers import Observer
from brax.v1.experimental.composer.observers import SimObserver as so
from jax import lax
from jax import numpy as jnp


def get_default_kwargs(fn):
  spec = inspect.getfullargspec(fn)
  kwargs = {}
  if spec.defaults:
    kwargs.update(
        {k: v for k, v in zip(spec.args[-len(spec.defaults):], spec.defaults)})
  kwargs.update(spec.kwonlydefaults or {})
  return kwargs


def pop_wrapper_kwargs(reward_kwargs: Dict[str, Any]):
  wrapper_kwargs = get_default_kwargs(wrapper)
  wrapper_kwargs = {
      k: reward_kwargs.pop(k, v) for k, v in wrapper_kwargs.items()
  }
  return reward_kwargs, wrapper_kwargs


def wrapper(reward_fn,
            scale: float = 1.0,
            offset: float = 0.0,
            done_bonus: float = 0.0,
            exclude_from_score: bool = False):
  """Return both scale/offset reward and raw reward."""

  def fn(*args, **kwargs):
    reward, done = reward_fn(*args, **kwargs)
    if exclude_from_score:
      score = jnp.zeros_like(reward)
    else:
      score = reward
    reward = (reward + offset) * scale
    score *= jnp.sign(scale)
    reward = jnp.where(done, reward + done_bonus, reward)
    score = jnp.where(done, score + done_bonus, score)
    return reward, score, done

  return fn


def index_obs_dict(obs_dict: Dict[str, jnp.ndarray], obs: Union[Observer,
                                                                jnp.ndarray]):
  """Index obs_dict with observer."""
  if isinstance(obs, Observer):
    assert obs.name in obs_dict, f'{obs.name} not in {obs_dict.keys()}'
    obs = obs_dict[obs.name]
  return jnp.array(obs)


def constant_reward(action: jnp.ndarray,
                    obs_dict: Dict[str, jnp.ndarray],
                    value: float = 1.0):
  """Constant reward."""
  del obs_dict
  reward = jnp.ones(action.shape[:-1]) * value
  return reward, jnp.zeros_like(reward)


def control_reward(action: jnp.ndarray, obs_dict: Dict[str, jnp.ndarray]):
  """Negative Control reward."""
  del obs_dict
  ctrl_cost = jnp.linalg.norm(action, axis=-1)
  return -ctrl_cost, jnp.zeros_like(ctrl_cost)


def exp_norm_reward(action: jnp.ndarray, obs_dict: Dict[str, jnp.ndarray],
                    obs: Observer, **kwargs):
  """Exponential negative norm of an observation as reward."""
  reward, done = distance_reward(action, obs_dict, obs1=obs, obs2=0, **kwargs)
  return jnp.exp(reward), done


def direction_reward(action: jnp.ndarray,
                     obs_dict: Dict[str, jnp.ndarray],
                     vel0: Union[Observer, jnp.ndarray],
                     vel1: Union[Observer, jnp.ndarray],
                     pos0: Union[Observer, jnp.ndarray],
                     pos1: Union[Observer, jnp.ndarray],
                     sign: float = -1.0,
                     norm_kwargs: Dict[str, Any] = None):
  """Positive direction reward based on inner product."""
  del action
  norm_kwargs = norm_kwargs or {}
  vel0 = index_obs_dict(obs_dict, vel0)
  vel1 = index_obs_dict(obs_dict, vel1)
  pos0 = index_obs_dict(obs_dict, pos0)
  pos1 = index_obs_dict(obs_dict, pos1)
  ndim = max(vel0.ndim, vel1.ndim, pos0.ndim, pos1.ndim)
  vel0 = vel0.reshape((1,) * (ndim - vel0.ndim) + vel0.shape)
  vel1 = vel1.reshape((1,) * (ndim - vel1.ndim) + vel1.shape)
  pos0 = pos0.reshape((1,) * (ndim - pos0.ndim) + pos0.shape)
  pos1 = pos1.reshape((1,) * (ndim - pos1.ndim) + pos1.shape)
  agent_sign = jnp.sign(jnp.sum((pos1 - pos0) * vel0, axis=-1))
  opp_sign = jnp.sign(jnp.sum((pos0 - pos1) * vel1, axis=-1))
  # get unit vector & direction
  vel1 /= jnp.linalg.norm(vel1, axis=-1, **norm_kwargs)
  vel1 *= jnp.sign(sign)
  inner_product = lax.cond(
      agent_sign, lambda x: lax.cond(x, lambda y: jnp.sum(vel0 * y, axis=-1),
                                     lambda y: jnp.zeros_like(x), vel1),
      jnp.zeros_like, opp_sign)
  return jnp.clip(inner_product, a_min=0.0), jnp.zeros_like(inner_product)


def norm_reward(action: jnp.ndarray, obs_dict: Dict[str, jnp.ndarray],
                obs: Observer, **kwargs):
  """Negative norm of an observation as reward."""
  return distance_reward(action, obs_dict, obs1=obs, obs2=0, **kwargs)


def state_reward(action: jnp.ndarray, obs_dict: Dict[str, jnp.ndarray],
                 obs: Union[Observer, jnp.ndarray]):
  """State reward."""
  del action
  obs = index_obs_dict(obs_dict, obs)
  obs = jnp.sum(obs, axis=-1)
  done = jnp.zeros_like(obs)
  return obs, done


def distance_reward(action: jnp.ndarray,
                    obs_dict: Dict[str, jnp.ndarray],
                    obs1: Union[Observer, jnp.ndarray],
                    obs2: Union[Observer, jnp.ndarray],
                    max_dist: float = 1e8,
                    min_dist: float = 0,
                    norm_kwargs: Dict[str, Any] = None):
  """Negative distance reward."""
  del action
  norm_kwargs = norm_kwargs or {}
  obs1 = index_obs_dict(obs_dict, obs1)
  obs2 = index_obs_dict(obs_dict, obs2)
  ndim = max(obs1.ndim, obs2.ndim)
  obs1 = obs1.reshape((1,) * (ndim - obs1.ndim) + obs1.shape)
  obs2 = obs2.reshape((1,) * (ndim - obs2.ndim) + obs2.shape)
  delta = obs1 - obs2
  dist = jnp.linalg.norm(delta, axis=-1, **norm_kwargs)
  # instead of clipping, terminate
  # dist = jnp.clip(dist, a_min=min_dist, a_max=max_dist)
  done = jnp.zeros_like(dist)
  done = jnp.where(dist < min_dist, jnp.ones_like(done), done)
  done = jnp.where(dist > max_dist, jnp.ones_like(done), done)
  return -dist, done


def get_reward_fns(*components: Dict[str, Any],
                   reward_type: str = 'root_goal',
                   **reward_kwargs):
  """Get components-based reward functions.

  `reward_type` can be:
    - a string specifying common pre-defined reward functions, e.g. 'root_goal'
    - a callable reward_fn(actions, obs_dict, ...), in which case each default
      value in `reward_kwargs` can also be a function, in which case it will be
      processed as v=v(*components).

  Args:
    *components: component dictionaries from Composer().metadata.components
    reward_type: a str or a callable, specifying a reward function
    **reward_kwargs: kwargs arguments to the reward function

  Returns:
    a callable reward_fn(actions, obs_dict)
  """
  reward_kwargs, wrapper_kwargs = pop_wrapper_kwargs(reward_kwargs)
  if reward_type == 'root_goal':
    assert len(components) == 1, components
    sdtype = reward_kwargs.pop('sdtype', 'body')
    sdcomp = reward_kwargs.pop('sdcomp', 'pos')
    indices = reward_kwargs.pop('indices', None)
    target_goal = reward_kwargs.pop('target_goal')
    target_goal = jnp.array(target_goal)
    reward_fn = functools.partial(
        distance_reward,
        obs1=so(sdtype, sdcomp, components[0]['root'], indices=indices),
        obs2=target_goal,
        **reward_kwargs)
  elif reward_type == 'root_dist':
    assert len(components) == 2, components
    reward_fn = functools.partial(
        distance_reward,
        obs1=so('body', 'pos', components[0]['root']),
        obs2=so('body', 'pos', components[1]['root']),
        **reward_kwargs)
  elif callable(reward_type):
    reward_kwargs = {
        k: v(*components) if callable(v) else v
        for k, v in reward_kwargs.items()
    }
    reward_fn = functools.partial(reward_type, **reward_kwargs)
  else:
    raise NotImplementedError(reward_type)
  return wrapper(reward_fn, **wrapper_kwargs), reward_fn


def get_observers_from_reward_fns(reward_fn):
  """Get observers variable from reward_fn."""
  defaults = get_default_kwargs(reward_fn)
  return [v for _, v in sorted(defaults.items()) if isinstance(v, Observer)]
