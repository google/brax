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

"""Reward functions.

Each function has the form:
    reward, done = reward_fn(action, obs_dict, ...)
    If action.shape = (...,action_dim), reward/done shapes are (...,reward_dim)
"""
import functools
import inspect
from typing import Any, Dict, Union
from brax.experimental.composer.observers import Observer
from brax.experimental.composer.observers import SimObserver as so
from jax import numpy as jnp


def pop_wrapper_kwargs(reward_kwargs: Dict[str, Any]):
  scale = reward_kwargs.pop('scale', 1.0)
  offset = reward_kwargs.pop('offset', 0.0)
  exclude_from_score = reward_kwargs.pop('exclude_from_score', False)
  return reward_kwargs, dict(
      scale=scale, offset=offset, exclude_from_score=exclude_from_score)


def wrapper(reward_fn,
            scale: float = 1.0,
            offset: float = 0.0,
            exclude_from_score: bool = False):
  """Return both scale/offset reward and raw reward."""

  def fn(*args, **kwargs):
    reward, done = reward_fn(*args, **kwargs)
    if exclude_from_score:
      score = jnp.zeros_like(reward)
    else:
      score = reward
    return (reward + offset) * scale, score, done

  return fn


def constant_reward(action: jnp.ndarray,
                    obs_dict: Dict[str, jnp.ndarray],
                    value: float = 1.0):
  """Constant reward."""
  del obs_dict
  reward = jnp.ones(action.shape[:-1]) * value
  return reward, jnp.zeros_like(reward)


def index_obs_dict(obs_dict: Dict[str, jnp.ndarray], obs: Union[Observer,
                                                                jnp.ndarray]):
  """Index obs_dict with observer."""
  if isinstance(obs, Observer):
    assert obs.name in obs_dict, f'{obs.name} not in {obs_dict.keys()}'
    obs = obs_dict[obs.name]
  return obs


def distance_reward(action: jnp.ndarray,
                    obs_dict: Dict[str, jnp.ndarray],
                    obs1: Union[Observer, jnp.ndarray],
                    obs2: Union[Observer, jnp.ndarray],
                    epsilon: float = 0.0,
                    max_dist: float = 1e8,
                    min_dist: float = 0,
                    norm_kwargs: Dict[str, Any] = None):
  """Distance reward."""
  del action
  norm_kwargs = norm_kwargs or {}
  obs1 = index_obs_dict(obs_dict, obs1)
  obs2 = index_obs_dict(obs_dict, obs2)
  ndim = max(obs1.ndim, obs2.ndim)
  obs1 = obs1.reshape((1,) * (ndim - obs1.ndim) + obs1.shape)
  obs2 = obs2.reshape((1,) * (ndim - obs2.ndim) + obs2.shape)
  delta = obs1 - obs2
  dist = jnp.linalg.norm(delta, axis=-1, **norm_kwargs)
  dist = jnp.clip(dist, a_min=min_dist, a_max=max_dist)
  done = jnp.where(epsilon > 0, dist < epsilon, jnp.zeros(dist.shape[:-1]))
  return -dist, done


def get_edge_reward_fns(component1: Dict[str, Any],
                        component2: Dict[str, Any],
                        reward_type: str = 'root_dist',
                        **reward_kwargs):
  """Get edge-based reward functions."""
  reward_kwargs, wrapper_kwargs = pop_wrapper_kwargs(reward_kwargs)
  if reward_type == 'root_dist':
    reward_fn = functools.partial(
        distance_reward,
        obs1=so('body', 'pos', component1['root']),
        obs2=so('body', 'pos', component2['root']),
        **reward_kwargs)
  else:
    raise NotImplementedError(reward_type)
  return wrapper(reward_fn, **wrapper_kwargs), reward_fn


def get_component_reward_fns(component: Dict[str, Any],
                             reward_type: str = 'root_goal',
                             **reward_kwargs):
  """Get component-based reward functions."""
  reward_kwargs, wrapper_kwargs = pop_wrapper_kwargs(reward_kwargs)
  if reward_type == 'root_goal':
    sdtype = reward_kwargs.pop('sdtype', 'body')
    sdcomp = reward_kwargs.pop('sdcomp', 'pos')
    indices = reward_kwargs.pop('indices', None)
    target_goal = reward_kwargs.pop('target_goal')
    target_goal = jnp.array(target_goal)
    reward_fn = functools.partial(
        distance_reward,
        obs1=so(sdtype, sdcomp, component['root'], indices=indices),
        obs2=target_goal,
        **reward_kwargs)
  else:
    raise NotImplementedError(reward_type)
  return wrapper(reward_fn, **wrapper_kwargs), reward_fn


def get_observers_from_reward_fns(reward_fn):
  """Get observers variable from reward_fn."""
  defaults = inspect.getfullargspec(reward_fn).kwonlydefaults or {}
  return [v for _, v in sorted(defaults.items()) if isinstance(v, Observer)]
