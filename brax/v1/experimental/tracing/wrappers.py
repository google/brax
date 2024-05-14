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

"""Wrappers that expose config data to jax tracing."""

from typing import Callable, Dict, List, Tuple, TypeVar, Union

from brax.v1 import pytree
from brax.v1.envs import env as brax_env
from brax.v1.experimental.tracing.customize import TracedConfig
import brax.v1.jumpy as jp
import jax

Pytree = TypeVar('Pytree')


class DomainRandomizationWrapper(brax_env.Wrapper):
  """Wraps environment methods to allow vectorization over config state."""

  def __init__(self, env_fn: Callable[..., brax_env.Env],
               custom_tree: Dict[str, Union[List[Pytree], Pytree,
                                            brax_env.State, jp.ndarray]],
               custom_tree_in_axes: Tuple[Dict[str, Union[List[Pytree], Pytree,
                                                          int]]]):
    self.custom_tree = custom_tree
    self.custom_tree_in_axes = custom_tree_in_axes
    self.env_fn = pytree.register(env_fn)

    def init_env_fn(custom_tree):

      def post_process_fn(config):
        config = TracedConfig(config, custom_tree=custom_tree)
        return config

      env = self.env_fn(post_process_fn=post_process_fn)
      return env

    v_init = jax.vmap(init_env_fn, in_axes=custom_tree_in_axes)
    env = jax.jit(v_init)(custom_tree)

    super().__init__(env)

  def reset(self, rng):

    def reset_fn(custom_tree):

      def post_process_fn(config):
        config = TracedConfig(config, custom_tree=custom_tree)
        return config

      env = self.env_fn(post_process_fn=post_process_fn)
      return env.reset(custom_tree['rng'])

    self.custom_tree_in_axes[0]['rng'] = 0
    self.custom_tree['rng'] = rng

    return jax.vmap(
        reset_fn, in_axes=self.custom_tree_in_axes)(
            self.custom_tree)

  def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:

    def step_fn(custom_tree):

      def post_process_fn(config):
        config = TracedConfig(config, custom_tree=custom_tree)
        return config

      env = self.env_fn(post_process_fn=post_process_fn)
      return env.step(custom_tree['state'], custom_tree['action'])

    self.custom_tree_in_axes[0]['state'] = jax.tree.map(
        lambda x: 0 if x.shape else None, state)
    self.custom_tree_in_axes[0]['action'] = 0
    self.custom_tree['state'] = state
    self.custom_tree['action'] = action

    return jax.vmap(step_fn, in_axes=self.custom_tree_in_axes)(self.custom_tree)

  @property
  def observation_size(self) -> int:
    env = self.env_fn()
    return env.observation_size

  @property
  def action_size(self) -> int:
    env = self.env_fn()
    return env.action_size
