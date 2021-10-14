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

"""Wraps the core environment with some extra statistics for training."""

from typing import Callable, Dict, Tuple

from brax import envs
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class EvalEnvState:
  """Env state + eval metrics."""
  core: envs.State
  current_episode_metrics: Dict[str, jnp.ndarray]
  completed_episodes_metrics: Dict[str, jnp.ndarray]
  completed_episodes: jnp.ndarray
  completed_episodes_steps: jnp.ndarray


Action = jnp.ndarray
EvalStepFn = Callable[[EvalEnvState, Action], EvalEnvState]


def wrap_for_eval(core_env: envs.Env,
                  rng: jnp.ndarray) -> Tuple[EvalEnvState, EvalStepFn]:
  """Returns a wrapped state and step function for training."""
  first_core = jax.jit(core_env.reset)(rng)
  first_core.metrics['reward'] = first_core.reward

  def step(state: EvalEnvState, action: Action) -> EvalEnvState:
    core = core_env.step(state.core, action)
    core.metrics['reward'] = core.reward
    # steps stores the highest step reached when done = True, and then
    # the next steps becomes action_repeat
    completed_episodes_steps = state.completed_episodes_steps + jnp.sum(
        core.info['steps'] * core.done)
    current_episode_metrics = jax.tree_multimap(lambda a, b: a + b,
                                                state.current_episode_metrics,
                                                core.metrics)
    completed_episodes = state.completed_episodes + jnp.sum(core.done)
    completed_episodes_metrics = jax.tree_multimap(
        lambda a, b: a + jnp.sum(b * core.done),
        state.completed_episodes_metrics, current_episode_metrics)
    current_episode_metrics = jax.tree_multimap(
        lambda a, b: a * (1 - core.done) + b * core.done,
        current_episode_metrics, core.metrics)

    return EvalEnvState(
        core=core,
        current_episode_metrics=current_episode_metrics,
        completed_episodes_metrics=completed_episodes_metrics,
        completed_episodes=completed_episodes,
        completed_episodes_steps=completed_episodes_steps)

  first_state = EvalEnvState(
      core=first_core,
      current_episode_metrics=jax.tree_map(jnp.zeros_like, first_core.metrics),
      completed_episodes_metrics=jax.tree_map(
          lambda x: jnp.zeros_like(jnp.sum(x)), first_core.metrics),
      completed_episodes=jnp.zeros(()),
      completed_episodes_steps=jnp.zeros(()))
  return first_state, jax.jit(step)
