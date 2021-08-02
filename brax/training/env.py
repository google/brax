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
import flax
import jax
import jax.numpy as jnp
from brax import envs


@flax.struct.dataclass
class EnvState:
  """Contains training state for the learner."""
  core: envs.State
  total_metrics: Dict[str, jnp.ndarray]
  total_episodes: jnp.ndarray

Action = jnp.ndarray
StepFn = Callable[[EnvState, Action], EnvState]


def wrap(core_env: envs.Env, rng: jnp.ndarray) -> Tuple[EnvState, StepFn]:
  """Returns a wrapped state and step function for training."""
  rng = jax.random.split(rng, core_env.batch_size)

  first_core = core_env.reset(rng)
  first_core.metrics['reward'] = first_core.reward
  first_total_metrics = jax.tree_map(jnp.sum, first_core.metrics)
  first_total_episodes = jnp.zeros(())

  first_state = EnvState(
      core=first_core,
      total_metrics=first_total_metrics,
      total_episodes=first_total_episodes)

  def step(state: EnvState, action: Action) -> EnvState:
    core = core_env.step(state.core, action)
    core.metrics['reward'] = core.reward
    def test_done(a, b):
      if a is first_core.done or a is first_core.metrics or a is first_core.reward:
        return b
      test_shape = [a.shape[0],] + [1 for _ in range(len(a.shape) - 1)]
      return jnp.where(jnp.reshape(core.done, test_shape), a, b)
    core = jax.tree_multimap(test_done, first_state.core, core)
    total_metrics = jax.tree_multimap(lambda a, b: a + jnp.sum(b),
                                      state.total_metrics, core.metrics)
    total_episodes = state.total_episodes + jnp.sum(core.done)
    return EnvState(
        core=core,
        total_metrics=total_metrics,
        total_episodes=total_episodes)

  return first_state, jax.jit(step)
