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

"""Wraps the core environment with some extra statistics for training.

*This is branched from training/env.py, and will be folded back later.*
"""

from typing import Callable, Dict, Tuple
from brax import envs
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class EnvState:
  """Contains training state for the learner."""
  core: envs.State
  total_metrics: Dict[str, jnp.ndarray]
  total_episodes: jnp.ndarray


Action = jnp.ndarray
StepFn = Callable[[EnvState, Action], EnvState]


def wrap(core_env: envs.Env,
         rng: jnp.ndarray,
         extra_step_kwargs: bool = True) -> Tuple[EnvState, StepFn]:
  """Returns a wrapped state and step function for training."""
  first_core = jax.jit(core_env.reset)(rng)
  first_core.metrics['reward'] = first_core.reward
  first_core.metrics.update(
      {f'reward/{k}': v for k, v in first_core.info.get('rewards', {}).items()})
  first_core.metrics['score'] = first_core.info.get(
      'score', jnp.zeros_like(first_core.reward))
  first_core.metrics.update(
      {f'score/{k}': v for k, v in first_core.info.get('scores', {}).items()})
  first_total_metrics = jax.tree_map(jnp.sum, first_core.metrics)
  first_total_episodes = jnp.zeros(())

  first_state = EnvState(
      core=first_core,
      total_metrics=first_total_metrics,
      total_episodes=first_total_episodes)

  def step(state: EnvState,
           action: Action,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> EnvState:
    if extra_step_kwargs:
      core = core_env.step(
          state.core,
          action,
          normalizer_params=normalizer_params,
          extra_params=extra_params)
    else:
      core = core_env.step(state.core, action)
    core.metrics['reward'] = core.reward
    core.metrics.update(
        {f'reward/{k}': v for k, v in core.info.get('rewards', {}).items()})
    core.metrics['score'] = core.info.get('score', jnp.zeros_like(core.reward))
    core.metrics.update(
        {f'score/{k}': v for k, v in core.info.get('scores', {}).items()})
    total_metrics = jax.tree_multimap(lambda a, b: a + jnp.sum(b),
                                      state.total_metrics, core.metrics)
    total_episodes = state.total_episodes + jnp.sum(core.done)
    return EnvState(
        core=core, total_metrics=total_metrics, total_episodes=total_episodes)

  return first_state, jax.jit(step)
