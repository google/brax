from brax.envs.base import Env, State, Wrapper
from brax.training import types
from brax.training.types import Params
import jax.numpy as jp
import jax

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs import training
from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs import training

class VisionEpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end.
  
  Compared to training.EpisodeWrapper, this wrapper avoids scanning 
  env.step and assumes that action repeat is already implemented by env.
  This avoids costly scanning over costly, unused render calls.
  """

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    return state

  def step(self, state: State, action: jax.Array) -> State:
    state = self.env.step(state, action)
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps
    return state.replace(done=done)

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
  
  env = VisionEpisodeWrapper(env, episode_length, action_repeat)
  if randomization_fn is None:
    env = training.VmapWrapper(env)
  else:
    env = training.DomainRandomizationVmapWrapper(env, randomization_fn)
  env = training.AutoResetWrapper(env)
  return env
