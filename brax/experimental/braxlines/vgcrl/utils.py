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

"""Utilities for goal-based RL and empowerment/unsupervised RL.

See:
  VGCRL https://arxiv.org/abs/2106.01404
"""


import functools
from typing import Callable, Dict
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.env import Env
from brax.envs.env import State
from brax.experimental.braxlines.common import dist_utils
from brax.training import networks
from brax.training.ppo import StepData
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions

DISC_PARAM_NAME = 'vgcrl_disc_params'


class Discriminator(object):
  """Container for p(z), q(z|o) in empowerment."""

  def __init__(
      self,
      q_fn='indexing',
      q_fn_params=None,
      dist_p='Uniform',
      dist_p_params=None,
      dist_q='FixedSigma',
      dist_q_params=None,
      ll_q_offset='auto',
      z_size=0,
      logits_clip_range=None,
      param_name: str = DISC_PARAM_NAME,
  ):
    self.z_size = z_size
    self.ll_q_offset = ll_q_offset
    self.model = None
    self.param_name = param_name
    self.q_fn_str = q_fn
    self.q_fn_params = q_fn_params or {}

    # define dist_params_to_dist for q_z_o
    dist_q_params = dist_q_params or {}
    q_scale = dist_q_params.get('scale', 1.)
    assert z_size
    if dist_q == 'FixedSigma':
      self.dist_q_fn = lambda x: tfd.MultivariateNormalDiag(
          x,
          jnp.ones(self.z_size) * q_scale,
      )
      # if an environment terminates, it's useful to ensure reward is positive,
      #  this ensures that within 3*std the log likelihood is positive.
      if self.ll_q_offset == 'auto':
        self.ll_q_offset = -tfd.MultivariateNormalDiag(
            jnp.zeros(self.z_size),
            jnp.ones(self.z_size) * q_scale).log_prob(
                jnp.ones(self.z_size) * q_scale * 3)
    elif dist_q == 'Categorial':
      self.dist_q_fn = lambda x: dist_utils.clipped_onehot_categorical(
          logits=x, clip_range=logits_clip_range)
      if self.ll_q_offset == 'auto':
        self.ll_q_offset = 0.
    else:
      raise NotImplementedError(dist_q)

    # define dist for p_z
    dist_p_params = dist_q_params or {}
    p_scale = dist_p_params.get('scale', 1.)
    if dist_p == 'Uniform':
      self.dist_p_fn = lambda: tfd.Uniform(
          low=jnp.ones(self.z_size) * -p_scale,
          high=jnp.ones(self.z_size) * p_scale,
      )
    elif dist_p == 'UniformCategorial':
      self.dist_p_fn = lambda: tfd.OneHotCategorical(
          logits=jnp.zeros(self.z_size))
    else:
      raise NotImplementedError(dist_p)

    self.initialized = False

  def init_model(self, rng: jnp.ndarray = None):
    """Initialize parts with model parameters."""
    model_params = None
    # define observation_to_dist_params mapping for q_z_o
    q_fn, q_fn_params = self.q_fn_str, self.q_fn_params
    if q_fn == 'indexing':
      indices = q_fn_params.get('indices')
      self.q_fn = lambda params, x: (x.take(indices, axis=-1),)
    elif q_fn == 'mlp':
      input_size = q_fn_params.get('input_size')
      output_size = q_fn_params.get('output_size')
      model = networks.make_model([32, 32, output_size], input_size)
      model_params = model.init(rng)
      self.model = model
      self.q_fn = lambda params, x: (model.apply(params, x),)
    elif q_fn == 'indexing_mlp':
      indices = q_fn_params.get('indices')
      output_size = q_fn_params.get('output_size')
      model = networks.make_model([32, 32, output_size], len(indices))
      model_params = model.init(rng)
      self.model = model
      q_fn_apply = lambda x: x.take(indices, axis=-1)
      self.q_fn = lambda params, x: (model.apply(params, q_fn_apply(x)),)
    else:
      raise NotImplementedError(q_fn)
    self.initialized = True
    return {self.param_name: model_params} if model_params else {}

  def dist_q_z_o(self,
                 data: jnp.ndarray,
                 params: Dict[str, Dict[str, jnp.ndarray]] = None):
    assert self.initialized, 'init_model() must be called'
    param = params.get(self.param_name, {})
    dist_params = self.q_fn(param, data)
    return self.dist_q_fn(*dist_params)

  def sample_p_z(self, batch_size: int, rng: jnp.ndarray):
    """Sample from p(z)."""
    dist_p = self.dist_p_fn()
    if batch_size:
      return dist_p.sample(seed=rng[-1], sample_shape=(batch_size,))
    else:
      return dist_p.sample(seed=rng, sample_shape=())

  def zero_z(self, batch_size: int):
    if batch_size:
      return jnp.zeros((batch_size, self.z_size))
    else:
      return jnp.zeros(self.z_size)

  def ll_q_z_o(self,
               z: jnp.ndarray,
               data: jnp.ndarray,
               params: Dict[str, Dict[str, jnp.ndarray]] = None,
               add_offset: bool = False):
    dist = self.dist_q_z_o(data, params=params)
    ll = dist.log_prob(z)
    if add_offset:
      ll += self.ll_q_offset
    return ll

  def split_obs(self, obs: jnp.ndarray):
    """Split observation."""
    env_obs = obs[..., :-self.z_size]
    z = obs[..., -self.z_size:]
    return env_obs, z

  def concat_obs(self, obs: jnp.ndarray, z: jnp.ndarray):
    """Concat observation."""
    new_obs = jnp.concatenate([obs, z], axis=-1)
    return new_obs


class ParameterizeWrapper(Env):
  """A wrapper that parameterizes Brax Env."""

  def __init__(self, environment: Env, disc: Discriminator):
    self._environment = environment
    self.action_repeat = self._environment.action_repeat
    self.batch_size = self._environment.batch_size
    self.sys = self._environment.sys
    self.disc = disc
    self.z_size = disc.z_size

  def concat(
      self,
      state: State,
      z: jnp.ndarray,
      params: Dict[str, Dict[str, jnp.ndarray]] = None,
      replace_reward: bool = True,
  ) -> State:
    """Concatenate state with param and recompute reward."""
    if replace_reward:
      new_reward = self.disc.ll_q_z_o(
          z, state.obs, params=params, add_offset=True)
      new_reward = jax.lax.stop_gradient(new_reward)
      state = state.replace(reward=new_reward)
    new_obs = jnp.concatenate([state.obs, z], axis=-1)
    state = state.replace(obs=new_obs)
    return state

  def reset(self, rng: jnp.ndarray, z: jnp.ndarray = None) -> State:
    """Resets the environment to an initial state."""
    state = self._environment.reset(rng)
    if z is None:
      z = self.disc.sample_p_z(self.batch_size, rng)
    else:
      assert z.shape[-1] == self.z_size, f'{z.shape}[-1] != {self.z_size}'
    return self.concat(state, z=z, replace_reward=False)

  def step(self,
           state: State,
           action: jnp.ndarray,
           params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    _, z = self.disc.split_obs(state.obs)
    state = self._environment.step(state, action)
    return self.concat(state, z=z, params=params, replace_reward=True)


def disc_loss_fn(data: StepData, rng: jnp.ndarray,
                 params: Dict[str, Dict[str,
                                        jnp.ndarray]], disc: Discriminator):
  """Discriminator loss function."""
  del rng
  disc_loss = 0
  if disc and disc.model:
    env_obs, z = disc.split_obs(data.obs)
    disc_loss = -jnp.mean(disc.ll_q_z_o(z, env_obs, params=params))
  return disc_loss


def create(env_name: str, disc: Discriminator, **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = envs.create(env_name, **kwargs)
  return ParameterizeWrapper(env, disc)


def create_fn(env_name: str, disc: Discriminator,
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, disc, **kwargs)
