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

"""Utilities for goal-based RL and empowerment/unsupervised RL.

See:
  VGCRL https://arxiv.org/abs/2106.01404
"""


import copy
import functools
from typing import Any, Callable, Dict, Optional, List, Tuple
from brax.v1.envs.env import Env
from brax.v1.envs.env import State
from brax.v1.experimental import normalization
from brax.v1.experimental.braxlines.common import dist_utils
from brax.v1.experimental.composer import composer
from brax.v1.experimental.composer import observers
from brax.training import networks
from brax.training.ppo import StepData
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

DISC_PARAM_NAME = 'vgcrl_disc_params'


class Discriminator(object):
  """Container for p(z), q(z|o) in empowerment."""

  def __init__(self,
               env: Env,
               q_fn='indexing',
               q_fn_params=None,
               dist_p='Uniform',
               dist_p_params=None,
               dist_q='FixedSigma',
               dist_q_params=None,
               z_size=0,
               obs_indices: Optional[List[Any]] = None,
               obs_scale: Optional[List[float]] = None,
               nonnegative_reward: bool = True,
               logits_clip_range: float = 10.0,
               param_name: str = DISC_PARAM_NAME,
               normalize_obs: bool = False,
               spectral_norm: bool = True):
    assert obs_scale is not None
    self.z_size = z_size
    self.env_obs_size = env.observation_size
    self.model = None
    self.param_name = param_name
    self.q_fn_str = q_fn
    self.q_fn_params = q_fn_params or {}
    self.normalize_obs = normalize_obs
    self.obs_indices, self.obs_labels = observers.index_preprocess(
        obs_indices, env)
    self.indexed_obs_size = len(self.obs_indices)
    self.obs_scale = jnp.array(obs_scale or 1.) * jnp.ones(
        self.indexed_obs_size)
    self.normalize_fn = normalization.make_data_and_apply_fn(
        [self.env_obs_size + self.z_size], self.normalize_obs)[1]
    self.spectral_norm = spectral_norm
    self.logits_clip_range = logits_clip_range
    self.nonnegative_reward = nonnegative_reward
    self.ll_q_offset = 0.0

    # define dist_params_to_dist for q_z_o
    dist_q_params = copy.deepcopy(dist_q_params) or {}
    q_scale = dist_q_params.pop('scale', 1.)
    q_scale = jnp.array(q_scale) * jnp.ones(self.z_size)
    assert z_size
    if dist_q == 'FixedSigma':
      self.dist_q_fn = lambda x: tfd.MultivariateNormalDiag(x, q_scale)
      # if an environment terminates, it's useful to ensure reward is positive,
      #  this ensures that within 3*std the log likelihood is positive.
      if self.nonnegative_reward:
        self.ll_q_offset = -tfd.MultivariateNormalDiag(
            jnp.zeros(self.z_size),
            jnp.ones(self.z_size) * q_scale).log_prob(
                jnp.ones(self.z_size) * q_scale * 3)
    elif dist_q == 'Categorial':
      self.dist_q_fn = lambda x: dist_utils.clipped_onehot_categorical(
          logits=x, clip_range=logits_clip_range)
      if self.nonnegative_reward:
        self.ll_q_offset = self.logits_clip_range
    else:
      raise NotImplementedError(dist_q)
    assert not dist_q_params, f'unused dist_q_params: {dist_q_params}'

    # define dist for p_z
    dist_p_params = copy.deepcopy(dist_p_params) or {}
    if dist_p == 'Uniform':
      p_scale = dist_p_params.pop('scale', 1.)
      p_scale = jnp.array(p_scale) * jnp.ones(self.z_size)
      self.dist_p_fn = lambda: tfd.Uniform(low=-p_scale, high=p_scale)
    elif dist_p == 'UniformCategorial':
      self.dist_p_fn = lambda: tfd.OneHotCategorical(
          logits=jnp.zeros(self.z_size))
    elif dist_p == 'Deterministic':
      p_value = dist_p_params.pop('value')
      p_value = jnp.array(p_value) * jnp.ones(self.z_size)
      self.dist_p_fn = lambda: tfd.Deterministic(loc=p_value)
    else:
      raise NotImplementedError(dist_p)
    assert not dist_p_params, f'unused dist_p_params: {dist_p_params}'

    self.initialized = False

  def init_model(self, rng: jnp.ndarray = None):
    """Initialize parts with model parameters."""
    model_params = None
    # define observation_to_dist_params mapping for q_z_o
    q_fn, q_fn_params = self.q_fn_str, self.q_fn_params
    if q_fn == 'indexing':
      self.q_fn = lambda params, x: (self.index_obs(x),)
    elif q_fn == 'mlp':
      input_size = q_fn_params.get('input_size')
      output_size = q_fn_params.get('output_size')
      model = networks.make_model([32, 32, output_size],
                                  input_size,
                                  spectral_norm=self.spectral_norm)
      self.model = model
      if self.spectral_norm:
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        model_params = model.init(rng1, rng2)
        self.q_fn = lambda params, x: (model.apply(
            params, x, rngs={'sing_vec': rng3}, mutable=['sing_vec'])[0],)
      else:
        model_params = model.init(rng)
        self.q_fn = lambda params, x: (model.apply(params, x),)
    elif q_fn == 'indexing_mlp':
      output_size = q_fn_params.get('output_size')
      model = networks.make_model([32, 32, output_size],
                                  len(self.obs_indices),
                                  spectral_norm=self.spectral_norm)
      self.model = model
      if self.spectral_norm:
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        model_params = model.init(rng1, rng2)
        self.q_fn = lambda params, x: (model.apply(
            params,
            self.index_obs(x),
            rngs={'sing_vec': rng3},
            mutable=['sing_vec'])[0],)
      else:
        model_params = model.init(rng)
        self.q_fn = lambda params, x: (model.apply(params, self.index_obs(x)),)
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

  def dist_p_z(self):
    return self.dist_p_fn()

  def sample_p_z(self, batch_size: int, rng: jnp.ndarray):
    """Sample from p(z)."""
    dist_p = self.dist_p_fn()
    if batch_size:
      return dist_p.sample(seed=rng, sample_shape=(batch_size,))
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
    env_obs = obs[..., :-self.z_size]
    z = obs[..., -self.z_size:]
    return env_obs, z

  def index_obs(self, obs: jnp.ndarray):
    if self.obs_indices:
      return obs[..., self.obs_indices]
    return obs

  def unindex_obs(self, indexed_obs: jnp.ndarray):
    if self.obs_indices:
      obs = jnp.zeros(indexed_obs.shape[:-1] + (self.env_obs_size,))
      obs = obs.at[..., self.obs_indices].add(indexed_obs)
      return obs
    return indexed_obs

  def concat_obs(self, obs: jnp.ndarray, z: jnp.ndarray):
    new_obs = jnp.concatenate([obs, z], axis=-1)
    return new_obs

  def disc_loss_fn(self, data: StepData, udata: StepData, rng: jnp.ndarray,
                   params: Dict[str, Dict[str, jnp.ndarray]]):
    return disc_loss_fn(
        data, udata, rng, params, disc=self, normalize_obs=self.normalize_obs)


class ParameterizeWrapper(Env):
  """A wrapper that parameterizes Brax Env."""

  def __init__(self,
               environment: Env,
               disc: Discriminator,
               obs_norm_reward_multiplier: float = 0.0,
               env_reward_multiplier: float = 0.0):
    self._environment = environment
    self.action_repeat = self._environment.action_repeat
    if hasattr(self._environment, 'batch_size'):
      self.batch_size = self._environment.batch_size
    else:
      self.batch_size = None
    self.sys = self._environment.sys
    self.disc = disc
    self.z_size = disc.z_size
    self.env_obs_size = self._environment.observation_size
    self.env_reward_multiplier = env_reward_multiplier
    self.obs_norm_reward_multiplier = obs_norm_reward_multiplier

  def concat(
      self,
      state: State,
      z: jnp.ndarray,
      normalizer_params: Dict[str, jnp.ndarray] = None,
      params: Dict[str, Dict[str, jnp.ndarray]] = None,
      replace_reward: bool = True,
  ) -> State:
    """Concatenate state with param and recompute reward."""
    new_obs = self.disc.concat_obs(state.obs, z)
    state = state.replace(obs=new_obs)
    if replace_reward:
      new_obs = self.disc.normalize_fn(normalizer_params, new_obs)
      env_obs, z = self.disc.split_obs(new_obs)
      new_reward = self.disc.ll_q_z_o(
          z, env_obs, params=params, add_offset=True)
      if self.obs_norm_reward_multiplier:
        new_reward += self.obs_norm_reward_multiplier * jnp.linalg.norm(
            self.disc.index_obs(env_obs), axis=-1)
      new_reward = jax.lax.stop_gradient(new_reward)
      state = state.replace(reward=new_reward +
                            self.env_reward_multiplier * state.reward)
    return state

  def reset(self, rng: jnp.ndarray, z: jnp.ndarray = None) -> State:
    """Resets the environment to an initial state."""
    state = self._environment.reset(rng)
    if z is None:
      z = self.disc.sample_p_z(self.batch_size, rng)
    else:
      assert z.shape[-1] == self.z_size, f'{z.shape}[-1] != {self.z_size}'
    return self.concat(state, z, replace_reward=False)

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    _, z = self.disc.split_obs(state.obs)
    state = self._environment.step(state, action)
    return self.concat(
        state, z, normalizer_params, extra_params, replace_reward=True)

  def step2(self, state: State, action: jnp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""
    _, z = self.disc.split_obs(state.obs)
    state = self._environment.step(state, action)
    return self.concat(state, z, replace_reward=False)


def disc_loss_fn(data: StepData,
                 udata: StepData,
                 rng: jnp.ndarray,
                 params: Dict[str, Dict[str, jnp.ndarray]],
                 disc: Discriminator,
                 normalize_obs: bool = False):
  """Discriminator loss function."""
  disc_loss = 0
  if disc and disc.model:
    d = data if normalize_obs else udata
    env_obs, z = disc.split_obs(d.obs)
    disc_loss = -jnp.mean(disc.ll_q_z_o(z, env_obs, params=params))
  return disc_loss, rng


def create(env_name: str, wrapper_params: Dict[str, Any], **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = composer.create(env_name=env_name, **kwargs)
  return ParameterizeWrapper(env, **wrapper_params)


def create_fn(env_name: str, wrapper_params: Dict[str, Any],
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, wrapper_params, **kwargs)


def create_disc_fn(algo_name: str,
                   observation_size: int,
                   obs_indices: Tuple[Any] = None,
                   scale: float = 1.0,
                   diayn_num_skills: int = 8,
                   logits_clip_range: float = 5.0,
                   spectral_norm: bool = False,
                   **kwargs):
  """Create a standard discriminator."""
  disc_fn = {
      'fixed_gcrl':
          functools.partial(
              Discriminator,
              q_fn='indexing',
              z_size=len(obs_indices),
              obs_indices=obs_indices,
              dist_p='Deterministic',
              dist_p_params=dict(value=scale),
          ),
      'gcrl':
          functools.partial(
              Discriminator,
              q_fn='indexing',
              z_size=len(obs_indices),
              obs_indices=obs_indices,
              dist_p_params=dict(scale=scale / 2.0),
          ),
      'cdiayn':
          functools.partial(
              Discriminator,
              q_fn='indexing_mlp',
              z_size=len(obs_indices),
              obs_indices=obs_indices,
              q_fn_params=dict(output_size=len(obs_indices),),
              spectral_norm=spectral_norm,
          ),
      'diayn':
          functools.partial(
              Discriminator,
              q_fn='indexing_mlp',
              z_size=diayn_num_skills,
              obs_indices=obs_indices,
              q_fn_params=dict(output_size=diayn_num_skills,),
              dist_p='UniformCategorial',
              dist_q='Categorial',
              logits_clip_range=logits_clip_range,
              spectral_norm=spectral_norm,
          ),
      'diayn_full':
          functools.partial(
              Discriminator,
              q_fn='mlp',
              z_size=diayn_num_skills,
              obs_indices=obs_indices,
              q_fn_params=dict(
                  input_size=observation_size,
                  output_size=diayn_num_skills,
              ),
              dist_p='UniformCategorial',
              dist_q='Categorial',
              logits_clip_range=logits_clip_range,
              spectral_norm=spectral_norm,
          ),
  }.get(algo_name, None)
  assert disc_fn, f'invalid algo_name: {algo_name}'
  disc_fn = functools.partial(disc_fn, obs_scale=scale, **kwargs)
  return disc_fn
