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

"""Utilities for adversarial IRL and state marginal matching RL.

See:
  "A Divergence Minimization Perspective on Imitation Learning"
  https://arxiv.org/abs/1911.02256
"""


import functools
from typing import Any, Callable, List, Tuple, Optional, Dict
from brax.envs.env import Env
from brax.envs.env import State
from brax.experimental.braxlines.common import dist_utils
from brax.experimental.composer import composer
from brax.experimental.composer import observers
from brax.training import networks
from brax.training import normalization
from brax.training.ppo import StepData
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions

DISC_PARAM_NAME = 'irl_disc_params'


class IRLDiscriminator(object):
  """Discriminator for target data versus on-policy data."""

  def __init__(
      self,
      env: Env,
      input_size: int,
      reward_type: str = 'gail',
      arch: Tuple[int] = (32, 32),
      obs_indices: Optional[List[Any]] = None,
      act_indices: Optional[List[int]] = None,
      include_action: bool = False,
      logits_clip_range: float = None,
      param_name: str = DISC_PARAM_NAME,
      target_data: jnp.ndarray = None,
      balance_data: bool = True,
      normalize_obs: bool = False,
  ):
    self.env_obs_size = env.observation_size
    self.arch = arch
    self.input_size = input_size
    self.dist_fn = lambda x: dist_utils.clipped_bernoulli(
        logits=x, clip_range=logits_clip_range)
    self.obs_indices = observers.index_preprocess(obs_indices, env)
    self.act_indices = act_indices
    self.include_action = include_action
    self.reward_type = reward_type
    self.param_name = param_name
    self.model = None
    self.initialized = False
    self.target_data = target_data
    self.balance_data = balance_data
    self.normalize_obs = normalize_obs
    self.normalize_fn = normalization.make_data_and_apply_fn(
        [self.env_obs_size], self.normalize_obs)[1]

  def set_target_data(self, target_data: jnp.ndarray):
    self.target_data = target_data

  def init_model(self, rng: jnp.ndarray = None):
    model = networks.make_model(self.arch + (1,), self.input_size)
    model_params = model.init(rng)
    self.fn = model.apply
    self.model = model
    self.initialized = True
    return {self.param_name: model_params}

  def dist(self,
           data: jnp.ndarray,
           params: Dict[str, Dict[str, jnp.ndarray]] = None):
    assert self.initialized, 'init_model() must be called'
    param = params[self.param_name]
    return self.dist_fn(self.fn(param, data))

  def ll(self,
         data: jnp.ndarray,
         labels: jnp.ndarray,
         params: Dict[str, Dict[str, jnp.ndarray]] = None) -> jnp.ndarray:
    dist = self.dist(data, params=params)
    ll = dist.log_prob(labels)
    ll = jnp.sum(ll, axis=-1)
    return ll

  def irl_reward(
      self,
      data: jnp.ndarray,
      params: Dict[str, Dict[str, jnp.ndarray]] = None) -> jnp.ndarray:
    """Compute IRL reward."""
    dist = self.dist(data, params=params)
    if self.reward_type == 'gail':
      r = -dist.log_prob(jnp.zeros_like(dist.logits))
    elif self.reward_type == 'gail2':
      # https://arxiv.org/abs/2106.00672
      r = dist.log_prob(jnp.ones_like(dist.logits))
    elif self.reward_type == 'airl':
      r = dist.logits
    elif self.reward_type == 'fairl':
      r = dist.logits
      r = jnp.exp(r) * -r
    else:
      raise NotImplementedError(self.reward_type)
    r = jnp.sum(r, axis=-1)
    return r

  def obs_act2data(self, obs: jnp.ndarray, act: jnp.ndarray):
    """Convert obs and actions into data."""
    assert obs.shape[:-1] == act.shape[:-1], f'obs={obs.shape}, act={act.shape}'
    data = obs
    data = self.index_obs(data)
    if self.include_action:
      if self.act_indices:
        act = act.take(self.act_indices, axis=-1)
      data = jnp.concatenate([data, act], axis=-1)
    return data

  def index_obs(self, obs: jnp.ndarray):
    if self.obs_indices:
      return obs.take(self.obs_indices, axis=-1)
    else:
      return obs

  def disc_loss_fn(self, data: StepData, udata: StepData, rng: jnp.ndarray,
                   params: Dict[str, Dict[str, jnp.ndarray]]):
    return disc_loss_fn(
        data,
        udata,
        rng,
        params,
        disc=self,
        normalize_obs=self.normalize_obs,
        target_data=self.target_data,
        balance_data=self.balance_data)


class IRLWrapper(Env):
  """A wrapper that adds an IRL reward to a Physax Env."""

  def __init__(
      self,
      environment: Env,
      disc: IRLDiscriminator,
  ):
    self._environment = environment
    self.action_repeat = self._environment.action_repeat
    self.batch_size = self._environment.batch_size
    self.sys = self._environment.sys
    self.disc = disc

  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""
    state = self._environment.reset(rng)
    return state.replace(reward=jnp.zeros_like(state.reward))

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    obs = self.disc.normalize_fn(normalizer_params, state.obs)
    new_reward = disc_reward_fn(obs, action, params=params, disc=self.disc)
    state = self._environment.step(state, action)
    return state.replace(reward=new_reward)


def disc_reward_fn(
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    params: Dict[str, Dict[str, jnp.ndarray]],
    disc: IRLDiscriminator,
) -> jnp.ndarray:
  data = disc.obs_act2data(obs, actions)
  new_reward = disc.irl_reward(data, params)
  return jax.lax.stop_gradient(new_reward)


def disc_loss_fn(data: StepData,
                 udata: StepData,
                 rng: jnp.ndarray,
                 params: Dict[str, Dict[str, jnp.ndarray]],
                 disc: IRLDiscriminator,
                 target_data: jnp.ndarray,
                 balance_data: bool = True,
                 normalize_obs: bool = False):
  """Discriminator loss function."""
  d = data if normalize_obs else udata
  target_d = target_data  # TODO: add normalize option for target_data
  d = disc.obs_act2data(d.obs[:d.actions.shape[0]], d.actions)
  if balance_data and d.shape[0] != target_d.shape[0]:
    rng, loss_key = jax.random.split(rng)
    if d.shape[0] > target_d.shape[0]:
      indices = jnp.arange(0, d.shape[0])
      indices = jax.random.shuffle(loss_key, indices)
      d = d[indices[:target_d.shape[0]]]
    else:
      indices = jnp.arange(0, target_d.shape[0])
      indices = jax.random.shuffle(loss_key, indices)
      target_d = target_d[indices[:d.shape[0]]]
  disc_loss = -jnp.mean(
      disc.ll(d, jnp.zeros(d.shape[:-1] + (1,)), params=params))
  disc_loss += -jnp.mean(
      disc.ll(target_d, jnp.ones(target_d.shape[:-1] + (1,)), params=params))
  return disc_loss, rng


def create(env_name: str, disc: IRLDiscriminator, **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = composer.create(env_name, **kwargs)
  return IRLWrapper(env, disc)


def create_fn(env_name: str, disc: IRLDiscriminator,
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, disc, **kwargs)
