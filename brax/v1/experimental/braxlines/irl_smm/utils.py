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

"""Utilities for adversarial IRL and state marginal matching RL.

See:
  "A Divergence Minimization Perspective on Imitation Learning"
  https://arxiv.org/abs/1911.02256
"""

import functools
import itertools
from typing import Any, Callable, List, Tuple, Optional, Dict
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

DISC_PARAM_NAME = "irl_disc_params"


class IRLDiscriminator(object):
  """Discriminator for target data versus on-policy data."""

  def __init__(
      self,
      env: Env,
      input_size: int,
      reward_type: str = "gail",
      arch: Tuple[int] = (32, 32),
      obs_indices: Optional[List[Any]] = None,
      act_indices: Optional[List[int]] = None,
      obs_scale: Optional[List[float]] = None,
      include_action: bool = False,
      logits_clip_range: float = 10.0,
      nonnegative_reward: bool = True,
      param_name: str = DISC_PARAM_NAME,
      target_data: jnp.ndarray = None,
      target_dist_fn=None,
      balance_data: bool = True,
      normalize_obs: bool = False,
      spectral_norm: bool = False,
      gradient_penalty_weight: float = 0.,
  ):
    assert obs_scale is not None
    self.env_obs_size = env.observation_size
    self.arch = arch
    self.input_size = input_size
    self.dist_fn = lambda x: dist_utils.clipped_bernoulli(
        logits=x, clip_range=logits_clip_range)
    self.obs_indices, self.obs_labels = observers.index_preprocess(
        obs_indices, env)
    self.indexed_obs_size = len(self.obs_indices)
    self.obs_scale = jnp.array(obs_scale or 1.) * jnp.ones(
        self.indexed_obs_size)
    if self.indexed_obs_size == 1:
      self.obs_labels_2d = (self.obs_labels[0], "None")
      self.obs_scale_2d = jnp.concatenate([self.obs_scale, self.obs_scale])
    else:
      self.obs_labels_2d = self.obs_labels[:2]
      self.obs_scale_2d = self.obs_scale[:2]
    self.act_indices = act_indices
    self.include_action = include_action
    self.reward_type = reward_type
    self.param_name = param_name
    self.model = None
    self.initialized = False
    self.target_data = target_data
    self.target_dist_fn = target_dist_fn
    if self.reward_type == "mle":
      assert self.target_dist_fn
    self.logits_clip_range = logits_clip_range
    self.nonnegative_reward = nonnegative_reward
    if self.nonnegative_reward:
      if self.reward_type == "mle":
        # approximately find the global min
        edges_mesh = itertools.product(*([(-1, 0, 1)] * self.indexed_obs_size))
        edges_mesh = jnp.stack([jnp.array(x) for x in edges_mesh])
        x_edges = edges_mesh * self.obs_scale[None]
        x_target_dist = target_dist_fn()
        x_samples = x_target_dist.sample(
            seed=jax.random.PRNGKey(0), sample_shape=(1000,))
        x = jnp.concatenate([x_edges, x_samples], axis=0)
        r = target_dist_fn().log_prob(x)
        self.target_dist_log_offset = -r.min()
      else:
        assert self.logits_clip_range
    if gradient_penalty_weight:
      balance_data = True
    self.balance_data = balance_data
    self.normalize_obs = normalize_obs
    self.normalize_fn = normalization.make_data_and_apply_fn(
        [self.env_obs_size], self.normalize_obs)[1]
    self.spectral_norm = spectral_norm
    self.gradient_penalty_weight = gradient_penalty_weight

  def set_target_data(self, target_data: jnp.ndarray):
    self.target_data = target_data

  def init_model(self, rng: jnp.ndarray = None):
    """Initialize neural network modules."""
    model = networks.make_model(
        self.arch + (1,), self.input_size, spectral_norm=self.spectral_norm)
    self.model = model
    if self.spectral_norm:
      rng1, rng2, rng3 = jax.random.split(rng, 3)
      model_params = model.init(rng1, rng2)
      self.fn = lambda params, x: model.apply(
          params, x, rngs={"sing_vec": rng3}, mutable=["sing_vec"])[0]
    else:
      model_params = model.init(rng)
      self.fn = model.apply

    def mean_fn(params, x):
      return jnp.mean(self.fn(params, x))

    self.dmean_fn_dx = jax.grad(mean_fn, argnums=1)

    self.model = model
    self.initialized = True
    return {self.param_name: model_params}

  def dist(self,
           data: jnp.ndarray,
           params: Dict[str, Dict[str, jnp.ndarray]] = None):
    assert self.initialized, "init_model() must be called"
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
    if self.reward_type == "gail":
      r = -dist.log_prob(jnp.zeros_like(dist.logits))
      r = jnp.sum(r, axis=-1)
    elif self.reward_type == "gail2":
      # https://arxiv.org/abs/2106.00672
      r = dist.log_prob(jnp.ones_like(dist.logits))
      if self.nonnegative_reward:
        r += self.logits_clip_range
      r = jnp.sum(r, axis=-1)
    elif self.reward_type == "airl":
      r = dist.logits
      if self.nonnegative_reward:
        r += self.logits_clip_range
      r = jnp.sum(r, axis=-1)
    elif self.reward_type == "fairl":
      r = dist.logits
      r = jnp.exp(r) * -r
      if self.nonnegative_reward:
        r += self.logits_clip_range
      r = jnp.sum(r, axis=-1)
    elif self.reward_type == "mle":  # for debugging
      assert not self.normalize_obs
      target_dist = self.target_dist_fn()
      r = target_dist.log_prob(data)
      if self.nonnegative_reward:
        r += self.target_dist_log_offset
    else:
      raise NotImplementedError(self.reward_type)
    return r

  def obs_act2data(self, obs: jnp.ndarray, act: jnp.ndarray):
    """Convert obs and actions into data."""
    assert obs.shape[:-1] == act.shape[:-1], f"obs={obs.shape}, act={act.shape}"
    data = obs
    data = self.index_obs(data)
    if self.include_action:
      if self.act_indices:
        act = act[..., self.act_indices]
      data = jnp.concatenate([data, act], axis=-1)
    return data

  def index_obs(self, obs: jnp.ndarray):
    if self.obs_indices:
      return obs[..., self.obs_indices]
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
        balance_data=self.balance_data,
        gradient_penalty_weight=self.gradient_penalty_weight,
    )


class IRLWrapper(Env):
  """A wrapper that adds an IRL reward to a Brax Env."""

  def __init__(
      self,
      environment: Env,
      disc: IRLDiscriminator,
      env_reward_multiplier: float = 0.0,
  ):
    self._environment = environment
    self.action_repeat = self._environment.action_repeat
    if hasattr(self._environment, "batch_size"):
      self.batch_size = self._environment.batch_size
    else:
      self.batch_size = None
    self.sys = self._environment.sys
    self.disc = disc
    self.env_reward_multiplier = env_reward_multiplier

  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""
    state = self._environment.reset(rng)
    return state.replace(reward=jnp.zeros_like(state.reward))

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    obs = self.disc.normalize_fn(normalizer_params, state.obs)
    new_reward = disc_reward_fn(
        obs, action, params=extra_params, disc=self.disc)
    state = self._environment.step(state, action)
    return state.replace(reward=new_reward +
                         self.env_reward_multiplier * state.reward)


def disc_reward_fn(
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    params: Dict[str, Dict[str, jnp.ndarray]],
    disc: IRLDiscriminator,
) -> jnp.ndarray:
  data = disc.obs_act2data(obs, actions)
  new_reward = disc.irl_reward(data, params)
  return jax.lax.stop_gradient(new_reward)


def disc_loss_fn(
    data: StepData,
    udata: StepData,
    rng: jnp.ndarray,
    params: Dict[str, Dict[str, jnp.ndarray]],
    disc: IRLDiscriminator,
    target_data: jnp.ndarray,
    balance_data: bool = True,
    normalize_obs: bool = False,
    gradient_penalty_weight: float = 0.,
):
  """Discriminator loss function."""
  d = data if normalize_obs else udata
  target_d = target_data  # TODO: add normalize option for target_data
  d = disc.obs_act2data(d.obs[:d.actions.shape[0]], d.actions)
  d = jnp.reshape(d, [-1, d.shape[-1]])
  target_d = jnp.reshape(target_d, [-1, target_d.shape[-1]])
  if balance_data and d.shape[0] != target_d.shape[0]:
    rng, loss_key = jax.random.split(rng)
    if d.shape[0] > target_d.shape[0]:
      indices = jnp.arange(0, d.shape[0])
      indices = jax.random.permutation(loss_key, indices)
      d = d[indices[:target_d.shape[0]]]
    else:
      indices = jax.random.permutation(loss_key, target_d.shape[0])
      target_d = target_d[indices[:d.shape[0]]]
  disc_loss = -jnp.mean(
      disc.ll(d, jnp.zeros(d.shape[:-1] + (1,)), params=params))
  disc_loss += -jnp.mean(
      disc.ll(target_d, jnp.ones(target_d.shape[:-1] + (1,)), params=params))

  if gradient_penalty_weight > 0.:
    assert (
        d.shape == target_d.shape
    ), f"d shape {d.shape} does not match target_d shape {target_d.shape}!"
    rng, sub_key = jax.random.split(rng)
    w_shape = [d.shape[0], 1]
    w = jax.random.uniform(sub_key, shape=w_shape, minval=0., maxval=1.)
    interp_d = w * d + (1. - w) * target_d

    p = params[disc.param_name]
    grad = disc.dmean_fn_dx(p, interp_d)
    if len(grad.shape) > 2:
      grad = jnp.reshape(grad, [grad.shape[0], -1])
    grad_norm = jnp.sum(jnp.linalg.norm(grad, axis=-1))
    disc_loss = disc_loss + gradient_penalty_weight * grad_norm

  return disc_loss, rng


def create(env_name: str, wrapper_params: Dict[str, Any], **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = composer.create(env_name, **kwargs)
  return IRLWrapper(env, **wrapper_params)


def create_fn(env_name: str, wrapper_params: Dict[str, Any],
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(
      create, env_name=env_name, wrapper_params=wrapper_params, **kwargs)


def make_2d(data):
  """Make data to be 2D in last dimension."""
  if data.shape[-1] == 1:
    return jnp.concatenate([data, jnp.zeros_like(data)], axis=-1)
  else:
    return data[..., :2]


def get_multimode_dist(num_modes: int = 1,
                       scale: float = 1.0,
                       indexed_obs_dim: int = 1):
  """Get a multimodal distribution of Gaussians."""
  if indexed_obs_dim == 1:
    return get_multimode_1d_dist(num_modes=num_modes, scale=scale)
  elif indexed_obs_dim == 2:
    return get_multimode_2d_dist(num_modes=num_modes, scale=scale)
  else:
    raise NotImplementedError(indexed_obs_dim)


def get_multimode_1d_dist(num_modes: int = 1, scale: float = 1.0):
  """Get a multimodal distribution of Gaussians."""
  scale = jnp.ones(1) * scale
  loc = jnp.linspace(-scale, scale, num_modes + 2)
  loc = loc[1:-1]
  scale = jnp.ones((num_modes, 1)) * scale[None] / (num_modes + 1) / 5.
  return tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=jnp.ones((num_modes,)) / num_modes),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=scale))


def get_multimode_2d_dist(num_modes: int = 1, scale: float = 1.0):
  """Get a multimodal distribution of Gaussians."""
  angles = jnp.linspace(0, jnp.pi * 2, num_modes + 1)
  angles = angles[:-1]
  x, y = jnp.cos(angles) * scale / 2., jnp.sin(angles) * scale / 2.
  loc = jnp.array([x, y]).T
  scale = jnp.ones((num_modes, 2)) * scale / 10.
  return tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=jnp.ones((num_modes,)) / num_modes),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=scale))
