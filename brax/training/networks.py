# Copyright 2025 The Brax Authors.
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

"""Network definitions."""

import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]


class MLPHead(linen.Module):
  """MLP over pre-processed latent vectors.

  For an example usage, see the Aloha sim2real code on
  https://github.com/google-deepmind/mujoco_playground.
  """

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False
  state_key: str = 'proprio'
  latent_key_prefix: str = 'latent_'  # Assume followed by integer index.
  latent_head_size: int = 64

  @linen.compact
  def __call__(self, data: Mapping[str, jax.Array]):
    latents = []
    latent_keys = sorted(
        [k for k in data.keys() if k.startswith(self.latent_key_prefix)],
        key=lambda x: int(x.split('_')[-1]),
    )
    assert len(latent_keys) > 0, 'No latent keys found'
    for key in latent_keys:
      latents.append(data[key])
    hidden = [
        self.activation(linen.Dense(self.latent_head_size)(latent))
        for latent in latents
    ]
    if self.state_key:
      hidden.append(data[self.state_key])
    hidden = jnp.concatenate(hidden, axis=-1)
    return MLP(
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        kernel_init=self.kernel_init,
        activate_final=self.activate_final,
        layer_norm=self.layer_norm,
    )(hidden)


class MLP(linen.Module):
  """MLP module."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
      )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
    return hidden


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
      )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


class CNN(linen.Module):
  """CNN module. Inputs are expected in Batch * HWC format."""

  num_filters: Sequence[int]
  kernel_sizes: Sequence[Tuple]
  strides: Sequence[Tuple]
  activation: ActivationFn = linen.relu
  use_bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, (num_filter, kernel_size, stride) in enumerate(
        zip(self.num_filters, self.kernel_sizes, self.strides)
    ):
      hidden = linen.Conv(
          num_filter,
          kernel_size=kernel_size,
          strides=stride,
          use_bias=self.use_bias,
      )(hidden)

      hidden = self.activation(hidden)
    return hidden


class VisionMLP(linen.Module):
  """Applies a CNN backbone then an MLP.

  The CNN architecture originates from the paper:
  "Human-level control through deep reinforcement learning",
  Nature 518, no. 7540 (2015): 529-533
  """

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  layer_norm: bool = False
  normalise_channels: bool = False
  state_obs_key: str = ''
  policy_head: bool = True  # = False is useful for frozen encoders.

  @linen.compact
  def __call__(self, data: dict):
    pixels_hidden = {k: v for k, v in data.items() if k.startswith('pixels/')}
    if self.normalise_channels:
      # Calculates shared statistics over an entire 2D image.
      image_layernorm = functools.partial(
          linen.LayerNorm,
          use_bias=False,
          use_scale=False,
          reduction_axes=(-1, -2),
      )

      def ln_per_chan(v: jax.Array):
        normalised = [
            image_layernorm()(v[..., chan]) for chan in range(v.shape[-1])
        ]
        return jnp.stack(normalised, axis=-1)

      pixels_hidden = jax.tree.map(ln_per_chan, pixels_hidden)

    natureCNN = functools.partial(
        CNN,
        num_filters=[32, 64, 64],
        kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        strides=[(4, 4), (2, 2), (1, 1)],
        activation=linen.relu,
        use_bias=False,
    )
    cnn_outs = [natureCNN()(pixels_hidden[key]) for key in pixels_hidden]
    cnn_outs = [jnp.mean(cnn_out, axis=(-2, -3)) for cnn_out in cnn_outs]
    if not self.policy_head:
      return jnp.concatenate(cnn_outs, axis=-1)
    if self.state_obs_key:
      cnn_outs.append(
          data[self.state_obs_key]
      )  # TODO: Try with dedicated state network

    hidden = jnp.concatenate(cnn_outs, axis=-1)
    return MLP(
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        kernel_init=self.kernel_init,
        activate_final=self.activate_final,
        layer_norm=self.layer_norm,
    )(hidden)


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]


class Param(linen.Module):
  """Scalar parameter module."""

  init_value: float = 0.0
  size: int = 1

  @linen.compact
  def __call__(self):
    return self.param(
        'value', init_fn=lambda keys: jnp.full((self.size,), self.init_value)
    )


class LogParam(linen.Module):
  """Scalar parameter module with log scale."""

  init_value: float = 1.0
  size: int = 1

  @linen.compact
  def __call__(self):
    log_value = self.param(
        'log_value',
        init_fn=lambda key: jnp.full((self.size,), jnp.log(self.init_value)),
    )
    return jnp.exp(log_value)


class PolicyModuleWithStd(linen.Module):
  """Policy module with learnable mean and standard deviation."""

  param_size: int
  hidden_layer_sizes: Sequence[int]
  activation: ActivationFn
  kernel_init: jax.nn.initializers.Initializer
  layer_norm: bool
  noise_std_type: Literal['scalar', 'log']
  init_noise_std: float
  state_dependent_std: bool = False

  @linen.compact
  def __call__(self, obs):
    if self.noise_std_type not in ['scalar', 'log']:
      raise ValueError(
          f'Unsupported noise std type: {self.noise_std_type}. Must be one of'
          ' "scalar" or "log".'
      )

    outputs = MLP(
        layer_sizes=list(self.hidden_layer_sizes),
        activation=self.activation,
        kernel_init=self.kernel_init,
        layer_norm=self.layer_norm,
        activate_final=True,
    )(obs)

    mean_params = linen.Dense(
        self.param_size, kernel_init=self.kernel_init
    )(outputs)

    if self.state_dependent_std:
      log_std_output = linen.Dense(
          self.param_size, kernel_init=self.kernel_init
      )(outputs)
      if self.noise_std_type == 'log':
        std_params = jnp.exp(log_std_output)
      else:
        std_params = log_std_output

    else:
      if self.noise_std_type == 'scalar':
        std_module = Param(
            self.init_noise_std, size=self.param_size, name='std_param'
        )
      else:
        std_module = LogParam(
            self.init_noise_std, size=self.param_size, name='std_logparam'
        )
      std_params = std_module()

    return mean_params, std_params


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
) -> FeedForwardNetwork:
  """Creates a policy network."""
  if distribution_type == 'tanh_normal':
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )
  elif distribution_type == 'normal':
    policy_module = PolicyModuleWithStd(
        param_size=param_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  def apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))

  def init(key):
    policy_module_params = policy_module.init(key, dummy_obs)
    return policy_module_params

  return FeedForwardNetwork(init=init, apply=apply)


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Creates a value network."""
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform(),
  )

  def apply(processor_params, value_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )


def make_q_network(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""

    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            layer_norm=layer_norm,
        )(hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
  )


def make_model(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    spectral_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).

  Returns:
    a model
  """
  warnings.warn(
      'make_model is deprecated, use make_{policy|q|value}_network instead.'
  )
  dummy_obs = jnp.zeros((1, obs_size))
  if spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng1, rng2: module.init(
            {'params': rng1, 'sing_vec': rng2}, dummy_obs
        ),
        apply=module.apply,
    )
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply
    )
  return model


def make_models(
    policy_params_size: int, obs_size: int
) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  warnings.warn(
      'make_models is deprecated, use make_{policy|q|value}_network instead.'
  )
  policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
  value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
  return policy_model, value_model


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState, obs_key: str
) -> running_statistics.RunningStatisticsState:
  return running_statistics.RunningStatisticsState(
      count=processor_params.count,
      mean=processor_params.mean[obs_key],
      summed_variance=processor_params.summed_variance[obs_key],
      std=processor_params.std[obs_key],
  )


def make_policy_network_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    output_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = [256, 256],
    activation: ActivationFn = linen.swish,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    state_obs_key: str = '',
    normalise_channels: bool = False,
) -> FeedForwardNetwork:
  """Creates a policy network for vision inputs."""
  module = VisionMLP(
      layer_sizes=list(hidden_layer_sizes) + [output_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
      normalise_channels=normalise_channels,
      state_obs_key=state_obs_key,
  )

  def apply(processor_params, policy_params, obs):
    if state_obs_key:
      state_obs = preprocess_observations_fn(
          obs[state_obs_key], normalizer_select(processor_params, state_obs_key)
      )
      obs = {**obs, state_obs_key: state_obs}
    return module.apply(policy_params, obs)

  dummy_obs = {
      key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
  }
  return FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply
  )


def make_value_network_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = [256, 256],
    activation: ActivationFn = linen.swish,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    state_obs_key: str = '',
    normalise_channels: bool = False,
) -> FeedForwardNetwork:
  """Creates a value network for vision inputs."""
  value_module = VisionMLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=kernel_init,
      normalise_channels=normalise_channels,
      state_obs_key=state_obs_key,
  )

  def apply(processor_params, policy_params, obs):
    if state_obs_key:
      state_obs = preprocess_observations_fn(
          obs[state_obs_key], normalizer_select(processor_params, state_obs_key)
      )
      obs = {**obs, state_obs_key: state_obs}
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = {
      key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
  }
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )


def make_policy_network_latents(
    param_size: int,
    observation_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Creates a policy network. Wraps MLPLatents.

  Has same API as make_policy_network. Used when the env observation has image
  latents rather than pixels.

  Returns:
    A FeedForwardNetwork that takes observations and returns policy
    parameters.
  """
  module = MLPHead(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
      state_key=obs_key,
  )

  def apply(processor_params, policy_params, obs):
    if obs_key:
      state_obs = preprocess_observations_fn(
          obs[obs_key], normalizer_select(processor_params, obs_key)
      )
      obs = {**obs, obs_key: state_obs}
    return module.apply(policy_params, obs)

  if not isinstance(observation_size, Mapping):
    raise NotImplementedError(
        'make_policy_network_latents only implemented for dictionary'
        ' observations'
    )

  dummy_obs = {
      key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
  }
  return FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply
  )
