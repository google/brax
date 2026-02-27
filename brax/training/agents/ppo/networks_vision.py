# Copyright 2026 The Brax Authors.
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

"""PPO vision networks."""

from typing import Any, Literal, Mapping, Sequence, Tuple, Union

from brax.training import distribution
from brax.training import networks
from brax.training import types
import flax
from flax import linen
import jax


_PADDING_MAP = {'zeros': 'SAME', 'valid': 'VALID'}


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.swish,
    normalise_channels: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    mean_clip_scale: float | None = None,
    mean_kernel_init_fn: networks.Initializer | None = None,
    mean_kernel_init_kwargs: Mapping[str, Any] | None = None,
    # CNN backbone configuration.
    cnn_output_channels: Sequence[int] = (32, 64, 64),
    cnn_kernel_size: Sequence[int] = (8, 4, 3),
    cnn_stride: Sequence[int] = (4, 2, 1),
    cnn_padding: str = 'zeros',
    cnn_activation: networks.ActivationFn = linen.relu,
    cnn_max_pool: bool = False,
    cnn_global_pool: str = 'avg',
    cnn_spatial_softmax: bool = False,
    cnn_spatial_softmax_temperature: float = 1.0,
) -> PPONetworks:
  """Make Vision PPO networks with preprocessor.

  Args:
    observation_size: mapping from observation key to shape.
    action_size: number of action dimensions.
    preprocess_observations_fn: observation preprocessor (e.g. normalizer).
    policy_hidden_layer_sizes: MLP layer sizes after the CNN for the policy.
    value_hidden_layer_sizes: MLP layer sizes after the CNN for the value fn.
    activation: MLP activation function.
    normalise_channels: if True, apply per-channel layer norm to pixel inputs.
    policy_obs_key: key for the proprioceptive state observation used by policy.
    value_obs_key: key for the proprioceptive state observation used by value.
    distribution_type: 'normal' or 'tanh_normal' action distribution.
    noise_std_type: 'scalar' or 'log' parameterisation for action noise std.
    init_noise_std: initial value for the noise std parameter.
    state_dependent_std: if True, std is a function of the state.
    policy_network_kernel_init_fn: kernel initializer factory for policy MLP.
    policy_network_kernel_init_kwargs: kwargs for policy kernel init factory.
    value_network_kernel_init_fn: kernel initializer factory for value MLP.
    value_network_kernel_init_kwargs: kwargs for value kernel init factory.
    mean_clip_scale: if set, clip mean output with soft saturation.
    mean_kernel_init_fn: kernel initializer factory for the mean head.
    mean_kernel_init_kwargs: kwargs for mean kernel init factory.
    cnn_output_channels: number of filters per conv layer.
    cnn_kernel_size: square kernel size per conv layer.
    cnn_stride: square stride per conv layer.
    cnn_padding: padding mode — 'zeros' (SAME) or 'valid' (VALID).
    cnn_activation: activation function or name (e.g. 'elu', 'relu').
    cnn_max_pool: whether to apply 2x2 max-pool after each conv layer.
    cnn_global_pool: pooling over spatial dims — 'avg', 'max', or 'none'.
    cnn_spatial_softmax: use spatial softmax instead of global pooling.
    cnn_spatial_softmax_temperature: temperature for spatial softmax.
  """
  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}
  mean_kernel_init_kwargs_ = mean_kernel_init_kwargs or {}

  # Resolve string-based CNN config values.
  resolved_padding = _PADDING_MAP.get(
      str(cnn_padding).lower(), cnn_padding
  )
  resolved_cnn_activation: networks.ActivationFn = (
      networks.ACTIVATION[cnn_activation]
      if isinstance(cnn_activation, str)
      else cnn_activation
  )

  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  policy_network = networks.make_policy_network_vision(
      observation_size=observation_size,
      output_size=parametric_action_distribution.param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
      hidden_layer_sizes=policy_hidden_layer_sizes,
      state_obs_key=policy_obs_key,
      normalise_channels=normalise_channels,
      distribution_type=distribution_type,
      noise_std_type=noise_std_type,
      init_noise_std=init_noise_std,
      state_dependent_std=state_dependent_std,
      mean_clip_scale=mean_clip_scale,
      mean_kernel_init=(
          mean_kernel_init_fn(**mean_kernel_init_kwargs_)
          if mean_kernel_init_fn is not None else None
      ),
      cnn_output_channels=tuple(cnn_output_channels),
      cnn_kernel_size=tuple(cnn_kernel_size),
      cnn_stride=tuple(cnn_stride),
      cnn_padding=resolved_padding,
      cnn_activation=resolved_cnn_activation,
      cnn_max_pool=cnn_max_pool,
      cnn_global_pool=cnn_global_pool,
      cnn_spatial_softmax=cnn_spatial_softmax,
      cnn_spatial_softmax_temperature=cnn_spatial_softmax_temperature,
  )

  value_network = networks.make_value_network_vision(
      observation_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      kernel_init=value_network_kernel_init_fn(**value_kernel_init_kwargs),
      hidden_layer_sizes=value_hidden_layer_sizes,
      state_obs_key=value_obs_key,
      normalise_channels=normalise_channels,
      cnn_output_channels=tuple(cnn_output_channels),
      cnn_kernel_size=tuple(cnn_kernel_size),
      cnn_stride=tuple(cnn_stride),
      cnn_padding=resolved_padding,
      cnn_activation=resolved_cnn_activation,
      cnn_max_pool=cnn_max_pool,
      cnn_global_pool=cnn_global_pool,
      cnn_spatial_softmax=cnn_spatial_softmax,
      cnn_spatial_softmax_temperature=cnn_spatial_softmax_temperature,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
