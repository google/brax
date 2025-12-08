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

"""Utility functions to compute running statistics.

This file was taken from acme and modified to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/jax/running_statistics.py
"""

import enum
from typing import Optional, Tuple, Union

from brax.training import types as training_types
from brax.training.acme import types
from flax import struct
import jax
import jax.numpy as jnp


class NormalizationMode(enum.IntEnum):
  WELFORD = 0
  EMA = 1


def _mode_from_string(mode: str) -> int:
  if mode == 'ema':
    return NormalizationMode.EMA
  elif mode == 'welford':
    return NormalizationMode.WELFORD
  else:
    raise ValueError(f'Unknown normalization mode: {mode}')


def _zeros_like(nest: types.Nest, dtype=None) -> types.Nest:
  return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(nest: types.Nest, dtype=None) -> types.Nest:
  return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


@struct.dataclass
class NestedMeanStd:
  """A container for running statistics (mean, std) of possibly nested data."""
  mean: types.Nest
  std: types.Nest


@struct.dataclass
class RunningStatisticsState(NestedMeanStd):
  """Full state of running statistics computation."""
  count: Union[jnp.ndarray, training_types.UInt64]
  summed_variance: types.Nest
  std_eps: float = 0.0
  mode: int = struct.field(pytree_node=False, default=NormalizationMode.WELFORD)


def init_state(
    nest: types.Nest,
    std_eps: float = 0.0,
    mode: str = 'welford',
) -> RunningStatisticsState:
  """Initializes the running statistics for the given nested structure.

  Args:
    nest: Nested structure to initialize statistics for.
    std_eps: Epsilon for numerical stability when getting std.
    mode: Normalization mode - 'welford' (default) or 'ema'.
  """
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  mode_int = _mode_from_string(mode)

  return RunningStatisticsState(
      count=training_types.UInt64(hi=0, lo=0),
      mean=_zeros_like(nest, dtype=dtype),
      summed_variance=_zeros_like(nest, dtype=dtype),
      std=_ones_like(nest, dtype=dtype),
      std_eps=std_eps,
      mode=mode_int,
  )


def _validate_batch_shapes(batch: types.NestedArray,
                           reference_sample: types.NestedArray,
                           batch_dims: Tuple[int, ...]) -> None:
  """Verifies shapes of the batch leaves against the reference sample.

  Checks that batch dimensions are the same in all leaves in the batch.
  Checks that non-batch dimensions for all leaves in the batch are the same
  as in the reference sample.

  Arguments:
    batch: the nested batch of data to be verified.
    reference_sample: the nested array to check non-batch dimensions.
    batch_dims: a Tuple of indices of batch dimensions in the batch shape.

  Returns:
    None.
  """
  def validate_node_shape(reference_sample: jnp.ndarray,
                          batch: jnp.ndarray) -> None:
    expected_shape = batch_dims + reference_sample.shape
    assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

  jax.tree_util.tree_map(validate_node_shape, reference_sample, batch)


def update(state: RunningStatisticsState,
           batch: types.Nest,
           *,
           weights: Optional[jnp.ndarray] = None,
           std_min_value: float = 1e-6,
           std_max_value: float = 1e6,
           pmap_axis_name: Optional[str] = None,
           validate_shapes: bool = True) -> RunningStatisticsState:
  """Updates the running statistics with the given batch of data.

  Note: data batch and state elements (mean, etc.) must have the same structure.

  Note: by default uses UInt64 for counts that get converted to float32 for division.
  This conversion has a small precision loss for large counts. float32 is used
  to accumulate variance, so can also suffer from precision loss due to the 24 bit
  mantissa for float32.
  To improve precision, consider setting jax_enable_x64 to True, see
  https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

  Arguments:
    state: The running statistics before the update.
    batch: The data to be used to update the running statistics.
    weights: Weights of the batch data. Should match the batch dimensions.
      Passing a weight of 2. should be equivalent to updating on the
      corresponding data point twice.
    std_min_value: Minimum value for the standard deviation.
    std_max_value: Maximum value for the standard deviation.
    pmap_axis_name: Name of the pmapped axis, if any.
    validate_shapes: If true, the shapes of all leaves of the batch will be
      validated. Enabled by default. Doesn't impact performance when jitted.

  Returns:
    Updated running statistics.
  """
  # We require exactly the same structure to avoid issues when flattened
  # batch and state have different order of elements.
  assert jax.tree_util.tree_structure(batch) == jax.tree_util.tree_structure(state.mean)
  batch_leaves = jax.tree_util.tree_leaves(batch)
  if not batch_leaves:  # State and batch are both empty. Nothing to normalize.
    return state
  batch_shape = batch_leaves[0].shape
  # We assume the batch dimensions always go first.
  batch_dims = batch_shape[:len(batch_shape) -
                           jax.tree_util.tree_leaves(state.mean)[0].ndim]
  batch_axis = range(len(batch_dims))
  if weights is None:
    step_increment = jnp.prod(jnp.array(batch_dims)).astype(jnp.int32)
  else:
    step_increment = jnp.sum(weights).astype(jnp.int32)
  if pmap_axis_name is not None:
    step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
  count = state.count + step_increment

  if isinstance(count, training_types.UInt64):
    # Convert UInt64 count to float32 for division operations.
    # Note: small precision loss due to float32's 24-bit mantissa.
    count_float = (jnp.float32(count.hi) * jnp.float32(2.0**32) +
                   jnp.float32(count.lo))
  else:
    count_float = jnp.float32(count)

  # Validation is important. If the shapes don't match exactly, but are
  # compatible, arrays will be silently broadcasted resulting in incorrect
  # statistics.
  if validate_shapes:
    if weights is not None:
      if weights.shape != batch_dims:
        raise ValueError(f'{weights.shape} != {batch_dims}')
    _validate_batch_shapes(batch, state.mean, batch_dims)

  if state.mode == NormalizationMode.EMA:
    # RSL-RL's EmpiricalNormalization algorithm uses
    # rate = batch_size / total_count instead of fixed alpha.
    rate = jnp.float32(step_increment) / count_float

    def _compute_ema_statistics(
        mean: jnp.ndarray,
        variance: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      batch_mean = jnp.mean(batch, axis=batch_axis)
      batch_var = jnp.var(batch, axis=batch_axis)
      if pmap_axis_name is not None:
        batch_mean = jax.lax.pmean(batch_mean, axis_name=pmap_axis_name)
        batch_var = jax.lax.pmean(batch_var, axis_name=pmap_axis_name)
      delta_mean = batch_mean - mean
      new_mean = mean + rate * delta_mean
      new_variance = variance + rate * (
          batch_var - variance + delta_mean * (batch_mean - new_mean)
      )
      return new_mean, new_variance

    updated_stats = jax.tree_util.tree_map(
        _compute_ema_statistics,
        state.mean,
        state.summed_variance,
        batch,
    )
    mean = jax.tree_util.tree_map(lambda _, x: x[0], state.mean, updated_stats)
    variance = jax.tree_util.tree_map(
        lambda _, x: x[1], state.mean, updated_stats
    )

    def compute_ema_std(
        variance: jnp.ndarray, _std: jnp.ndarray
    ) -> jnp.ndarray:
      variance = jnp.maximum(variance, 0)
      std = jnp.sqrt(variance + state.std_eps)
      std = jnp.clip(std, std_min_value, std_max_value)
      return std

    std = jax.tree_util.tree_map(compute_ema_std, variance, state.std)

    return RunningStatisticsState(
        count=count,
        mean=mean,
        summed_variance=variance,
        std=std,
        std_eps=state.std_eps,
        mode=state.mode,
    )

  def _compute_node_statistics(
      mean: jnp.ndarray, summed_variance: jnp.ndarray,
      batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert isinstance(mean, jnp.ndarray), type(mean)
    assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
    # The mean and the sum of past variances are updated with Welford's
    # algorithm using batches (see https://stackoverflow.com/q/56402955).
    diff_to_old_mean = batch - mean
    if weights is not None:
      expanded_weights = jnp.reshape(
          weights,
          list(weights.shape) + [1] * (batch.ndim - weights.ndim))
      diff_to_old_mean = diff_to_old_mean * expanded_weights
    mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count_float
    if pmap_axis_name is not None:
      mean_update = jax.lax.psum(
          mean_update, axis_name=pmap_axis_name)
    mean = mean + mean_update

    diff_to_new_mean = batch - mean
    variance_update = diff_to_old_mean * diff_to_new_mean
    variance_update = jnp.sum(variance_update, axis=batch_axis)
    if pmap_axis_name is not None:
      variance_update = jax.lax.psum(variance_update, axis_name=pmap_axis_name)
    summed_variance = summed_variance + variance_update
    return mean, summed_variance

  updated_stats = jax.tree_util.tree_map(_compute_node_statistics, state.mean,
                                         state.summed_variance, batch)
  # Extract `mean` and `summed_variance` from `updated_stats` nest.
  mean = jax.tree_util.tree_map(lambda _, x: x[0], state.mean, updated_stats)
  summed_variance = jax.tree_util.tree_map(lambda _, x: x[1], state.mean,
                                           updated_stats)

  def compute_std(summed_variance: jnp.ndarray,
                  std: jnp.ndarray) -> jnp.ndarray:
    assert isinstance(summed_variance, jnp.ndarray)
    # Summed variance can get negative due to rounding errors.
    summed_variance = jnp.maximum(summed_variance, 0)
    std = jnp.sqrt(summed_variance / count_float + state.std_eps)
    std = jnp.clip(std, std_min_value, std_max_value)
    return std

  std = jax.tree_util.tree_map(compute_std, summed_variance, state.std)

  return RunningStatisticsState(
      count=count,
      mean=mean,
      summed_variance=summed_variance,
      std=std,
      std_eps=state.std_eps,
      mode=state.mode,
  )


def normalize(batch: types.NestedArray,
              mean_std: NestedMeanStd,
              max_abs_value: Optional[float] = None) -> types.NestedArray:
  """Normalizes data using running statistics."""

  def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                     std: jnp.ndarray) -> jnp.ndarray:
    # Only normalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    data = (data - mean) / std
    if max_abs_value is not None:
      # TODO(b/124318564): remove pylint directive
      data = jnp.clip(data, -max_abs_value, +max_abs_value)
    return data

  return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: types.NestedArray,
                mean_std: NestedMeanStd) -> types.NestedArray:
  """Denormalizes values in a nested structure using the given mean/std.

  Only values of inexact types are denormalized.
  See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
  hierarchy.

  Args:
    batch: a nested structure containing batch of data.
    mean_std: mean and standard deviation used for denormalization.

  Returns:
    Nested structure with denormalized values.
  """

  def denormalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                       std: jnp.ndarray) -> jnp.ndarray:
    # Only denormalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    return data * std + mean

  return jax.tree_util.tree_map(denormalize_leaf, batch, mean_std.mean, mean_std.std)
