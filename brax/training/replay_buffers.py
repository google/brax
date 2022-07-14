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

"""Replay buffers for Brax."""

import abc
from typing import Generic, Tuple, TypeVar

from brax.training.types import PRNGKey
import flax
import jax
from jax import flatten_util
import jax.numpy as jnp

State = TypeVar('State')
Sample = TypeVar('Sample')


class ReplayBuffer(abc.ABC, Generic[State, Sample]):
  """Contains replay buffer methods."""

  @abc.abstractmethod
  def init(self, key: PRNGKey) -> State:
    """Init the replay buffer."""

  @abc.abstractmethod
  def insert(self, buffer_state: State, samples: Sample) -> State:
    """Insert data in the replay buffer."""

  @abc.abstractmethod
  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""

  @abc.abstractmethod
  def size(self, buffer_state: State) -> int:
    """Total amount of elements that are sampleable."""


@flax.struct.dataclass
class _ReplayBufferState:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  current_position: jnp.ndarray
  current_size: jnp.ndarray
  key: PRNGKey


class UniformSamplingQueue(ReplayBuffer, Generic[Sample]):
  """Replay buffer with uniform sampling.

  * It behaves as a limited size queue (if buffer is full it removes the oldest
    elements when new one is inserted).
  * It supports batch insertion only (no single element)
  * It performs uniform random sampling with replacement of a batch of size
    `sample_batch_size`
  """

  def __init__(self, max_replay_size: int, dummy_data_sample: Sample,
               sample_batch_size: int):
    self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])

    dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(
        dummy_data_sample)
    self._unflatten_fn = jax.vmap(self._unflatten_fn)
    data_size = len(dummy_flatten)

    self._data_shape = (max_replay_size, data_size)
    self._data_dtype = dummy_flatten.dtype
    self._sample_batch_size = sample_batch_size

  def init(self, key: PRNGKey) -> _ReplayBufferState:
    return _ReplayBufferState(
        data=jnp.zeros(self._data_shape, self._data_dtype),
        current_size=jnp.zeros((), jnp.int32),
        current_position=jnp.zeros((), jnp.int32),
        key=key)

  def insert(self, buffer_state: _ReplayBufferState,
             samples: Sample) -> _ReplayBufferState:
    """Insert data in the replay buffer.

    Args:
      buffer_state: Buffer state
      samples: Sample to insert with a leading batch size.

    Returns:
      New buffer state.
    """
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'buffer_state.data.shape ({buffer_state.data.shape}) '
          f'doesn\'t match the expected value ({self._data_shape})')

    update = self._flatten_fn(samples)
    data = buffer_state.data

    # Make sure update is not larger than the maximum replay size.
    if len(update) > len(data):
      raise ValueError(
          'Trying to insert a batch of samples larger than the maximum replay '
          f'size. num_samples: {len(update)}, max replay size {len(data)}')

    # If needed, roll the buffer to make sure there's enough space to fit
    # `update` after the current position.
    position = buffer_state.current_position
    roll = jnp.minimum(0, len(data) - position - len(update))
    data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0),
                        lambda: data)
    position = position + roll

    # Update the buffer and the control numbers.
    data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
    position = (position + len(update)) % len(data)
    size = jnp.minimum(buffer_state.current_size + len(update), len(data))

    return _ReplayBufferState(
        data=data,
        current_position=position,
        current_size=size,
        key=buffer_state.key)

  def sample(
      self, buffer_state: _ReplayBufferState
  ) -> Tuple[_ReplayBufferState, Sample]:
    """Sample a batch of data.

    Args:
      buffer_state: Buffer state

    Returns:
      New buffer state and a batch with leading dimension 'sample_batch_size'.
    """
    assert buffer_state.data.shape == self._data_shape
    key, sample_key = jax.random.split(buffer_state.key)
    idx = jax.random.randint(
        sample_key, (self._sample_batch_size,),
        minval=0,
        maxval=buffer_state.current_size)
    batch = jnp.take(buffer_state.data, idx, axis=0, mode='clip')
    return buffer_state.replace(key=key), self._unflatten_fn(batch)

  def size(self, buffer_state: _ReplayBufferState) -> int:
    return buffer_state.current_size
