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
class ReplayBufferState:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  current_position: jnp.ndarray
  current_size: jnp.ndarray
  key: PRNGKey


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
  """Base class for limited-size FIFO reply buffers.

  Implements an `insert()` method which behaves like a limited-size queue.
  I.e. it adds samples to the end of the queue and, if necessary, removes the
  oldest samples form the queue in order to keep the maximum size within the
  specified limit.

  Derived classes must implement the `sample()` method.
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

  def init(self, key: PRNGKey) -> ReplayBufferState:
    return ReplayBufferState(
        data=jnp.zeros(self._data_shape, self._data_dtype),
        current_size=jnp.zeros((), jnp.int32),
        current_position=jnp.zeros((), jnp.int32),
        key=key)

  def insert(self, buffer_state: ReplayBufferState,
             samples: Sample) -> ReplayBufferState:
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

    return buffer_state.replace(
        data=data, current_position=position, current_size=size)

  def sample(
      self,
      buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
    raise NotImplementedError(f'{self.__class__}.sample() is not implemented.')

  def size(self, buffer_state: ReplayBufferState) -> int:
    return buffer_state.current_size


class Queue(QueueBase[Sample], Generic[Sample]):
  """Implements a limited-size queue replay buffer."""

  def sample(
      self,
      buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
    """Sample a batch of data.

    Args:
      buffer_state: Buffer state

    Returns:
      New buffer state and a batch with leading dimension 'sample_batch_size'.
    """
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'Data shape expected by the replay buffer ({self._data_shape}) does '
          f'not match the shape of the buffer state ({buffer_state.data.shape})'
      )

    # Note that this may be out of bound, but the operations below would still
    # work fine as they take this number modulo the buffer size.
    first_element_idx = (
        buffer_state.current_position - buffer_state.current_size)
    idx = jnp.arange(self._sample_batch_size) + first_element_idx

    flat_batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')

    # TODO: Raise an error instad of padding with zeros
    #                    when the buffer does not contain enough elements.
    # If the sample batch size is larger than the number of elements in the
    # queue, `mask` would contain 0s for all elements that are past the current
    # position. Otherwise, `mask` will be only ones.
    # mask.shape = (self._sample_batch_size,)
    mask = idx < buffer_state.current_position
    # mask.shape = (self._sample_batch_size, 1)
    mask = jnp.expand_dims(mask, axis=range(1, flat_batch.ndim))
    flat_batch = flat_batch * mask

    # The effective size of the sampled batch.
    sample_size = jnp.minimum(self._sample_batch_size,
                              buffer_state.current_size)
    # Remove the sampled batch from the queue.
    new_state = buffer_state.replace(current_size=buffer_state.current_size -
                                     sample_size)
    return new_state, self._unflatten_fn(flat_batch)


class UniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
  """Implements an uniform sampling limited-size replay queue.

  * It behaves as a limited size queue (if buffer is full it removes the oldest
    elements when new one is inserted).
  * It supports batch insertion only (no single element)
  * It performs uniform random sampling with replacement of a batch of size
    `sample_batch_size`
  """

  def sample(
      self,
      buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'Data shape expected by the replay buffer ({self._data_shape}) does '
          f'not match the shape of the buffer state ({buffer_state.data.shape})'
      )

    key, sample_key = jax.random.split(buffer_state.key)
    idx = jax.random.randint(
        sample_key, (self._sample_batch_size,),
        minval=buffer_state.current_position - buffer_state.current_size,
        maxval=buffer_state.current_position)
    batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
    return buffer_state.replace(key=key), self._unflatten_fn(batch)
