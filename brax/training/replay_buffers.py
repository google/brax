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

"""Replay buffers for Brax."""

import abc
import math
from typing import Generic, Optional, Sequence, Tuple, TypeVar

from brax.training.types import PRNGKey
import flax
import jax
from jax import flatten_util
from jax.experimental import pjit
import jax.numpy as jnp

State = TypeVar('State')
Sample = TypeVar('Sample')


class ReplayBuffer(abc.ABC, Generic[State, Sample]):
  """Contains replay buffer methods."""

  @abc.abstractmethod
  def init(self, key: PRNGKey) -> State:
    """Init the replay buffer."""

  def insert(self, buffer_state: State, samples: Sample) -> State:
    """Insert data into the replay buffer."""
    self.check_can_insert(buffer_state, samples, 1)
    return self.insert_internal(buffer_state, samples)

  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""
    self.check_can_sample(buffer_state, 1)
    return self.sample_internal(buffer_state)

  def check_can_insert(self, buffer_state: State, samples: Sample, shards: int):
    """Checks whether insert can be performed. Do not JIT this method."""
    pass

  def check_can_sample(self, buffer_state: State, shards: int):
    """Checks whether sampling can be performed. Do not JIT this method."""
    pass

  @abc.abstractmethod
  def size(self, buffer_state: State) -> int:
    """Total amount of elements that are sampleable."""

  @abc.abstractmethod
  def insert_internal(self, buffer_state: State, samples: Sample) -> State:
    """Insert data into the replay buffer."""

  @abc.abstractmethod
  def sample_internal(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""


@flax.struct.dataclass
class ReplayBufferState:
  """Contains data related to a replay buffer."""

  data: jnp.ndarray
  insert_position: jnp.ndarray
  sample_position: jnp.ndarray
  key: PRNGKey


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
  """Base class for limited-size FIFO reply buffers.

  Implements an `insert()` method which behaves like a limited-size queue.
  I.e. it adds samples to the end of the queue and, if necessary, removes the
  oldest samples form the queue in order to keep the maximum size within the
  specified limit.

  Derived classes must implement the `sample()` method.
  """

  def __init__(
      self,
      max_replay_size: int,
      dummy_data_sample: Sample,
      sample_batch_size: int,
  ):
    self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])

    dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(
        dummy_data_sample
    )
    self._unflatten_fn = jax.vmap(self._unflatten_fn)
    data_size = len(dummy_flatten)

    self._data_shape = (max_replay_size, data_size)
    self._data_dtype = dummy_flatten.dtype
    self._sample_batch_size = sample_batch_size
    self._size = 0

  def init(self, key: PRNGKey) -> ReplayBufferState:
    return ReplayBufferState(
        data=jnp.zeros(self._data_shape, self._data_dtype),
        sample_position=jnp.zeros((), jnp.int32),
        insert_position=jnp.zeros((), jnp.int32),
        key=key,
    )

  def check_can_insert(self, buffer_state, samples, shards):
    """Checks whether insert operation can be performed."""
    assert isinstance(shards, int), 'This method should not be JITed.'
    insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
    if self._data_shape[0] < insert_size:
      raise ValueError(
          'Trying to insert a batch of samples larger than the maximum replay'
          f' size. num_samples: {insert_size}, max replay size'
          f' {self._data_shape[0]}'
      )
    self._size = min(self._data_shape[0], self._size + insert_size)

  def insert_internal(
      self, buffer_state: ReplayBufferState, samples: Sample
  ) -> ReplayBufferState:
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
          f"doesn't match the expected value ({self._data_shape})"
      )

    update = self._flatten_fn(samples)
    data = buffer_state.data

    # If needed, roll the buffer to make sure there's enough space to fit
    # `update` after the current position.
    position = buffer_state.insert_position
    roll = jnp.minimum(0, len(data) - position - len(update))
    data = jax.lax.cond(
        roll, lambda: jnp.roll(data, roll, axis=0), lambda: data
    )
    position = position + roll

    # Update the buffer and the control numbers.
    data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
    position = (position + len(update)) % (len(data) + 1)
    sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

    return buffer_state.replace(
        data=data,
        insert_position=position,
        sample_position=sample_position,
    )

  def sample_internal(
      self, buffer_state: ReplayBufferState
  ) -> Tuple[ReplayBufferState, Sample]:
    raise NotImplementedError(f'{self.__class__}.sample() is not implemented.')

  def size(self, buffer_state: ReplayBufferState) -> int:
    return buffer_state.insert_position - buffer_state.sample_position  # pytype: disable=bad-return-type  # jax-ndarray


class Queue(QueueBase[Sample], Generic[Sample]):
  """Implements a limited-size queue replay buffer."""

  def __init__(
      self,
      max_replay_size: int,
      dummy_data_sample: Sample,
      sample_batch_size: int,
      cyclic: bool = False,
  ):
    """Initializes the queue.

    Args:
      max_replay_size: Maximum number of elements queue can have.
      dummy_data_sample: Example record to be stored in the queue, it is used to
        derive shapes.
      sample_batch_size: How many elements sampling from the queue should return
        in a batch.
      cyclic: Should sampling from the queue behave cyclicly, ie. once recently
        inserted element was sampled, sampling starts from the beginning of the
        buffer. For example, if the current queue content is [0, 1, 2] and
        `sample_batch_size` is 2, then consecutive calls to sample will give:
        [0, 1], [2, 0], [1, 2]...
    """
    super().__init__(max_replay_size, dummy_data_sample, sample_batch_size)
    self._cyclic = cyclic

  def check_can_sample(self, buffer_state, shards):
    """Checks whether sampling can be performed. Do not JIT this method."""
    assert isinstance(shards, int), 'This method should not be JITed.'
    if self._size < self._sample_batch_size:
      raise ValueError(
          f'Trying to sample {self._sample_batch_size * shards} elements, but'
          f' only {self._size * shards} available.'
      )
    if not self._cyclic:
      self._size -= self._sample_batch_size

  def sample_internal(
      self, buffer_state: ReplayBufferState
  ) -> Tuple[ReplayBufferState, Sample]:
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
    idx = (jnp.arange(self._sample_batch_size) + buffer_state.sample_position) % buffer_state.insert_position

    flat_batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')

    # Remove the sampled batch from the queue.
    sample_position = buffer_state.sample_position + self._sample_batch_size
    if self._cyclic:
      sample_position = sample_position % buffer_state.insert_position

    new_state = buffer_state.replace(sample_position=sample_position)
    return new_state, self._unflatten_fn(flat_batch)

  def size(self, buffer_state: ReplayBufferState) -> int:
    if self._cyclic:
      return buffer_state.insert_position  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      return buffer_state.insert_position - buffer_state.sample_position  # pytype: disable=bad-return-type  # jax-ndarray


class UniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
  """Implements an uniform sampling limited-size replay queue.

  * It behaves as a limited size queue (if buffer is full it removes the oldest
    elements when new one is inserted).
  * It supports batch insertion only (no single element)
  * It performs uniform random sampling with replacement of a batch of size
    `sample_batch_size`
  """

  def sample_internal(
      self, buffer_state: ReplayBufferState
  ) -> Tuple[ReplayBufferState, Sample]:
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'Data shape expected by the replay buffer ({self._data_shape}) does '
          f'not match the shape of the buffer state ({buffer_state.data.shape})'
      )

    key, sample_key = jax.random.split(buffer_state.key)
    idx = jax.random.randint(
        sample_key,
        (self._sample_batch_size,),
        minval=buffer_state.sample_position,
        maxval=buffer_state.insert_position,
    )
    batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
    return buffer_state.replace(key=key), self._unflatten_fn(batch)


class PmapWrapper(ReplayBuffer[State, Sample]):
  """Wrapper to distribute the buffer on multiple devices.

  Each device stores a replay buffer 'buffer' such that no data moves from one
  device to another.
  The total capacity of this replay buffer is the number of devices multiplied
  by the size of the wrapped buffer.
  The sample size is also the number of devices multiplied by the size of the
  wrapped buffer.
  This should not be used inside a pmapped function:
  You should just use the regular replay buffer in that case.
  """

  def __init__(
      self,
      buffer: ReplayBuffer[State, Sample],
      local_device_count: Optional[int] = None,
  ):
    self._buffer = buffer
    self._num_devices = local_device_count or jax.local_device_count()

  def init(self, key: PRNGKey) -> State:
    key = jax.random.fold_in(key, jax.process_index())
    keys = jax.random.split(key, self._num_devices)
    return jax.pmap(self._buffer.init)(keys)

  # NB: In multi-hosts setups, every host is expected to give a different batch.
  def insert(self, buffer_state: State, samples: Sample) -> State:
    self._buffer.check_can_insert(buffer_state, samples, self._num_devices)
    samples = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, self._num_devices) + x.shape[1:]), samples
    )
    # This is to enforce we're gonna iterate on the start of the batch before
    # the end of the batch.
    samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
    return jax.pmap(self._buffer.insert_internal)(buffer_state, samples)

  # NB: In multi-hosts setups, every host will get a different batch.
  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    self._buffer.check_can_sample(buffer_state, self._num_devices)
    buffer_state, samples = jax.pmap(self._buffer.sample_internal)(buffer_state)
    samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
    samples = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), samples
    )
    return buffer_state, samples

  def insert_internal(self, buffer_state: State, samples: Sample) -> State:
    """Insert data into the replay buffer."""
    raise ValueError('This function should not be called.')

  def sample_internal(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""
    raise ValueError('This function should not be called.')

  def size(self, buffer_state: State) -> int:
    axis_name = 'x'

    def psize(buffer_state):
      return jax.lax.psum(self._buffer.size(buffer_state), axis_name=axis_name)

    return jax.pmap(psize, axis_name=axis_name)(buffer_state)[0]


class PjitWrapper(ReplayBuffer[State, Sample]):
  """Wrapper to distribute the buffer on multiple devices with pjit.

  Each device stores a part of the replay buffer depending on its index on axis
  'axis_name'.
  The total capacity of this replay buffer is the size of the mesh multiplied
  by the size of the wrapped buffer.
  The sample size is also the size of the mesh multiplied by the size of the
  sample in the wrapped buffer. Sample batches from each shard are concatenated
  (i.e. for random sampling, each shard will sample from the data they can see).
  """

  def __init__(
      self,
      buffer: ReplayBuffer[State, Sample],
      mesh: jax.sharding.Mesh,
      axis_names: Sequence[str],
  ):
    """Constructor.

    Args:
      buffer: The buffer to replicate.
      mesh: Device mesh for pjitting context.
      axis_names: The axes along which the replay buffer data should be
        partitionned.
    """
    self._buffer = buffer
    self._mesh = mesh
    self._num_devices = math.prod(mesh.shape[name] for name in axis_names)

    def init(key: PRNGKey) -> State:
      keys = jax.random.split(key, self._num_devices)
      return jax.vmap(self._buffer.init)(keys)

    def insert(buffer_state: State, samples: Sample) -> State:
      samples = jax.tree_util.tree_map(
          lambda x: jnp.reshape(x, (-1, self._num_devices) + x.shape[1:]),
          samples,
      )
      # This is to enforce we're gonna iterate on the start of the batch before
      # the end of the batch.
      samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
      return jax.vmap(self._buffer.insert_internal)(buffer_state, samples)

    def sample(buffer_state: State) -> Tuple[State, Sample]:
      buffer_state, samples = jax.vmap(self._buffer.sample_internal)(
          buffer_state
      )
      samples = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), samples)
      samples = jax.tree_util.tree_map(
          lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), samples
      )
      return buffer_state, samples

    def size(buffer_state: State) -> int:
      return jnp.sum(jax.vmap(self._buffer.size)(buffer_state))  # pytype: disable=bad-return-type  # jnp-type

    partition_spec = jax.sharding.PartitionSpec((axis_names),)
    self._partitioned_init = pjit.pjit(init, out_shardings=partition_spec)
    self._partitioned_insert = pjit.pjit(
        insert,
        out_shardings=partition_spec,
    )
    self._partitioned_sample = pjit.pjit(
        sample,
        out_shardings=partition_spec,
    )
    # This will return the TOTAL size across all devices.
    self._partitioned_size = pjit.pjit(size, out_shardings=None)

  def init(self, key: PRNGKey) -> State:
    """See base class."""
    with self._mesh:
      return self._partitioned_init(key)

  def insert(self, buffer_state: State, samples: Sample) -> State:
    """See base class."""
    self._buffer.check_can_insert(buffer_state, samples, self._num_devices)
    with self._mesh:
      return self._partitioned_insert(buffer_state, samples)

  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    """See base class."""
    self._buffer.check_can_sample(buffer_state, self._num_devices)
    with self._mesh:
      return self._partitioned_sample(buffer_state)

  def size(self, buffer_state: State) -> int:
    """See base class. The total size (sum of all partitions) is returned."""
    with self._mesh:
      return self._partitioned_size(buffer_state)

  def insert_internal(self, buffer_state: State, samples: Sample) -> State:
    """Insert data into the replay buffer."""
    raise ValueError('This function should not be called.')

  def sample_internal(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""
    raise ValueError('This function should not be called.')


@flax.struct.dataclass
class PrimitiveReplayBufferState(Generic[Sample]):
  """The state of the primitive replay buffer."""

  samples: Optional[Sample] = None
