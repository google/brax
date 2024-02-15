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

"""Test for Brax's Replay buffers."""

from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from brax.training import replay_buffers
import jax
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import numpy as np


def get_dummy_data():
  return {'a': jnp.zeros(()), 'b': jnp.zeros((5, 5))}


def get_dummy_batch(batch_size: int = 8):
  return {
      'a': jnp.arange(batch_size, dtype=jnp.float32),
      'b': jnp.reshape(
          jnp.arange(batch_size * 5 * 5, dtype=jnp.float32), (batch_size, 5, 5)
      ),
  }


AXIS_NAME = 'x'
EXTRA_AXIS = 'y'


def get_mesh():
  devices = jax.devices()
  return jax.sharding.Mesh(
      np.array(devices).reshape(2 if len(devices) % 2 == 0 else 1, -1),
      (AXIS_NAME, EXTRA_AXIS),
  )


def sorted_data(buffer_state: replay_buffers.ReplayBufferState) -> List[int]:
  """Returns the data of the buffer, rolled so set the current position to 0."""
  return jnp.sort(jnp.ravel(buffer_state.data)).tolist()


def no_wrap(buffer):
  return buffer


def pjit_wrap(buffer):
  return replay_buffers.PjitWrapper(
      buffer,
      mesh=get_mesh(),
      axis_names=(AXIS_NAME,),
  )


def pmap_wrap(buffer):
  buffer = replay_buffers.PmapWrapper(
      buffer, local_device_count=2 if len(jax.devices()) > 1 else 1
  )
  buffer.insert_internal = jax.jit(buffer.insert_internal)
  buffer.sample_internal = jax.jit(buffer.sample_internal)
  return buffer


def jit_wrap(buffer):
  buffer.insert_internal = jax.jit(buffer.insert_internal)
  buffer.sample_internal = jax.jit(buffer.sample_internal)
  return buffer


QUEUE_FACTORIES = [
    (replay_buffers.UniformSamplingQueue, no_wrap),
    (replay_buffers.UniformSamplingQueue, pjit_wrap),
    (replay_buffers.UniformSamplingQueue, pmap_wrap),
    (replay_buffers.UniformSamplingQueue, jit_wrap),
    (replay_buffers.Queue, no_wrap),
    (replay_buffers.Queue, pjit_wrap),
    (replay_buffers.Queue, pmap_wrap),
    (replay_buffers.Queue, jit_wrap),
]


WRAPPERS = [no_wrap, pjit_wrap, pmap_wrap, jit_wrap]


def assert_equal(obj, data, expected_data):
  data = process_allgather(data)
  obj.assertTrue(
      jnp.all(data == expected_data), f'Not equal: {data} and {expected_data}'
  )


class QueueReplayTest(parameterized.TestCase):
  """Tests for Replay Buffers."""

  @parameterized.parameters(QUEUE_FACTORIES)
  def testInsert(self, queue_cls, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    replay_buffer = wrapper(
        queue_cls(
            max_replay_size=16 // size_denominator,
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // size_denominator,
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    assert_equal(self, replay_buffer.size(buffer_state), 0)

    buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
    assert_equal(self, replay_buffer.size(buffer_state), 8)

    # Hit the max replay_size.
    buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch(16))
    assert_equal(self, replay_buffer.size(buffer_state), 16)

  @parameterized.parameters(QUEUE_FACTORIES)
  def testInvalidStateShape(self, queue_cls, wrapper) -> None:
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    max_replay_size = 16 // size_denominator
    replay_buffer = wrapper(
        queue_cls(
            max_replay_size=max_replay_size,
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // mesh.shape[AXIS_NAME],
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)

    # Make sure inserting in the buffer works wihtout crashing.
    replay_buffer.insert(buffer_state, get_dummy_batch())

    # Expect an exception if `buffer_state.data` was corrupted.
    invalid_state = buffer_state.replace(data=jnp.arange(10))
    with self.assertRaises(ValueError) as context_manager:
      replay_buffer.insert(invalid_state, get_dummy_batch())
    self.assertContainsSubsequence(str(context_manager.exception), 'shape')

    # Expect an exception if batch_size is larger than max_replay_size.
    if wrapper in [pjit_wrap, pmap_wrap] and len(jax.devices()) != 1:
      return
    replay_buffer.insert(buffer_state, get_dummy_batch(max_replay_size))
    with self.assertRaisesRegex(
        ValueError,
        (
            'Trying to insert a batch of samples larger than the maximum replay'
            ' size. num_samples: 18, max replay size 16'
        ),
    ) as context_manager:
      replay_buffer.insert(buffer_state, get_dummy_batch(max_replay_size + 2))

  @parameterized.parameters(QUEUE_FACTORIES)
  def testInsertRollingBatches(self, queue_cls, wrapper) -> None:
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    max_size = 6 // size_denominator
    replay_buffer = wrapper(
        queue_cls(
            max_replay_size=max_size, dummy_data_sample=0, sample_batch_size=2
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    assert_equal(self, replay_buffer.size(buffer_state), 0)
    assert_equal(self, sorted_data(buffer_state), [0, 0, 0, 0, 0, 0])

    buffer_state = replay_buffer.insert(buffer_state, jnp.array([1, 2, 3, 4]))
    assert_equal(self, replay_buffer.size(buffer_state), 4)
    assert_equal(self, sorted_data(buffer_state), [0, 0, 1, 2, 3, 4])

    buffer_state = replay_buffer.insert(buffer_state, jnp.array([5, 6, 7, 8]))
    assert_equal(self, replay_buffer.size(buffer_state), 6)
    assert_equal(self, sorted_data(buffer_state), [3, 4, 5, 6, 7, 8])

    buffer_state = replay_buffer.insert(
        buffer_state, jnp.array([7, 8, 9, 10, 11, 12])
    )
    assert_equal(self, replay_buffer.size(buffer_state), 6)
    assert_equal(self, sorted_data(buffer_state), [7, 8, 9, 10, 11, 12])

  @parameterized.parameters(QUEUE_FACTORIES)
  def testInsertRolling2DimBatches(self, queue_cls, wrapper) -> None:
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    max_size = 8
    mesh = get_mesh()
    replay_buffer = wrapper(
        queue_cls(
            max_replay_size=max_size // size_denominator,
            dummy_data_sample=jnp.zeros((2,)),
            sample_batch_size=2,
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    assert_equal(self, replay_buffer.size(buffer_state), 0)
    if len(jax.devices()) == 1:
      assert_equal(
          self,
          buffer_state.data,
          [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
      )

    buffer_state = replay_buffer.insert(
        buffer_state,
        jnp.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=jnp.float32),
    )
    assert_equal(self, replay_buffer.size(buffer_state), 4)
    if len(jax.devices()) == 1:
      assert_equal(
          self,
          buffer_state.data,
          [[1, 1], [2, 2], [3, 3], [4, 4], [0, 0], [0, 0], [0, 0], [0, 0]],
      )

    buffer_state = replay_buffer.insert(
        buffer_state,
        jnp.array(
            [[5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
            dtype=jnp.float32,
        ),
    )
    assert_equal(self, replay_buffer.size(buffer_state), 8)
    if len(jax.devices()) == 1:
      assert_equal(
          self,
          buffer_state.data,
          [[3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]],
      )

  @parameterized.parameters(WRAPPERS)
  def testUniformSamplingQueueSample(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    replay_buffer = wrapper(
        replay_buffers.UniformSamplingQueue(
            max_replay_size=10 // size_denominator,
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // size_denominator,
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    assert_equal(self, replay_buffer.size(buffer_state), 0)

    buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch(8))
    assert_equal(self, replay_buffer.size(buffer_state), 8)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, replay_buffer.size(buffer_state), 8)
    assert_equal(self, samples['a'].shape, (2,))
    assert_equal(self, samples['b'].shape, (2, 5, 5))
    for sample in samples['b']:
      assert_equal(
          self, jnp.reshape(sample - sample[0, 0], (-1,)), range(5 * 5)
      )

  @parameterized.parameters(WRAPPERS)
  def testUniformSamplingQueueCyclicSample(self, wrapper):
    replay_buffer = wrapper(
        replay_buffers.UniformSamplingQueue(
            max_replay_size=10, dummy_data_sample=0, sample_batch_size=2
        )
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)

    assert_equal(self, replay_buffer.size(buffer_state), 0)

    buffer_state = replay_buffer.insert(
        buffer_state, jnp.zeros(10, dtype=jnp.int32)
    )
    assert_equal(self, replay_buffer.size(buffer_state), 10)
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.insert_position, 10)
    else:
      assert_equal(self, buffer_state.insert_position, [5, 5])

  @parameterized.parameters(WRAPPERS)
  def testQueueSamplePyTree(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=16 // size_denominator,
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // size_denominator,
        ),
    )
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    assert_equal(self, replay_buffer.size(buffer_state), 0)

    buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
    assert_equal(self, replay_buffer.size(buffer_state), 8)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, replay_buffer.size(buffer_state), 6)
    assert_equal(self, samples['a'].shape, (2,))
    assert_equal(self, samples['b'].shape, (2, 5, 5))
    for sample in samples['b']:
      assert_equal(
          self, jnp.reshape(sample - sample[0, 0], (-1,)), range(5 * 5)
      )

  @parameterized.parameters(WRAPPERS)
  def testQueueSample(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    mesh = get_mesh()
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=10 // size_denominator,
            dummy_data_sample=0,
            sample_batch_size=4 // size_denominator,
        )
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(4))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(
          self,
          buffer_state.data,
          [[0], [1], [2], [3], [0], [0], [0], [0], [0], [0]],
      )
      assert_equal(self, buffer_state.insert_position, 4)
    else:
      assert_equal(
          self,
          buffer_state.data,
          [[[0], [2], [0], [0], [0]], [[1], [3], [0], [0], [0]]],
      )
      assert_equal(self, buffer_state.insert_position, [2, 2])
    assert_equal(self, replay_buffer.size(buffer_state), 4)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(4, 10))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(
          self,
          buffer_state.data,
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
      )
      assert_equal(self, buffer_state.insert_position, 10)
    else:
      assert_equal(
          self,
          buffer_state.data,
          [[[0], [2], [4], [6], [8]], [[1], [3], [5], [7], [9]]],
      )
      assert_equal(self, buffer_state.insert_position, [5, 5])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [0, 1, 2, 3])
    assert_equal(self, replay_buffer.size(buffer_state), 6)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [4, 5, 6, 7])
    assert_equal(self, replay_buffer.size(buffer_state), 2)

    with self.assertRaisesRegex(
        ValueError, 'Trying to sample 4 elements, but only 2 available.'
    ):
      buffer_state, samples = replay_buffer.sample(buffer_state)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(20, 24))
    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [8, 9, 20, 21])
    assert_equal(self, replay_buffer.size(buffer_state), 2)

  @parameterized.parameters(WRAPPERS)
  def testCyclicQueueSample(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    mesh = get_mesh()
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=10 // size_denominator,
            dummy_data_sample=0,
            sample_batch_size=4 // size_denominator,
            cyclic=True,
        )
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(6))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(
          self,
          buffer_state.data,
          [[0], [1], [2], [3], [4], [5], [0], [0], [0], [0]],
      )
      assert_equal(self, buffer_state.insert_position, 6)
    else:
      assert_equal(
          self,
          buffer_state.data,
          [[[0], [2], [4], [0], [0]], [[1], [3], [5], [0], [0]]],
      )
      assert_equal(self, buffer_state.insert_position, [3, 3])
    assert_equal(self, replay_buffer.size(buffer_state), 6)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [0, 1, 2, 3])
    assert_equal(self, replay_buffer.size(buffer_state), 6)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [4, 5, 0, 1])
    assert_equal(self, replay_buffer.size(buffer_state), 6)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(6, 10))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(
          self,
          buffer_state.data,
          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
      )
      assert_equal(self, buffer_state.insert_position, 10)
      assert_equal(self, buffer_state.sample_position, 2)
    else:
      assert_equal(
          self,
          buffer_state.data,
          [[[0], [2], [4], [6], [8]], [[1], [3], [5], [7], [9]]],
      )
      assert_equal(self, buffer_state.insert_position, [5, 5])
      assert_equal(self, buffer_state.sample_position, [1, 1])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [2, 3, 4, 5])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [6, 7, 8, 9])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(20, 24))
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [4, 5, 6, 7])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [8, 9, 20, 21])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [22, 23, 4, 5])
    assert_equal(self, replay_buffer.size(buffer_state), 10)

  @parameterized.parameters(WRAPPERS)
  def testQueueInsertWhenFull(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    max_replay_size = 10 // size_denominator
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=max_replay_size,
            dummy_data_sample=0,
            sample_batch_size=4 // size_denominator,
        ),
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(10))
    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(10, 12))
    if len(jax.devices()) == 1:
      assert_equal(
          self,
          buffer_state.data,
          [[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
      )
      assert_equal(self, buffer_state.insert_position, 10)
    assert_equal(self, replay_buffer.size(buffer_state), 10)

  @parameterized.parameters(WRAPPERS)
  def testQueueWrappedSample(self, wrapper):
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=10 // size_denominator,
            dummy_data_sample=0,
            sample_batch_size=8 // size_denominator,
        ),
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)
    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(10))
    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(10, 16))
    if len(jax.devices()) == 1:
      assert_equal(
          self,
          buffer_state.data,
          [[6], [7], [8], [9], [10], [11], [12], [13], [14], [15]],
      )
    assert_equal(self, replay_buffer.size(buffer_state), 10)
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.insert_position, 10)
    else:
      assert_equal(self, buffer_state.insert_position, [5, 5])

    # This sample contains elements from both the beggining and the end of
    # the buffer.
    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, jnp.array([6, 7, 8, 9, 10, 11, 12, 13]))
    assert_equal(self, replay_buffer.size(buffer_state), 2)
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.insert_position, 10)
    else:
      assert_equal(self, buffer_state.insert_position, [5, 5])

  @parameterized.parameters(WRAPPERS)
  def testQueueBatchSizeEqualsMaxSize(self, wrapper):
    batch_size = 8
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    mesh = get_mesh()
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=batch_size // size_denominator,
            dummy_data_sample=0,
            sample_batch_size=batch_size // size_denominator,
        )
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(batch_size))
    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, range(batch_size))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.sample_position, 8)
    else:
      assert_equal(self, buffer_state.sample_position, [4, 4])

    buffer_state = replay_buffer.insert(
        buffer_state, jnp.zeros(batch_size, dtype=jnp.int32)
    )
    buffer_state = replay_buffer.insert(
        buffer_state, jnp.ones(batch_size, dtype=jnp.int32)
    )
    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, [1] * batch_size)
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.sample_position, 8)
    else:
      assert_equal(self, buffer_state.sample_position, [4, 4])

  @parameterized.parameters(WRAPPERS)
  def testQueueSampleFromEmpty(self, wrapper) -> None:
    batch_size = 10
    mesh = get_mesh()
    size_denominator = (
        1 if wrapper in [no_wrap, jit_wrap] else mesh.shape[AXIS_NAME]
    )
    replay_buffer = wrapper(
        replay_buffers.Queue(
            max_replay_size=batch_size // size_denominator,
            dummy_data_sample=0,
            sample_batch_size=batch_size // size_denominator,
        )
    )
    rng = jax.random.PRNGKey(0)

    buffer_state = replay_buffer.init(rng)

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(batch_size))
    buffer_state, samples = replay_buffer.sample(buffer_state)
    assert_equal(self, samples, range(batch_size))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.sample_position, 10)
    else:
      assert_equal(self, buffer_state.sample_position, [5, 5])

    with self.assertRaisesRegex(
        ValueError, 'Trying to sample 10 elements, but only 0 available.'
    ):
      replay_buffer.sample(buffer_state)
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.sample_position, 10)
    else:
      assert_equal(self, buffer_state.sample_position, [5, 5])

    buffer_state = replay_buffer.insert(buffer_state, jnp.arange(10, 14))
    if jax.device_count() == 1 or wrapper in [no_wrap, jit_wrap]:
      assert_equal(self, buffer_state.sample_position, 6)
    else:
      assert_equal(self, buffer_state.sample_position, [3, 3])
    with self.assertRaisesRegex(
        ValueError, 'Trying to sample 10 elements, but only 4 available.'
    ):
      buffer_state, samples = replay_buffer.sample(buffer_state)


if __name__ == '__main__':
  absltest.main()
