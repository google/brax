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

"""Test for distributed Brax's Replay buffers.

Note that it doesn't test multi hosts.
"""
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from brax.training import replay_buffers
import jax
from jax._src import config as jax_config
from jax.experimental import global_device_array
from jax.experimental import maps
from jax.experimental import pjit
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np


def get_dummy_data():
  return {'a': jnp.zeros(()), 'b': jnp.zeros((5, 5))}


def get_dummy_batch(batch_size: int = 10):
  return {
      'a':
          jnp.arange(batch_size, dtype=jnp.float32),
      'b':
          jnp.reshape(
              jnp.arange(batch_size * 5 * 5, dtype=jnp.float32),
              (batch_size, 5, 5))
  }


AXIS_NAME = 'x'
EXTRA_AXIS = 'y'


def get_mesh():
  return maps.Mesh(
      np.array(jax.devices()).reshape(2, -1), (AXIS_NAME, EXTRA_AXIS))


QUEUE_FACTORIES = [replay_buffers.UniformSamplingQueue, replay_buffers.Queue]
USE_GDA = [False, True]


class QueueReplayTest(parameterized.TestCase):
  """Tests for Replay Buffers."""

  def assertEqualGDAFriendly(self, data, expected_data):
    if not isinstance(data, global_device_array.GlobalDeviceArray):
      self.assertTrue(jnp.all(data == expected_data))
      return
    for shard in data.local_shards:
      if not shard.index:
        self.assertEqual(shard.data, expected_data)
      else:
        self.assertTrue(jnp.all(shard.data == expected_data[shard.index[0]]))

  @parameterized.parameters(itertools.product(QUEUE_FACTORIES, USE_GDA))
  def testInsert(self, queue_cls, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        queue_cls(
            max_replay_size=16 // mesh.shape[AXIS_NAME],
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

      buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)

      # Hit the max replay_size.
      buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 16)

  # TODO: Try multihost.
  @parameterized.parameters(itertools.product(QUEUE_FACTORIES, USE_GDA))
  def testInsertPartitioned(self, queue_cls, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        queue_cls(
            max_replay_size=16 // mesh.shape[AXIS_NAME],
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME,
        batch_partion_spec=pxla.PartitionSpec(EXTRA_AXIS) if use_gda else None)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

      with mesh:
        partitionned_batch = pjit.pjit(
            lambda x: x,
            in_axis_resources=None,
            out_axis_resources=pxla.PartitionSpec(EXTRA_AXIS))(
                get_dummy_batch(12))

        buffer_state = replay_buffer.insert(buffer_state, partitionned_batch)
        self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 12)

        # Hit the max replay_size.
        buffer_state = replay_buffer.insert(buffer_state, partitionned_batch)
        self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 16)

  @parameterized.parameters(itertools.product(QUEUE_FACTORIES, USE_GDA))
  def testInsertRollingBatches(self, queue_cls, use_gda):
    mesh = get_mesh()
    max_size = 6 // mesh.shape[AXIS_NAME]
    replay_buffer = replay_buffers.PjitWrapper(
        queue_cls(
            max_replay_size=max_size,
            dummy_data_sample=0,
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)
      self.assertEqualGDAFriendly(
          buffer_state.data, jnp.array([[[0], [0], [0]], [[0], [0], [0]]]))

      def insert(buffer_state, batch):
        return replay_buffer.insert(buffer_state, jnp.array(batch))

      buffer_state = insert(buffer_state, [0, 1, 2, 3])
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 4)
      self.assertEqualGDAFriendly(
          buffer_state.data, jnp.array([[[0], [2], [0]], [[1], [3], [0]]]))

      buffer_state = insert(buffer_state, [4, 5, 6, 7])
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 6)
      self.assertEqualGDAFriendly(
          buffer_state.data, jnp.array([[[2], [4], [6]], [[3], [5], [7]]]))

      buffer_state = insert(buffer_state, [8, 9, 10, 11])
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 6)
      self.assertEqualGDAFriendly(
          buffer_state.data, jnp.array([[[8], [10], [6]], [[9], [11], [7]]]))

  @parameterized.parameters(itertools.product(QUEUE_FACTORIES, USE_GDA))
  def testInsertRolling2DimBatches(self, queue_cls, use_gda) -> None:
    mesh = get_mesh()
    max_size = 6 // mesh.shape[AXIS_NAME]
    replay_buffer = replay_buffers.PjitWrapper(
        queue_cls(
            max_replay_size=max_size,
            dummy_data_sample=jnp.zeros((2,)),
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]))

      def insert(buffer_state, batch):
        return replay_buffer.insert(buffer_state,
                                    jnp.array(batch, dtype=jnp.float32))

      buffer_state = insert(buffer_state, [[1, 1], [2, 2], [3, 3], [4, 4]])
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 4)
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[1, 1], [3, 3], [0, 0]], [[2, 2], [4, 4], [0, 0]]]))

      buffer_state = insert(buffer_state, [[5, 5], [6, 6], [7, 7], [8, 8]])
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 6)
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[3, 3], [5, 5], [7, 7]], [[4, 4], [6, 6], [8, 8]]]))

  @parameterized.parameters(USE_GDA)
  def testUniformSamplingQueueSample(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.UniformSamplingQueue(
            max_replay_size=16 // mesh.shape[AXIS_NAME],
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

      buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)

      buffer_state, samples = replay_buffer.sample(buffer_state)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)
      self.assertEqual(samples['a'].shape, (2,))
      self.assertEqual(samples['b'].shape, (2, 5, 5))
      with mesh:
        shifted_samples = pjit.pjit(
            lambda x: x - x[:, :1, :1],
            in_axis_resources=None,
            out_axis_resources=None)(
                samples['b'])
      self.assertEqualGDAFriendly(shifted_samples,
                                  jnp.stack([get_dummy_batch(1)['b']] * 2))

  @parameterized.parameters(USE_GDA)
  def testUniformSamplingQueueCyclicSample(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.UniformSamplingQueue(
            max_replay_size=10 // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert

      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

      buffer_state = insert(buffer_state, jnp.zeros(10, dtype=jnp.int32))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([0, 0]))

  @parameterized.parameters(USE_GDA)
  def testQueueSamplePyTree(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=16 // mesh.shape[AXIS_NAME],
            dummy_data_sample=get_dummy_data(),
            sample_batch_size=2 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)
      buffer_state = replay_buffer.init(rng)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

      buffer_state = replay_buffer.insert(buffer_state, get_dummy_batch())
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)

      buffer_state, samples = replay_buffer.sample(buffer_state)
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 8)
      self.assertEqual(samples['a'].shape, (2,))
      self.assertEqual(samples['b'].shape, (2, 5, 5))
      with mesh:
        shifted_samples = pjit.pjit(
            lambda x: x - x[:, :1, :1],
            in_axis_resources=None,
            out_axis_resources=None)(
                samples['b'])
      self.assertEqualGDAFriendly(shifted_samples,
                                  jnp.stack([get_dummy_batch(1)['b']] * 2))

  @parameterized.parameters(USE_GDA)
  def testQueueSample(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=10 // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=4 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)

      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert
      sample = replay_buffer.sample

      buffer_state = insert(buffer_state, jnp.arange(4))
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[0], [2], [0], [0], [0]], [[1], [3], [0], [0], [0]]]))
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([2, 2]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 4)

      buffer_state = insert(buffer_state, jnp.arange(4, 10))
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[0], [2], [4], [6], [8]], [[1], [3], [5], [7], [9]]]))
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([0, 0]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)

      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.array([0, 1, 2, 3]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 6)

      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.array([4, 5, 6, 7]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 2)

      buffer_state = insert(buffer_state, jnp.arange(20, 26))
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.array([8, 9, 20, 21]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 4)

      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.array([22, 23, 24, 25]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 0)

  @parameterized.parameters(USE_GDA)
  def testQueueInsertWhenFull(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=10 // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=4 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)

      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert

      buffer_state = insert(buffer_state, jnp.arange(10))
      buffer_state = insert(buffer_state, jnp.arange(10, 12))
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[10], [2], [4], [6], [8]], [[11], [3], [5], [7], [9]]]))
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([1, 1]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)

  @parameterized.parameters(USE_GDA)
  def testQueueWrappedSample(self, use_gda):
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=10 // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=8 // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)

      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert
      sample = replay_buffer.sample

      buffer_state = insert(buffer_state, jnp.arange(10))
      buffer_state = insert(buffer_state, jnp.arange(10, 16))
      self.assertEqualGDAFriendly(
          buffer_state.data,
          jnp.array([[[10], [12], [14], [6], [8]], [[11], [13], [15], [7],
                                                    [9]]]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 10)
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([3, 3]))

      # This sample contains elements from both the beggining and the end of
      # the buffer.
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples,
                                  jnp.array([6, 7, 8, 9, 10, 11, 12, 13]))
      self.assertEqualGDAFriendly(replay_buffer.size(buffer_state), 2)
      self.assertEqualGDAFriendly(buffer_state.current_position,
                                  jnp.array([3, 3]))

  @parameterized.parameters(USE_GDA)
  def testQueueBatchSizeEqualsMaxSize(self, use_gda):
    batch_size = 6
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=batch_size // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=batch_size // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)

      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert
      sample = replay_buffer.sample

      buffer_state = insert(buffer_state, jnp.arange(batch_size))
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.arange(batch_size))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([0, 0]))

      buffer_state = insert(buffer_state,
                            jnp.zeros(batch_size, dtype=jnp.int32))
      buffer_state = insert(buffer_state, jnp.ones(batch_size, dtype=jnp.int32))
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples,
                                  jnp.ones(batch_size, dtype=jnp.int32))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([0, 0]))

  @parameterized.parameters(USE_GDA)
  def testQueueSampleFromEmpty(self, use_gda):
    batch_size = 6
    mesh = get_mesh()
    replay_buffer = replay_buffers.PjitWrapper(
        replay_buffers.Queue(
            max_replay_size=batch_size // mesh.shape[AXIS_NAME],
            dummy_data_sample=0,
            sample_batch_size=batch_size // mesh.shape[AXIS_NAME]),
        mesh=mesh,
        axis_name=AXIS_NAME)
    with jax_config.parallel_functions_output_gda(use_gda):
      rng = jax.random.PRNGKey(0)

      buffer_state = replay_buffer.init(rng)
      insert = replay_buffer.insert
      sample = replay_buffer.sample

      buffer_state = insert(buffer_state, jnp.arange(batch_size))
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.arange(batch_size))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([0, 0]))

      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples,
                                  jnp.zeros(batch_size, dtype=jnp.int32))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([0, 0]))

      buffer_state = insert(buffer_state, jnp.arange(10, 14))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([2, 2]))
      buffer_state, samples = sample(buffer_state)
      self.assertEqualGDAFriendly(samples, jnp.array([10, 11, 12, 13, 0, 0]))
      self.assertEqualGDAFriendly(buffer_state.current_size, jnp.array([0, 0]))


if __name__ == '__main__':
  absltest.main()
