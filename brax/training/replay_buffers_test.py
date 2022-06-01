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

"""Test for Brax's Replay buffers."""

from typing import List

from absl.testing import absltest
from brax.training import replay_buffers
import jax
import jax.numpy as jnp


def get_dummy_data():
  return {'a': jnp.zeros(()), 'b': jnp.zeros((5, 5))}


def get_dummy_batch(batch_size: int = 8):
  return {
      'a':
          jnp.arange(batch_size, dtype=jnp.float32),
      'b':
          jnp.reshape(
              jnp.arange(batch_size * 5 * 5, dtype=jnp.float32),
              (batch_size, 5, 5))
  }


def sorted_data(buffer_state: replay_buffers._ReplayBufferState) -> List[int]:
  """Returns the data of the buffer, rolled so set the current position to 0."""
  return jnp.sort(jnp.squeeze(buffer_state.data, axis=1)).tolist()


class UniformSamplingQueueReplayTest(absltest.TestCase):
  """Tests for Replay Buffers."""

  def testInsert(self):
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=10,
        dummy_data_sample=get_dummy_data(),
        sample_batch_size=2)
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    self.assertEqual(replay_buffer.size(buffer_state), 0)

    buffer_state = jax.jit(replay_buffer.insert)(buffer_state,
                                                 get_dummy_batch())
    self.assertEqual(replay_buffer.size(buffer_state), 8)

    # Hit the max replay_size.
    buffer_state = jax.jit(replay_buffer.insert)(buffer_state,
                                                 get_dummy_batch())
    self.assertEqual(replay_buffer.size(buffer_state), 10)

  def testInvalidStateShape(self) -> None:
    max_replay_size = 10
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=get_dummy_data(),
        sample_batch_size=2)
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    insert = jax.jit(replay_buffer.insert)

    # Make sure inserting in the buffer works wihtout crashing.
    insert(buffer_state, get_dummy_batch())

    # Expect an exception if `buffer_state.data` was corrupted.
    invalid_state = buffer_state.replace(data=jnp.arange(10))
    with self.assertRaises(ValueError) as context_manager:
      insert(invalid_state, get_dummy_batch())
    self.assertContainsSubsequence(str(context_manager.exception), 'shape')

    # Expect an exception if batch_size is larger than max_replay_size.
    insert(buffer_state, get_dummy_batch(max_replay_size))
    with self.assertRaises(ValueError) as context_manager:
      insert(buffer_state, get_dummy_batch(max_replay_size + 1))
    self.assertContainsSubsequence(
        str(context_manager.exception), 'maximum replay size')

  def testInsertRollingBatches(self) -> None:
    max_size = 5
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_size, dummy_data_sample=0, sample_batch_size=2)
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    self.assertEqual(replay_buffer.size(buffer_state), 0)
    self.assertEqual(sorted_data(buffer_state), [0, 0, 0, 0, 0])

    @jax.jit
    def insert(buffer_state, batch):
      return replay_buffer.insert(buffer_state, jnp.array(batch))

    buffer_state = insert(buffer_state, [1, 2, 3])
    self.assertEqual(replay_buffer.size(buffer_state), 3)
    self.assertEqual(sorted_data(buffer_state), [0, 0, 1, 2, 3])

    buffer_state = insert(buffer_state, [4, 5, 6])
    self.assertEqual(replay_buffer.size(buffer_state), 5)
    self.assertEqual(sorted_data(buffer_state), [2, 3, 4, 5, 6])

    buffer_state = insert(buffer_state, [7, 8, 9, 10, 11])
    self.assertEqual(replay_buffer.size(buffer_state), 5)
    self.assertEqual(sorted_data(buffer_state), [7, 8, 9, 10, 11])

  def testInsertRolling2DimBatches(self) -> None:
    max_size = 5
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_size,
        dummy_data_sample=jnp.zeros((2,)),
        sample_batch_size=2)
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    self.assertEqual(replay_buffer.size(buffer_state), 0)
    self.assertTrue(
        jnp.all(buffer_state.data == jnp.array([[0, 0], [0, 0], [0, 0], [0, 0],
                                                [0, 0]])))

    @jax.jit
    def insert(buffer_state, batch):
      return replay_buffer.insert(buffer_state,
                                  jnp.array(batch, dtype=jnp.float32))

    buffer_state = insert(buffer_state, [[1, 1], [2, 2], [3, 3]])
    self.assertEqual(replay_buffer.size(buffer_state), 3)
    self.assertTrue(
        jnp.all(buffer_state.data == jnp.array([[1, 1], [2, 2], [3, 3], [0, 0],
                                                [0, 0]])))

    buffer_state = insert(buffer_state, [[4, 4], [5, 5], [6, 6]])
    self.assertEqual(replay_buffer.size(buffer_state), 5)
    self.assertTrue(
        jnp.all(buffer_state.data == jnp.array([[2, 2], [3, 3], [4, 4], [5, 5],
                                                [6, 6]])))

  def testSample(self):
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=10,
        dummy_data_sample=get_dummy_data(),
        sample_batch_size=2)
    rng = jax.random.PRNGKey(0)
    buffer_state = replay_buffer.init(rng)
    self.assertEqual(replay_buffer.size(buffer_state), 0)

    buffer_state = jax.jit(replay_buffer.insert)(buffer_state,
                                                 get_dummy_batch())
    self.assertEqual(replay_buffer.size(buffer_state), 8)

    buffer_state, samples = jax.jit(replay_buffer.sample)(buffer_state)
    self.assertEqual(replay_buffer.size(buffer_state), 8)
    self.assertEqual(samples['a'].shape, (2,))
    self.assertEqual(samples['b'].shape, (2, 5, 5))
    for sample in samples['b']:
      self.assertSequenceEqual(
          list(jnp.reshape(sample - sample[0, 0], (-1,))), range(5 * 5))


if __name__ == '__main__':
  absltest.main()
