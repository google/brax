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

"""Tests for running statistics."""

from absl.testing import absltest
from absl.testing import parameterized
from brax.training.acme import running_statistics
import jax.numpy as jnp


class UntilCountTest(parameterized.TestCase):

  @parameterized.parameters('welford', 'ema')
  def test_statistics_freeze_after_until_count(self, mode):
    state = running_statistics.init_state(jnp.zeros(1), mode=mode)
    for value in (0.0, 10.0):
      state = running_statistics.update(
          state, jnp.full((4, 1), value), until_count=8
      )
    frozen = state
    # The count is now 8, so further batches must not move the statistics.
    state = running_statistics.update(
        state, jnp.full((4, 1), 100.0), until_count=8
    )
    self.assertEqual(float(frozen.mean[0]), 5.0)
    self.assertEqual(float(state.mean[0]), float(frozen.mean[0]))
    self.assertEqual(float(state.std[0]), float(frozen.std[0]))
    self.assertEqual(
        float(state.summed_variance[0]), float(frozen.summed_variance[0])
    )

  @parameterized.parameters('welford', 'ema')
  def test_statistics_update_before_until_count(self, mode):
    state = running_statistics.init_state(jnp.zeros(1), mode=mode)
    state = running_statistics.update(
        state, jnp.full((4, 1), 6.0), until_count=8
    )
    self.assertEqual(float(state.mean[0]), 6.0)


if __name__ == '__main__':
  absltest.main()
