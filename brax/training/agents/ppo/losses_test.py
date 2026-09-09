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

"""Tests for PPO losses."""

from absl.testing import absltest
from brax.training.agents.ppo import losses as ppo_losses
import jax
from jax import numpy as jnp
import numpy as np


class QuantileHuberLossTest(absltest.TestCase):
  """Tests for quantile_huber_loss."""

  def testSymmetricTarget(self):
    """Target at median of quantiles — symmetric around center."""
    # quantiles: [1, 2, 3, 4, 5], target: 3.0, kappa: 1.0
    # 5 quantiles -> tau = [0.1, 0.3, 0.5, 0.7, 0.9]
    #
    # delta = target - quantiles = [2, 1, 0, -1, -2]
    # |delta| =                    [2, 1, 0, 1,  2]
    #
    # Huber(kappa=1.0):
    #   |d|<=1: 0.5*d^2     -> [-, 0.5, 0.0, 0.5, -]
    #   |d|>1:  k*(|d|-0.5k) -> [1.5, -, -, -, 1.5]
    #   huber = [1.5, 0.5, 0.0, 0.5, 1.5]
    #
    # (delta < 0) = [0, 0, 0, 1, 1]
    # weight = |tau - 1(d<0)| = [0.1, 0.3, 0.5, 0.3, 0.1]
    #
    # loss_elements = weight * huber = [0.15, 0.15, 0.0, 0.15, 0.15]
    # mean = 0.6 / 5 = 0.12
    quantiles = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # (1, 1, 5)
    targets = jnp.array([[3.0]])  # (1, 1)
    loss = ppo_losses.quantile_huber_loss(quantiles, targets, kappa=1.0)
    np.testing.assert_allclose(float(loss), 0.12, atol=1e-6)

  def testAsymmetricTargetAbove(self):
    """Target above all quantiles — higher tau penalizes more."""
    # quantiles: [1, 2, 3, 4, 5], target: 6.0, kappa: 1.0
    # tau = [0.1, 0.3, 0.5, 0.7, 0.9]
    #
    # delta = [5, 4, 3, 2, 1] (all positive, target > all quantiles)
    # (delta < 0) = [0, 0, 0, 0, 0]
    # weight = |tau - 0| = tau = [0.1, 0.3, 0.5, 0.7, 0.9]
    #
    # Huber: [1*(5-0.5)=4.5, 1*(4-0.5)=3.5, 1*(3-0.5)=2.5,
    #         1*(2-0.5)=1.5, 0.5*1^2=0.5]
    #
    # loss = [0.45, 1.05, 1.25, 1.05, 0.45]
    # mean = 4.25 / 5 = 0.85
    quantiles = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    targets = jnp.array([[6.0]])
    loss = ppo_losses.quantile_huber_loss(quantiles, targets, kappa=1.0)
    np.testing.assert_allclose(float(loss), 0.85, atol=1e-6)

  def testZeroDelta(self):
    """Target equals all quantiles — loss should be zero."""
    quantiles = jnp.array([[[3.0, 3.0, 3.0, 3.0, 3.0]]])
    targets = jnp.array([[3.0]])
    loss = ppo_losses.quantile_huber_loss(quantiles, targets, kappa=1.0)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)

  def testBatchAndTimeDims(self):
    """Verify loss averages correctly over T, B, and quantile dims."""
    # Shape: (T=2, B=1, Q=5) — repeat same inputs over time
    quantiles = jnp.array([
        [[1.0, 2.0, 3.0, 4.0, 5.0]],  # t=0
        [[1.0, 2.0, 3.0, 4.0, 5.0]],  # t=1
    ])  # (2, 1, 5)
    targets = jnp.array([[3.0], [3.0]])  # (2, 1)
    loss = ppo_losses.quantile_huber_loss(quantiles, targets, kappa=1.0)
    # Same as single-step case since inputs are identical
    np.testing.assert_allclose(float(loss), 0.12, atol=1e-6)


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
