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

"""Tests for geometry."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import math
import jax
from jax import numpy as jp
import numpy as np


def _get_rand_norm(seed: int):
  np.random.seed(seed)
  theta = np.random.random(1) * 2 * np.pi
  a = (np.random.random(1) - 0.5) * 2.0
  phi = np.arccos(a)
  x = np.sin(phi) * np.cos(theta)
  y = np.sin(phi) * np.sin(theta)
  z = np.cos(phi)
  return jp.array([x, y, z]).squeeze()


class MathTest(absltest.TestCase):

  def test_inv_approximate(self):
    # create a 4x4 matrix we know is invertible
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    x = jp.eye(4) * 0.001 + x @ x.T

    x_inv = jp.linalg.inv(x)
    x_inv_approximate = math.inv_approximate(x, jp.zeros((4, 4)), num_iter=100)

    np.testing.assert_array_almost_equal(x_inv_approximate, x_inv)

  def test_from_to(self):
    v1 = jp.array([1.0, 0.0, 0.0])
    rot = math.from_to(v1, v1)
    np.testing.assert_array_almost_equal(v1, math.rotate(v1, rot))

    rot = math.from_to(v1, -v1)
    np.testing.assert_array_almost_equal(-v1, math.rotate(v1, rot))

    rot = math.from_to(-v1, v1)
    np.testing.assert_array_almost_equal(v1, math.rotate(-v1, rot))

    v1 = jp.array([0.0, 1.0, 0.0])
    rot = math.from_to(v1, -v1)
    np.testing.assert_array_almost_equal(-v1, math.rotate(v1, rot))

    v2 = jp.array([-0.5, 0.5, 0.0])
    v2 /= jp.linalg.norm(v2)
    rot = math.from_to(v1, v2)
    np.testing.assert_array_almost_equal(v2, math.rotate(v1, rot))


class OrthoganalsTest(parameterized.TestCase):
  """Tests the orthogonals function."""

  @parameterized.parameters(range(100))
  def test_orthogonals(self, i):
    a = _get_rand_norm(i)
    b, c = math.orthogonals(a)
    np.testing.assert_almost_equal(jp.linalg.norm(a), 1)
    np.testing.assert_almost_equal(jp.linalg.norm(b), 1)
    np.testing.assert_almost_equal(jp.linalg.norm(c), 1)
    self.assertAlmostEqual(np.abs(a.dot(b)), 0, 6)
    self.assertAlmostEqual(np.abs(b.dot(c)), 0, 6)
    self.assertAlmostEqual(np.abs(a.dot(c)), 0, 6)


if __name__ == '__main__':
  absltest.main()
