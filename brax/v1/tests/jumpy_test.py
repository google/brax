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

"""Tests for brax.jumpy."""

from absl.testing import absltest
from brax.v1 import jumpy as jp
import jax
from jax import numpy as jnp
import numpy as np


class ForiLoopTest(absltest.TestCase):
  """Tests jumpy.fori_loop when jitted and not jitted."""

  def testForiLoopTest(self):
    a = jp.fori_loop(2, 4, lambda i, x: i + x, jp.array(1.))
    self.assertIsInstance(a, np.float64)
    self.assertEqual(a.shape, ())
    self.assertAlmostEqual(a, 1.0 + 2.0 + 3.0)

  def testForiLoopTestJit(self):
    a = jax.jit(lambda: jp.fori_loop(2, 4, lambda i, x: i + x, jp.array(1.)))()
    self.assertIsInstance(a, jnp.ndarray)
    self.assertEqual(a.shape, ())
    self.assertAlmostEqual(a, 1.0 + 2.0 + 3.0)


if __name__ == '__main__':
  absltest.main()
