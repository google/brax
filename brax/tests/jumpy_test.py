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

"""Tests for brax.jumpy."""

import jax
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp

from brax import jumpy as jp


class ForiLoopTest(absltest.TestCase):
  """Tests jumpy.fori_loop when jitted and not jitted."""
  WANT = 1. + 2. + 3.

  def function(self) -> jp.ndarray:
    def body_func(i, a):
      return i + a

    a = jp.fori_loop(2, 4, body_func, jp.array(1.))
    return a

  def testForiLoopTest(self):
    a = self.function()
    self.assertIsInstance(a, np.float_)
    self.assertEqual(a.shape, ())
    self.assertAlmostEqual(a, np.array(self.WANT))

  def testForiLoopTestJit(self):
    compiled = jax.jit(self.function)
    a = compiled()
    self.assertIsInstance(a, jnp.ndarray)
    self.assertEqual(a.shape, ())
    self.assertAlmostEqual(a, jnp.array(self.WANT))


if __name__ == '__main__':
  absltest.main()
