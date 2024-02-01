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

"""Tests for brax.base."""

from absl.testing import absltest
from brax import test_utils
import jax
import numpy as np


class BaseTest(absltest.TestCase):

  def test_write_mass_array(self):
    sys = test_utils.load_fixture('ant.xml')
    rng = jax.random.PRNGKey(0)
    noise = jax.random.uniform(rng, (sys.link.inertia.mass.shape[0],))
    new_mass = sys.link.inertia.mass + noise

    sys_w = sys.tree_replace({'link.inertia.mass': new_mass})
    np.testing.assert_array_equal(sys_w.link.inertia.mass, new_mass)

  def test_write_mass_value(self):
    sys = test_utils.load_fixture('ant.xml')
    sys_w = sys.tree_replace({'link.inertia.mass': 1.0})
    self.assertEqual(sys_w.link.inertia.mass, 1.0)

  def test_write_array(self):
    sys = test_utils.load_fixture('ant.xml')
    np.random.seed(0)

    expected = np.random.uniform(sys.elasticity.shape)
    sys_w = sys.tree_replace({'elasticity': expected})
    np.testing.assert_array_equal(sys_w.elasticity, expected)


if __name__ == '__main__':
  absltest.main()
