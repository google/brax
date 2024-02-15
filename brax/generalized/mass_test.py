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

# pylint:disable=g-multiple-import
"""Tests for mass matrices."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.generalized import pipeline
import jax
import mujoco
import numpy as np


class MassTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ant.xml',),
      ('triple_pendulum.xml',),
      ('humanoid.xml',),
      ('half_cheetah.xml',),
  )
  def test_matrix(self, xml_file):
    """Test mass matrix calculation."""
    sys = test_utils.load_fixture(xml_file)
    model = test_utils.load_fixture_mujoco(xml_file)
    mj_mass_mx = np.zeros((sys.qd_size(), sys.qd_size()))

    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)
      mujoco.mj_fullM(model, mj_mass_mx, mj_next.qM)
      np.testing.assert_almost_equal(state.mass_mx, mj_mass_mx, 5)


if __name__ == '__main__':
  absltest.main()
