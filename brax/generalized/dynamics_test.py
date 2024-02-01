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
"""Tests for dynamics."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.generalized import pipeline
import jax
from jax import numpy as jp
import numpy as np


class DynamicsTest(parameterized.TestCase):

  @parameterized.parameters(
      'ant.xml', 'triple_pendulum.xml', ('humanoid.xml',),
      ('half_cheetah.xml',), ('swimmer.xml',),
  )
  def test_transform_com(self, xml_file):
    """Test dynamics transform com."""
    sys = test_utils.load_fixture(xml_file)
    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)

      np.testing.assert_almost_equal(
          state.root_com[0], mj_next.subtree_com[0], 5
      )
      mj_cinr_i = np.zeros((state.cinr.i.shape[0], 3, 3))
      mj_cinr_i[:, [0, 1, 2], [0, 1, 2]] = mj_next.cinert[1:, 0:3]  # diagonal
      mj_cinr_i[:, [0, 0, 1], [1, 2, 2]] = mj_next.cinert[1:, 3:6]  # upper tri
      mj_cinr_i[:, [1, 2, 2], [0, 0, 1]] = mj_next.cinert[1:, 3:6]  # lower tri
      mj_cinr_pos = mj_next.cinert[1:, 6:9]

      np.testing.assert_almost_equal(state.cinr.i, mj_cinr_i, 5)
      np.testing.assert_almost_equal(state.cinr.transform.pos, mj_cinr_pos, 5)
      np.testing.assert_almost_equal(state.cinr.mass, mj_next.cinert[1:, 9], 6)
      np.testing.assert_almost_equal(state.cd.matrix(), mj_next.cvel[1:], 4)
      np.testing.assert_almost_equal(state.cdof.matrix(), mj_next.cdof, 6)
      np.testing.assert_almost_equal(state.cdofd.matrix(), mj_next.cdof_dot, 5)

  @parameterized.parameters(
      'ant.xml', 'triple_pendulum.xml', ('humanoid.xml',),
      ('half_cheetah.xml',), ('swimmer.xml',),
  )
  def test_forward(self, xml_file):
    """Test dynamics forward."""
    sys = test_utils.load_fixture(xml_file)
    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      act = jp.zeros(sys.act_size())
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)
      state = jax.jit(pipeline.step)(sys, state, act)

      np.testing.assert_allclose(
          state.qf_smooth, mj_next.qfrc_smooth, rtol=1e-4, atol=1e-4
      )


if __name__ == '__main__':
  absltest.main()
