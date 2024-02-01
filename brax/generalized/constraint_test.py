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
"""Tests for constraints."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.generalized import pipeline
import jax
from jax import numpy as jp
import numpy as np


class ConstraintTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ant.xml',),
      ('triple_pendulum.xml',),
      ('humanoid.xml',),
      ('half_cheetah.xml',),
      ('solver_params_v2.xml',),
      ('swimmer.xml',),
  )
  def test_jacobian(self, xml_file):
    """Test constraint jacobian."""
    sys = test_utils.load_fixture(xml_file)
    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)
      efc_j = np.reshape(mj_next.efc_J, (-1, sys.qd_size()))
      efc_aref = np.reshape(mj_next.efc_aref, efc_j.shape[0:1])

      # ignore zero rows (brax jacobians are static size)
      np.testing.assert_almost_equal(
          np.sort(state.con_jac[np.any(state.con_jac, axis=1)], axis=0),
          np.sort(efc_j, axis=0),
          6,
      )
      np.testing.assert_almost_equal(
          np.sort(state.con_aref[np.any(state.con_jac, axis=1)], axis=0),
          np.sort(efc_aref, axis=0),
          3,
      )

  @parameterized.parameters(
      ('ant.xml',),
      ('triple_pendulum.xml',),
      ('humanoid.xml',),
      ('half_cheetah.xml',),
      ('solver_params_v2.xml',),
      ('swimmer.xml',),
  )
  def test_force(self, xml_file):
    """Test constraint force."""
    sys = test_utils.load_fixture(xml_file)
    # brax can offer constraint solutions that are just as good as mujoco's
    # but we have to crank up solver iterations pretty high.  improving
    # brax's constraint solver will have a significant impact on perf
    sys = sys.replace(solver_iterations=500)
    err, mj_err = 0, 0
    # force PGS so we can reference efc_AR:
    samples = test_utils.sample_mujoco_states(xml_file, force_pgs=True)
    for mj_prev, mj_next in samples:
      act = jp.zeros(sys.act_size())
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)
      state = jax.jit(pipeline.step)(sys, state, act)
      efc_jt = np.reshape(mj_next.efc_J, (-1, sys.qd_size())).T
      # recover con_frc by backing it out from qf_constraint
      con_frc = np.linalg.lstsq(efc_jt, state.qf_constraint, None)[0]
      err += np.sum((mj_next.efc_AR @ con_frc + mj_next.efc_b) ** 2)
      mj_err += np.sum(
          (mj_next.efc_AR @ mj_next.efc_force + mj_next.efc_b) ** 2
      )

    self.assertLessEqual(err, mj_err + 0.01)


if __name__ == '__main__':
  absltest.main()
