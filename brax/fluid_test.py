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
"""Tests for fluid."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.generalized import dynamics
from brax.generalized import pipeline as g_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline
import jax
from jax import numpy as jp
import mujoco
import numpy as np

assert_almost_equal = np.testing.assert_almost_equal


class FluidTest(parameterized.TestCase):

  @parameterized.parameters(
      ('fluid_box.xml', 0.0, 0.0),
      ('fluid_box.xml', 0.0, 1.3),
      ('fluid_box.xml', 2.0, 0.0),
      ('fluid_box.xml', 2.0, 1.3),
      ('fluid_box_offset_com.xml', 0.0, 1.3),
      ('fluid_box_offset_com.xml', 2.0, 0.0),
      ('fluid_box_offset_com.xml', 2.0, 1.3),
  )
  def test_fluid_mj_generalized(self, config, density, viscosity):
    """Tests fluid interactions."""
    sys = test_utils.load_fixture(config)
    sys = sys.replace(density=density, viscosity=viscosity)

    mj_model = test_utils.load_fixture_mujoco(config)
    mj_model.opt.density = density
    mj_model.opt.viscosity = viscosity
    mj_data = mujoco.MjData(mj_model)
    # initialize qd so that the object interacts with the fluid
    mj_data.qvel[:3] = [1, 2, 4]
    mj_data.qvel[3] = jp.pi
    q, qd = jp.asarray(mj_data.qpos), jp.asarray(mj_data.qvel)

    # check qfrc_passive after the first step
    mujoco.mj_step(mj_model, mj_data)
    state = jax.jit(g_pipeline.init)(sys, q, qd)
    qfrc_passive = jax.jit(dynamics._passive)(sys, state)
    np.testing.assert_array_almost_equal(
        qfrc_passive[3:], mj_data.qfrc_passive[3:], 2
    )
    m = max(jp.abs(mj_data.qfrc_passive[:3])) + 1e-6
    np.testing.assert_array_almost_equal(
        qfrc_passive[:3] / m, mj_data.qfrc_passive[:3] / m, 1
    )

    # check q/qd after multiple steps
    for _ in range(500):
      mujoco.mj_step(mj_model, mj_data)
    mq, mqd = jp.asarray(mj_data.qpos), jp.asarray(mj_data.qvel)

    for _ in range(500):
      state = jax.jit(g_pipeline.step)(sys, state, jp.zeros((sys.act_size(),)))
    gq, gqd = state.q, state.qd

    np.testing.assert_array_almost_equal(gq, mq, 2)
    np.testing.assert_array_almost_equal(gqd, mqd, 2)

  @parameterized.parameters(
      ('fluid_sphere.xml', p_pipeline, 0.0, 0.0),
      ('fluid_sphere.xml', p_pipeline, 0.0, 1.3),
      ('fluid_sphere.xml', p_pipeline, 2.0, 0.0),
      ('fluid_sphere.xml', p_pipeline, 2.0, 1.3),
      ('fluid_sphere.xml', s_pipeline, 0.0, 0.0),
      ('fluid_sphere.xml', s_pipeline, 0.0, 1.3),
      ('fluid_sphere.xml', s_pipeline, 2.0, 0.0),
      ('fluid_sphere.xml', s_pipeline, 2.0, 1.3),
      ('fluid_two_spheres.xml', p_pipeline, 2.0, 1.3),
      ('fluid_two_spheres.xml', s_pipeline, 2.0, 1.3),
  )
  def test_fluid_positional_spring(self, config, pipeline, density, viscosity):
    """Tests fluid interactions for pbd/spring compared to generalized."""
    sys = test_utils.load_fixture(config)
    sys = sys.replace(density=density, viscosity=viscosity)

    q, qd = sys.init_q, jp.zeros(sys.qd_size())
    qd = qd.at[:3].set(jp.array([1, 2, 4]))
    qd = qd.at[3].set(jp.pi)

    state_g = jax.jit(g_pipeline.init)(sys, q, qd)
    for _ in range(500):
      state_g = jax.jit(g_pipeline.step)(
          sys, state_g, jp.zeros((sys.act_size(),))
      )
    gq, gqd = state_g.q, state_g.qd

    state = jax.jit(pipeline.init)(sys, q, qd)
    for _ in range(500):
      state = jax.jit(pipeline.step)(sys, state, jp.zeros((sys.act_size(),)))
    tq, tqd = state.q, state.qd

    np.testing.assert_array_almost_equal(tq, gq, 3)
    np.testing.assert_array_almost_equal(tqd, gqd, 2)


if __name__ == '__main__':
  absltest.main()
