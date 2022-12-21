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

# pylint:disable=g-multiple-import
"""Tests for actuators."""

from absl.testing import absltest
from brax.v2 import actuator
from brax.v2 import test_utils
from brax.v2.generalized import pipeline
from jax import numpy as jp
import mujoco
import numpy as np

assert_almost_equal = np.testing.assert_almost_equal


# TODO: run actuator tests over all backend pipelines
def _actuator_step(sys, q, qd, dt, act):
  sys = sys.replace(dt=dt)
  state = pipeline.step(sys, pipeline.init(sys, q, qd), act)
  return state.q, state.qd


class ActuatorTest(absltest.TestCase):

  def test_motor(self):
    """Tests a single pendulum with motor actuator."""
    sys = test_utils.load_fixture('single_pendulum_motor.xml')
    mj_model = test_utils.load_fixture_mujoco('single_pendulum_motor.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)

    tau = jp.array([0.5 * 9.81])  # -mgl sin(theta)
    act = jp.array([1.0 / 150.0 * 0.5 * 9.81])
    tau2 = actuator.to_tau(sys, act, q)
    np.testing.assert_array_almost_equal(tau, tau2, 5)

    q2, qd2 = _actuator_step(sys, q, qd, dt=0.01, act=act)
    np.testing.assert_array_almost_equal(q2, jp.array([0]), 5)
    np.testing.assert_array_almost_equal(qd2, jp.array([0]), 5)

  def test_position(self):
    """Tests a single pendulum with position actuator."""
    sys = test_utils.load_fixture('single_pendulum_position.xml')
    mj_model = test_utils.load_fixture_mujoco('single_pendulum_position.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)
    theta = jp.pi / 2.0  # pendulum is vertical
    q = jp.array([theta])

    # a position actuator at the bottom should not move the pendulum
    act = jp.array([theta])
    tau = actuator.to_tau(sys, act, q)
    np.testing.assert_array_almost_equal(tau, jp.array([0]), 5)

    # put the pendulum into the horizontal position with the positional actuator
    # we know alpha = -2 theta/t^2, and I * alpha = (act-bias) * gear, and we
    # then solve for act. Semi-implicit Euler leaves off a factor of two.
    act = jp.array([-(theta * 0.5**2) / (0.01**2 * 10.0) + theta])
    q2, _ = _actuator_step(sys, q, qd, dt=0.01, act=act)
    np.testing.assert_array_almost_equal(q2, jp.array([0]), 1)

  def test_three_link_pendulum(self):
    """Tests a three link pendulum with a motor actuator."""
    sys = test_utils.load_fixture('triple_pendulum_motor.xml')
    mj_model = test_utils.load_fixture_mujoco('triple_pendulum_motor.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)
    theta = jp.pi / 2.0  # pendulum is vertical
    q = jp.array([theta, 0.0, 0.0])

    # test that with zero action the triple pendulum does not move
    act = jp.array([0] * 3)
    q1, qd1 = _actuator_step(sys, q, qd, dt=0.01, act=act)
    np.testing.assert_array_almost_equal(q1, q, 2)
    np.testing.assert_array_almost_equal(qd1, jp.zeros_like(qd1), 2)

    # test that each torque results in directionally accurate q and qd
    act = 1.0 / 150.0 * jp.array([0, -10, 10])
    q2, qd2 = _actuator_step(sys, q, qd, dt=0.001, act=act)
    self.assertAlmostEqual(q2[0], q[0], 3)
    self.assertLess(q2[1], q[1])
    self.assertGreater(q2[2], q[2])
    self.assertLess(qd2[1], -0.2)
    self.assertGreater(qd2[2], -0.2)

  def test_spherical_pendulum(self):
    """Tests a spherical pendulum with a position actuator."""
    sys = test_utils.load_fixture('single_spherical_pendulum_position.xml')
    mj_model = test_utils.load_fixture_mujoco(
        'single_spherical_pendulum_position.xml'
    )
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)

    act = jp.array([2.0, 3.0, 1.0])
    tau = actuator.to_tau(sys, act, q)
    expected_tau = jp.array([act[1], act[2], act[0]]) * 10.0
    np.testing.assert_array_almost_equal(tau, expected_tau)

    mj_data.ctrl = act
    mujoco.mj_step(mj_model, mj_data)
    expected_q, expected_qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)

    q2, qd2 = _actuator_step(sys, q, qd, dt=sys.dt, act=act)
    np.testing.assert_array_almost_equal(q2, expected_q)
    np.testing.assert_array_almost_equal(qd2, expected_qd)


if __name__ == '__main__':
  absltest.main()
