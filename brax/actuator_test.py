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
"""Tests for actuators."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import actuator
from brax import test_utils
from brax.generalized import pipeline as g_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline
import jax
from jax import numpy as jp
import mujoco
import numpy as np

assert_almost_equal = np.testing.assert_almost_equal


def _actuator_step(pipeline, sys, q, qd, act, dt, n):
  sys = sys.tree_replace({'opt.timestep': dt})
  def f(state, _):
    return jax.jit(pipeline.step)(sys, state, act), None

  state = jax.lax.scan(f, pipeline.init(sys, q, qd), (), n)[0]
  return state.q, state.qd


class ActuatorTest(parameterized.TestCase):

  @parameterized.parameters(
      (g_pipeline, 0.01, 100, 5),
      (s_pipeline, 0.001, 1000, 3),
      (p_pipeline, 0.001, 1000, 3),
  )
  def test_motor(self, pipeline, dt, n, decimal):
    """Tests a single pendulum with motor actuator."""
    sys = test_utils.load_fixture('single_pendulum_motor.xml')
    mj_model = test_utils.load_fixture_mujoco('single_pendulum_motor.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)

    tau = jp.array([0.5 * 9.81])  # -mgl sin(theta)
    act = jp.array([1.0 / 150.0 * 0.5 * 9.81])
    tau2 = actuator.to_tau(sys, act, q, qd)
    np.testing.assert_array_almost_equal(tau, tau2, 5)

    q2, qd2 = _actuator_step(pipeline, sys, q, qd, act=act, dt=dt, n=n)
    np.testing.assert_array_almost_equal(q2, jp.array([0]), decimal=decimal)
    np.testing.assert_array_almost_equal(qd2, jp.array([0]), decimal=decimal)

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
    tau = actuator.to_tau(sys, act, q, qd)
    np.testing.assert_array_almost_equal(tau, jp.array([0]), 5)

    # put the pendulum into the horizontal position with the positional actuator
    # we know alpha = -2 theta/t^2, and I * alpha = (act-bias) * gear, and we
    # then solve for act. Semi-implicit Euler leaves off a factor of two.
    act = jp.array([-(theta * 0.5**2) / (0.01**2 * 10.0) + theta])
    q2, _ = _actuator_step(g_pipeline, sys, q, qd, act=act, dt=0.01, n=1)
    np.testing.assert_array_almost_equal(q2, jp.array([0]), 1)

  def test_velocity(self):
    """Tests a single pendulum with velocity actuator."""
    sys = test_utils.load_fixture('single_pendulum_velocity.xml')
    mj_model = test_utils.load_fixture_mujoco('single_pendulum_velocity.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)
    theta = jp.pi / 2.0  # pendulum is vertical
    q = jp.array([theta])

    act = jp.array([0])
    tau = actuator.to_tau(sys, act, q, qd)
    np.testing.assert_array_almost_equal(tau, jp.array([0]), 5)

    # set the act to rotate at 1/s
    act = jp.array([1])
    _, qd = _actuator_step(g_pipeline, sys, q, qd, act=act, dt=0.001, n=200)
    np.testing.assert_array_almost_equal(qd, jp.array([1]), 3)

  def test_force_limitted(self):
    """Tests that forcerange limits work on actuators."""
    sys = test_utils.load_fixture('single_pendulum_position_frclimit.xml')
    mj_model = test_utils.load_fixture_mujoco(
        'single_pendulum_position_frclimit.xml'
    )
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)

    for act, frclimit in [(1000, 3.1), (-1000, -2.5)]:
      act = jp.array([act])
      tau = actuator.to_tau(sys, act, q, qd)
      # test that tau matches frclimit * 10, since gear=10
      self.assertEqual(tau[0], frclimit * 10)
      # test that tau matches MJ qfrc_actuator
      mj_data.ctrl = act
      mujoco.mj_step(mj_model, mj_data)
      self.assertEqual(tau[0], mj_data.qfrc_actuator)

  @parameterized.parameters((g_pipeline,), (s_pipeline,), (p_pipeline,))
  def test_three_link_pendulum(self, pipeline):
    """Tests a three link pendulum with a motor actuator."""
    sys = test_utils.load_fixture('triple_pendulum_motor.xml')
    mj_model = test_utils.load_fixture_mujoco('triple_pendulum_motor.xml')
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.array(mj_data.qpos), jp.array(mj_data.qvel)
    theta = jp.pi / 2.0  # pendulum is vertical
    q = jp.array([theta, 0.0, 0.0])

    # test that with zero action the triple pendulum does not move
    act = jp.array([0] * 3)
    q1, qd1 = _actuator_step(pipeline, sys, q, qd, act=act, dt=0.01, n=1)
    np.testing.assert_array_almost_equal(q1, q, 2)
    np.testing.assert_array_almost_equal(qd1, jp.zeros_like(qd1), 2)

    # now test that each torque results in directionally accurate q and qd
    act = 1.0 / 150.0 * jp.array([0, -10, 10])
    q2, qd2 = _actuator_step(pipeline, sys, q, qd, act=act, dt=1e-3, n=1)
    self.assertAlmostEqual(q2[0], q[0], 2)
    self.assertLess(q2[1], q[1])
    self.assertGreater(q2[2], q[2])
    self.assertLess(qd2[1], -0.2)
    self.assertGreater(qd2[2], -0.2)

  @parameterized.parameters(
      ('single_spherical_pendulum_motor.xml',), 'single_pendulum_motor.xml'
  )
  def test_spherical_pendulum_mj_generalized(self, config):
    """Tests a spherical pendulum with a motor actuator against mujoco."""
    # Note that positional actuators between mujoco and brax are not equivalent
    # so we only test motor actuators here.
    sys = test_utils.load_fixture(config)
    mj_model = test_utils.load_fixture_mujoco(config)
    mj_data = mujoco.MjData(mj_model)
    q, qd = jp.asarray(mj_data.qpos), jp.asarray(mj_data.qvel)

    act = jp.array([0.1, 0.2, 0.3])[: sys.act_size()]
    mj_data.ctrl = act
    for _ in range(1000):
      mujoco.mj_step(mj_model, mj_data)
    mq, mqd = jp.asarray(mj_data.qpos), jp.asarray(mj_data.qvel)

    gq, gqd = _actuator_step(
        g_pipeline, sys, q, qd, act=act, dt=sys.opt.timestep, n=1000
    )
    np.testing.assert_array_almost_equal(gq, mq, 3)
    np.testing.assert_array_almost_equal(gqd, mqd, 3)

  @parameterized.parameters(
      'single_pendulum_position.xml',
      'single_pendulum_motor.xml',
      'single_spherical_pendulum_position.xml',
  )
  def test_single_pendulum_spring_positional(self, config):
    sys = test_utils.load_fixture(config)
    act = jp.array([0.05, 0.1, 0.15])[: sys.act_size()]

    q, qd = sys.init_q, jp.zeros(sys.qd_size())

    sq, sqd = _actuator_step(
        s_pipeline, sys, q, qd, act=act, dt=sys.opt.timestep, n=500
    )
    pq, pqd = _actuator_step(
        p_pipeline, sys, q, qd, act=act, dt=sys.opt.timestep, n=500
    )
    np.testing.assert_array_almost_equal(sq, pq, 2)
    np.testing.assert_array_almost_equal(sqd, pqd, 2)


if __name__ == '__main__':
  absltest.main()
