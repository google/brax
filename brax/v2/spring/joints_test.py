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

"""Tests for spring physics joints."""

from absl.testing import absltest
from absl.testing import parameterized
from brax.v2 import kinematics
from brax.v2 import test_utils
from brax.v2.spring import pipeline
import jax
from jax import numpy as jp


class JointTest(parameterized.TestCase):

  @parameterized.parameters(
      (2.0, 0.125, 0.0625), (5.0, 0.125, 0.03125), (1.0, 0.0625, 0.1)
  )
  def test_pendulum_period(self, mass, radius, vel):
    """A small spherical mass swings for approximately one period."""
    # config = text_format.Parse(JointTest._CONFIG, brax.Config())
    sys = test_utils.load_fixture('single_pendulum.xml')

    dist_to_anchor = 0.5
    inertia_cm = 2.0 / 5.0 * mass * radius**2.0
    inertia_about_anchor = mass * dist_to_anchor**2.0 + inertia_cm
    g = 9.81
    # formula for period of pendulum
    period = (
        2 * jp.pi * jp.sqrt(inertia_about_anchor / (mass * g * dist_to_anchor))
    )
    num_timesteps = 20_000
    sys = sys.replace(dt=period / num_timesteps)
    link = sys.link.replace(constraint_limit_stiffness=jp.array([0.0] * 1))
    link = link.replace(constraint_stiffness=jp.array([10_000.0] * 1))
    link = link.replace(constraint_ang_damping=jp.array([0.0] * 1))
    link = link.replace(constraint_damping=jp.array([0.0] * 1))
    sys = sys.replace(link=link)
    sys = sys.replace(ang_damping=0.0)
    sys = sys.replace(
        link=sys.link.replace(
            inertia=sys.link.inertia.replace(
                i=jp.array([0.4 * mass * radius**2 * jp.eye(3)] * 1),
                mass=jp.array([mass]),
            )
        )
    )

    # init with small initial velocity for small angle approx. validity
    state = pipeline.init(sys, sys.init_q, jp.zeros(sys.qd_size()))
    x, xd = kinematics.forward(
        sys,
        jp.array(
            [
                -jp.pi / 2.0,
            ]
        ),
        jp.array([vel]),
    )
    state = state.replace(x=x, xd=xd)

    j_spring_step = jax.jit(pipeline.step)
    for _ in range(num_timesteps):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))

    self.assertAlmostEqual(state.xd.ang[0, 0], vel, 2)  # returned to the origin


if __name__ == '__main__':
  absltest.main()
