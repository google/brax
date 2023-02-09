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
"""Tests for spring physics pipeline."""

from absl.testing import absltest
from brax.v2 import kinematics
from brax.v2 import test_utils
from brax.v2.generalized import pipeline as g_pipeline
from brax.v2.positional import pipeline
import jax
from jax import numpy as jp
import numpy as np


class PipelineTest(absltest.TestCase):

  def test_pendulum(self):
    sys = test_utils.load_fixture('double_pendulum.xml')
    sys = sys.replace(dt=0.0005)

    state = pipeline.init(sys, sys.init_q, jp.zeros(sys.qd_size()))
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(2_000):
      state = j_spring_step(sys, state, jp.zeros(sys.qd_size()))
    x = state.x

    # compare against generalized step
    q, qd = jp.zeros(sys.q_size()), jp.zeros(sys.qd_size())
    state = g_pipeline.init(sys, q, qd)
    j_g_step = jax.jit(g_pipeline.step)
    j_forward = jax.jit(kinematics.forward)
    for _ in range(2_000):
      state = j_g_step(sys, state, jp.zeros(sys.qd_size()))
    x_g, _ = j_forward(sys, state.q, state.qd)

    # trajectories should be close after .1 second of simulation
    self.assertLess(jp.linalg.norm(x_g.pos - x.pos), 2e-2)

  def test_sliding_capsule(self):
    sys = test_utils.load_fixture('capsule.xml')
    sys = sys.replace(dt=0.001)
    sys = sys.replace(collide_scale=0.25)

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(5.0)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.qd_size()))
    x, xd = state.x, state.xd

    # capsule slides to a stop
    self.assertAlmostEqual(x.pos[0, 2], 0.25, delta=1e-2)
    np.testing.assert_allclose(
        x.rot, jp.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-3
    )
    np.testing.assert_allclose(xd.vel, jp.zeros_like(xd.vel), atol=1e-2)
    np.testing.assert_allclose(xd.ang, jp.zeros_like(xd.ang), atol=1e-2)


if __name__ == '__main__':
  absltest.main()
