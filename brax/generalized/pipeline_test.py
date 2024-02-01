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
"""Tests for generalized pipeline."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import test_utils
from brax.generalized import pipeline
from brax.io import mjcf
import jax
from jax import numpy as jp
import numpy as np


class PipelineTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ant.xml',),
      ('triple_pendulum.xml',),
      ('humanoid.xml',),
      ('half_cheetah.xml',),
      ('swimmer.xml',),
  )
  def test_forward(self, xml_file):
    """Test pipeline step."""
    sys = test_utils.load_fixture(xml_file)
    # crank up solver iterations just to demonstrate close match to mujoco
    sys = sys.replace(solver_iterations=500)
    for mj_prev, mj_next in test_utils.sample_mujoco_states(xml_file):
      state = jax.jit(pipeline.init)(sys, mj_prev.qpos, mj_prev.qvel)
      state = jax.jit(pipeline.step)(sys, state, jp.zeros(sys.act_size()))

      np.testing.assert_allclose(state.q, mj_next.qpos, atol=0.002)
      np.testing.assert_allclose(state.qd, mj_next.qvel, atol=0.5)


class GradientTest(absltest.TestCase):
  """Tests that gradients are not NaN."""

  def test_grad(self):
    """Tests that gradients are not NaN."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" axis="1 0 0" damping="1"/>
          <joint type="slide" axis="0 1 0" damping="1"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    sys = mjcf.loads(xml)
    init_state = jax.jit(pipeline.init)(
        sys, sys.init_q, jp.zeros(sys.qd_size())
    )

    def fn(xd):
      qd = jp.zeros(sys.qd_size()).at[0].set(xd)
      state = init_state.replace(qd=qd)
      for _ in range(10):
        state = jax.jit(pipeline.step)(sys, state, None)
      return state.qd[0]

    grad = jax.grad(fn)(-1.0)
    self.assertFalse(np.isnan(grad))


if __name__ == '__main__':
  absltest.main()
