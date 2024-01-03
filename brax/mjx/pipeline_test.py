# Copyright 2023 The Brax Authors.
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
from brax import test_utils
from brax.mjx import pipeline
import jax
from jax import numpy as jp
import mujoco
import numpy as np


class PipelineTest(absltest.TestCase):

  def test_pendulum(self):
    sys = test_utils.load_fixture('double_pendulum.xml')
    model = sys.get_mjx_model()

    state = pipeline.init(model, sys.init_q, jp.zeros(sys.qd_size()))
    step_fn = jax.jit(pipeline.step)
    for _ in range(20):
      state = step_fn(model, state, jp.zeros(sys.act_size()))

    # compare against mujoco
    model = test_utils.load_fixture_mujoco('double_pendulum.xml')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 20)

    np.testing.assert_almost_equal(data.qpos, state.q, decimal=4)
    np.testing.assert_almost_equal(data.qvel, state.qd, decimal=3)
    np.testing.assert_almost_equal(data.xpos[1:], state.x.pos, decimal=4)


if __name__ == '__main__':
  absltest.main()
