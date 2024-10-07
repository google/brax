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
# pylint:disable=g-importing-member
"""Tests for spring physics pipeline."""

from absl.testing import absltest
from brax import test_utils
from brax.base import Contact
from brax.mjx import pipeline
import jax
from jax import numpy as jp
import mujoco
import numpy as np


class PipelineTest(absltest.TestCase):

  def test_pendulum(self):
    model = test_utils.load_fixture('double_pendulum.xml')

    state = pipeline.init(model, model.init_q, jp.zeros(model.qd_size()))

    self.assertIsInstance(state.contact, Contact)

    step_fn = jax.jit(pipeline.step)
    for _ in range(20):
      state = step_fn(model, state, jp.zeros(model.act_size()))

    # compare against mujoco
    model = test_utils.load_fixture_mujoco('double_pendulum.xml')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 20)

    np.testing.assert_almost_equal(data.qpos, state.q, decimal=4)
    np.testing.assert_almost_equal(data.qvel, state.qd, decimal=3)
    np.testing.assert_almost_equal(data.xpos[1:], state.x.pos, decimal=4)

  def test_pipeline_init_with_ctrl(self):
    model = test_utils.load_fixture('single_spherical_pendulum_position.xml')
    ctrl = jp.array([0.3, 0.5, 0.4])
    state = pipeline.init(
        model,
        model.init_q,
        jp.zeros(model.qd_size()),
        ctrl=ctrl,
    )
    np.testing.assert_array_almost_equal(state.ctrl, ctrl)


if __name__ == '__main__':
  absltest.main()
