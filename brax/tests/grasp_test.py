# Copyright 2021 The Brax Authors.
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

"""Tests for brax.envs.grasp."""

from absl.testing import absltest
import jax
import jax.numpy as np
from brax import envs


class GraspTest(absltest.TestCase):

  def testGrasp(self):
    env = envs.create('grasp')
    grasp_action = np.array([
        -.4, -.35, -0., -1,  # gripper arm 1
        .4, .35, 0., 1,  # gripper arm 2
        .4, .35, 0., 1,  # gripper arm 3
        .4, .35, 0., 1,  # gripper arm 4
        0., 0., -.9  # position action
    ])

    jit_env_step = jax.jit(env.step)
    state = env.reset(jax.random.PRNGKey(0))

    for _ in range(500):
      state = jit_env_step(state, grasp_action)

    self.assertGreater(state.qp.pos[1, 2], 1.47)  # ball lifted off ground
    self.assertLess(state.qp.ang[1, 2], .01)  # ball not rolling


if __name__ == '__main__':
  absltest.main()
