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

"""Tests the dm env wrapper."""

from absl.testing import absltest
from brax import envs
from brax.envs.wrappers import dm_env
import numpy as np


class DmEnvTest(absltest.TestCase):

  def test_action_space(self):
    """Tests the action space of the DmEnvWrapper."""
    base_env = envs.create('pusher')
    env = dm_env.DmEnvWrapper(base_env)
    np.testing.assert_array_equal(
        env.action_spec().minimum, base_env.sys.actuator.ctrl_range[:, 0])
    np.testing.assert_array_equal(
        env.action_spec().maximum, base_env.sys.actuator.ctrl_range[:, 1])


if __name__ == '__main__':
  absltest.main()
