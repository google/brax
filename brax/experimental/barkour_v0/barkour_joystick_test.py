# Copyright 2023 DeepMind Technologies Limited.
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

"""Tests that the barkour joystick env runs."""

from absl.testing import absltest
from brax.experimental.barkour_v0 import barkour_joystick
import jax
import jax.numpy as jp


class BarkourJoystickTest(absltest.TestCase):

  def test_env_runs(self):
    env = barkour_joystick.Barkourv0()
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    _ = jax.jit(env.step)(state, jp.zeros(env.sys.act_size()))


if __name__ == "__main__":
  absltest.main()
