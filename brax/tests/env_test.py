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

"""Tests for brax.envs."""

import logging
import time

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax import jumpy as jp
import jax

_EARLY_TERMINATION = ('ant', 'humanoid')
_EXPECTED_SPS = {'ant': 1000, 'fetch': 1000}


class EnvTest(parameterized.TestCase):

  @parameterized.parameters(_EXPECTED_SPS.items())
  def testSpeed(self, env_name, expected_sps):
    batch_size = 128
    episode_length = 1000
    auto_reset = (env_name not in _EARLY_TERMINATION)

    env = envs.create(
        env_name,
        episode_length=episode_length,
        auto_reset=auto_reset,
        batch_size=batch_size)
    zero_action = jp.zeros((batch_size, env.action_size))

    @jax.jit
    def run_env(state):

      def step(carry, _):
        state, = carry
        state = env.step(state, zero_action)
        return (state,), ()

      (state,), _ = jax.lax.scan(step, (state,), (), length=episode_length)
      return state

    # warmup
    rng = jp.random_prngkey(0)
    state = jax.jit(env.reset)(rng)
    state = run_env(state)
    state.done.block_until_ready()

    sps = []
    for seed in range(5):
      rng = jp.random_prngkey(seed)
      state = jax.jit(env.reset)(rng)
      jax.device_put(state)
      t = time.time()
      state = run_env(state)
      state.done.block_until_ready()
      sps.append((batch_size * episode_length) / (time.time() - t))
      self.assertTrue(jp.all(state.done))

    mean_sps = jp.mean(jp.array(sps))
    logging.info('%s SPS %s %s', env_name, mean_sps, sps)
    self.assertGreater(mean_sps, expected_sps * 0.99)


if __name__ == '__main__':
  absltest.main()
