# Copyright 2026 The Brax Authors.
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

"""Tests for training wrappers."""

import functools

from absl.testing import absltest
from brax import envs
from brax.envs.wrappers import training
import jax
import jax.numpy as jp
import numpy as np


class TrainingTest(absltest.TestCase):

  def test_domain_randomization_wrapper(self):
    def rand(sys, rng):
      @jax.vmap
      def get_offset(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

      sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
      in_axes = jax.tree.map(lambda x: None, sys)
      in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
      return sys_v, in_axes

    env = envs.create('ant')
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 256)
    env = training.wrap(
        env,
        episode_length=200,
        randomization_fn=functools.partial(rand, rng=rng),
    )

    # set the same key across the batch for env.reset so that only the
    # randomization wrapper creates variability in the env.step
    key = jp.zeros((256, 2), dtype=jp.uint32)
    state = jax.jit(env.reset)(key)
    self.assertEqual(state.pipeline_state.q[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.pipeline_state.q[:, 0]).shape[0], 1)

    # test that the DomainRandomizationWrapper creates variability in env.step
    state = jax.jit(env.step)(state, jp.zeros((256, env.sys.act_size())))
    self.assertEqual(state.pipeline_state.q[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.pipeline_state.q[:, 0]).shape[0], 256)

  def test_history_wrapper(self):
    base = envs.get_environment('ant')
    obs_size, act_size, length = base.observation_size, base.action_size, 3
    env = training.HistoryWrapper(base, length)
    frame = obs_size + act_size
    self.assertEqual(env.observation_size, length * frame)

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    hist = state.obs.reshape(length, frame)
    np.testing.assert_array_equal(hist[:-1], 0)
    np.testing.assert_array_equal(hist[-1, obs_size:], 0)
    base_state = jax.jit(base.reset)(jax.random.PRNGKey(0))
    np.testing.assert_allclose(hist[-1, :obs_size], base_state.obs)

    step = jax.jit(env.step)
    actions = [jp.full((act_size,), 0.1 * (i + 1)) for i in range(4)]
    observed = []
    for a in actions:
      state = step(state, a)
      observed.append(state.obs.reshape(length, frame))
    # Oldest entries are dropped; the newest frame is (obs, action just taken).
    np.testing.assert_allclose(observed[-1][-1, obs_size:], actions[-1])
    np.testing.assert_allclose(observed[-1][-2, obs_size:], actions[-2])
    np.testing.assert_allclose(observed[-1][-3, obs_size:], actions[-3])
    np.testing.assert_allclose(observed[0][-2], hist[-1])
    self.assertEqual(state.obs.shape, (length * frame,))

  def test_history_wrapper_batched_autoreset(self):
    length = 2
    base = envs.get_environment('ant')
    env = training.wrap(
        training.HistoryWrapper(base, length), episode_length=3
    )
    frame = base.observation_size + base.action_size
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    self.assertEqual(state.obs.shape, (4, length * frame))
    step = jax.jit(env.step)
    for _ in range(3):
      state = step(state, jp.ones((4, base.action_size)))
    # Episode ended on the third step: observation is reset to its first value,
    # with an empty history.
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_array_equal(state.obs[:, :frame], 0)
    first_obs = state.obs
    state = step(state, jp.ones((4, base.action_size)))
    # The next episode starts from the reset history: its newest frame moves
    # up one slot and nothing from the previous episode is carried over.
    np.testing.assert_allclose(state.obs[:, :frame], first_obs[:, frame:])
    np.testing.assert_array_equal(state.obs[:, -base.action_size:], 1)

  def test_history_wrapper_invalid_context_length(self):
    with self.assertRaises(ValueError):
      training.HistoryWrapper(envs.get_environment('ant'), 0)


if __name__ == '__main__':
  absltest.main()
