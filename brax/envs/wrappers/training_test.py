# Copyright 2025 The Brax Authors.
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

  def test_autoreset_termination(self):
    for env_id in ["ant", "halfcheetah"]:
        with self.subTest(env_id=env_id):
            self._run_termination(env_id)

  def _run_termination(self, env_id):
    env = envs.create(env_id)
    key = jax.random.PRNGKey(42)
    max_steps_in_episode = env.episode_length

    state = jax.jit(env.reset)(key)
    action = jp.zeros(env.sys.act_size())

    env_step_fn =  jax.jit(env.step)

    def step_fn(state, _):
        next_state = env_step_fn(state, action)
        return next_state, (next_state.obs, next_state.done, next_state.info)

    _, (observations, dones, infos) = jax.lax.scan(
        f=step_fn, init=state, xs=None, length=max_steps_in_episode + 1
    )
    
    observations_step = infos["obs_st"]
    # Should have at least finished once
    assert sum(dones) >= 1
    for i, (obs, done, obs_st) in enumerate(zip(observations, dones, observations_step)):
      if done:
        # Ensure we stored the last obs from finished episode, \\
        # which differs from first obs of new episode
        assert not jp.array_equal(obs_st, obs)
      else:
         assert jp.array_equal(obs_st, obs)

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


if __name__ == '__main__':
  absltest.main()
