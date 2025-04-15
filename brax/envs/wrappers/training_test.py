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

"""Tests for training wrappers."""

import functools

from absl.testing import absltest
from brax import envs
from brax.envs.wrappers import training
import jax
import jax.numpy as jp
import numpy as np

class DummyState:
  """A minimal dummy state that mimics brax.envs.base.State."""
  def __init__(self, obs, reward, done, metrics=None, info=None, pipeline_state=None):
    self.obs = obs
    self.reward = reward
    self.done = done
    self.metrics = metrics if metrics is not None else {}
    self.info = info if info is not None else {}
    self.pipeline_state = pipeline_state

  def replace(self, **kwargs):
    return DummyState(
        obs=kwargs.get('obs', self.obs),
        reward=kwargs.get('reward', self.reward),
        done=kwargs.get('done', self.done),
        metrics=kwargs.get('metrics', self.metrics),
        info=kwargs.get('info', self.info),
        pipeline_state=kwargs.get('pipeline_state', self.pipeline_state),
    )

class DummyEnv:
  """A dummy environment for testing wrappers that always returns constant values."""
  def __init__(self, constant_obs, constant_reward, act_size=2):
    self.observation_size = constant_obs.shape
    self._act_size = act_size
    self.constant_obs = constant_obs
    self.constant_reward = constant_reward
    self.sys = type('DummySys', (), {})()

  def reset(self, rng):
    return DummyState(
        obs=self.constant_obs,
        reward=self.constant_reward,
        done=jp.array(False),
        metrics={'reward': self.constant_reward},
        info={},
        pipeline_state=jp.zeros((1,))
    )

  def step(self, state, action):
    new_info = dict(state.info)
    new_info['action_received'] = action
    return state.replace(
        obs=self.constant_obs,
        reward=self.constant_reward,
        done=jp.array(False),
        metrics={'reward': self.constant_reward},
        info=new_info,
        pipeline_state=state.pipeline_state
    )

  def unwrapped(self):
    return self


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


  def test_running_mean_std(self):
    rms = training.RunningMeanStd(shape=(3,), epsilon=1e-8)
    x = jp.array([[1.0, 2.0, 3.0],
                  [3.0, 4.0, 5.0]])
    rms.update(x)
    expected_mean = jp.array([2.0, 3.0, 4.0])
    expected_var = jp.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(rms.mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(rms.var, expected_var, rtol=1e-5)

  def test_clip_vec_action(self):
    constant_obs = jp.array([0.0, 0.0, 0.0])
    constant_reward = 0.0
    dummy_env = DummyEnv(constant_obs, constant_reward, act_size=2)

    clip_wrapper = training.ClipVecAction(dummy_env, low=-1.0, high=1.0)
    rng = jax.random.PRNGKey(42)
    state = clip_wrapper.reset(rng)

    action = jp.array([[-2.0, 2.0]])
    new_state = clip_wrapper.step(state, action)
    clipped = new_state.info.get('action_received')
    np.testing.assert_allclose(clipped, jp.array([[-1.0, 1.0]]), rtol=1e-5)

  def test_normalize_vec_observation(self):
    constant_obs = jp.array([1.0, 1.0, 1.0])
    constant_reward = 0.0
    dummy_env = DummyEnv(constant_obs, constant_reward)
    norm_wrapper = training.NormalizeVecObservation(dummy_env)

    rng = jax.random.PRNGKey(0)
    state = norm_wrapper.reset(rng)
    np.testing.assert_allclose(state.obs, jp.zeros_like(constant_obs), rtol=1e-5)

  def test_normalize_vec_reward(self):
    constant_obs = jp.array([0.0, 0.0, 0.0])
    constant_reward = 1.0
    dummy_env = DummyEnv(constant_obs, constant_reward)
    norm_reward_wrapper = training.NormalizeVecReward(dummy_env, gamma=0.99, epsilon=1e-8)
    rng = jax.random.PRNGKey(0)
    state = norm_reward_wrapper.reset(rng)
    state = norm_reward_wrapper.step(state, jp.array([0.0] * dummy_env._act_size))
    expected_norm = 1.0 / jp.sqrt(1e-8)
    np.testing.assert_allclose(state.reward, expected_norm, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
