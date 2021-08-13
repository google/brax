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

"""Training tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

# Dependency imports

from absl.testing import absltest
from absl.testing import parameterized
from flax import serialization
import jax
from brax import envs
from brax.training import ppo


def run_test(seed, total_env_steps=50000000, normalize_observations=True):
  env_name = 'ant'
  eval_frequency = 10
  reward_scaling = 10
  episode_length = 1000
  action_repeat = 1
  unroll_length = 5
  num_minibatches = 32
  num_update_epochs = 4
  discounting = 0.95
  learning_rate = 2.25e-4
  entropy_cost = 1e-2
  num_envs = 2048
  batch_size = 1024
  max_devices_per_host = 8
  env_fn = envs.create_fn(env_name)

  inference, params, metrics = ppo.train(
      environment_fn=env_fn,
      action_repeat=action_repeat,
      num_envs=num_envs,
      max_devices_per_host=max_devices_per_host,
      normalize_observations=normalize_observations,
      num_timesteps=total_env_steps,
      log_frequency=eval_frequency,
      batch_size=batch_size,
      unroll_length=unroll_length,
      num_minibatches=num_minibatches,
      num_update_epochs=num_update_epochs,
      learning_rate=learning_rate,
      entropy_cost=entropy_cost,
      discounting=discounting,
      seed=seed,
      reward_scaling=reward_scaling,
      episode_length=episode_length)

  return inference, params, metrics, env_fn


class TrainingTest(parameterized.TestCase):

  def testTraining(self):
    _, _, metrics, _ = run_test(seed=0)
    logging.info(metrics)
    reward = metrics['eval/episode_reward']
    self.assertGreater(reward, 5300 * 0.995)

  @parameterized.parameters(True, False)
  def testModelEncoding(self, normalize_observations=True):
    _, params, _, env_fn = run_test(
        seed=0,
        total_env_steps=1000,
        normalize_observations=normalize_observations)
    env = env_fn()
    base_params, inference = ppo.make_params_and_inference_fn(
        env.observation_size, env.action_size, normalize_observations)
    byte_encoding = serialization.to_bytes(params)
    decoded_params = serialization.from_bytes(base_params, byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params, state.obs, state.rng)
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
