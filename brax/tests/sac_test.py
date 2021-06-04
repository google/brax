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

"""SAC training test."""

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
from brax.training import sac


def run_test(seed, total_env_steps=1572864, normalize_observations=True):
  agent_name = 'ant'
  eval_frequency = 262144
  reward_scaling = 10
  episode_length = 1000
  action_repeat = 1
  discounting = 0.95
  learning_rate = 6e-4
  num_envs = 128
  batch_size = 512
  min_replay_size = 8192
  grad_updates_per_step = 0.125
  max_devices_per_host = 1
  env_fn = envs.create_fn(agent_name)

  inference, params, metrics = sac.train(
      environment_fn=env_fn,
      action_repeat=action_repeat,
      num_envs=num_envs,
      normalize_observations=normalize_observations,
      num_timesteps=total_env_steps,
      log_frequency=eval_frequency,
      min_replay_size=min_replay_size,
      batch_size=batch_size,
      learning_rate=learning_rate,
      discounting=discounting,
      grad_updates_per_step=grad_updates_per_step,
      seed=seed,
      reward_scaling=reward_scaling,
      max_devices_per_host=max_devices_per_host,
      episode_length=episode_length,
      )
  return inference, params, metrics, env_fn


class TrainingTest(parameterized.TestCase):

  def testTraining(self):
    _, _, metrics, _ = run_test(seed=0)
    logging.info(metrics)
    reward = metrics['eval/episode_reward']
    self.assertGreater(reward, 2500)

  @parameterized.parameters(True, False)
  def testModelEncoding(self, normalize_observations=True):
    _, params, _, env_fn = run_test(
        seed=0,
        total_env_steps=1000,
        normalize_observations=normalize_observations)
    env = env_fn()
    base_params, inference = sac.make_params_and_inference_fn(
        env.observation_size, env.action_size, normalize_observations)
    byte_encoding = serialization.to_bytes(params)
    decoded_params = serialization.from_bytes(base_params, byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params, state.obs, state.rng)
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
