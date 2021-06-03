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

"""Training test for Evolution Strategy training."""

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
from brax.training import es


def run_test(seed, num_timesteps=400000000):
  env_name = 'ant'
  log_frequency = 10
  center_fitness = True
  fitness_shaping = 2
  population_size = 512
  episode_length = 1000
  fitness_episode_length = 1000
  action_repeat = 1
  learning_rate = 0.01
  normalize_observations = True
  l2coeff = 0
  perturbation_std = 0.04
  env_fn = envs.create_fn(env_name)

  inference, params, metrics = es.train(
      environment_fn=env_fn,
      num_timesteps=num_timesteps,
      episode_length=episode_length,
      fitness_episode_length=fitness_episode_length,
      action_repeat=action_repeat,
      learning_rate=learning_rate,
      normalize_observations=normalize_observations,
      seed=seed,
      population_size=population_size,
      log_frequency=log_frequency,
      l2coeff=l2coeff,
      center_fitness=center_fitness,
      fitness_shaping=fitness_shaping,
      perturbation_std=perturbation_std)

  return inference, params, metrics, env_fn


class TrainingTest(parameterized.TestCase):

  def testTraining(self):
    _, _, metrics, _ = run_test(seed=0)
    logging.info(metrics)
    reward = metrics['eval/episode_reward']
    self.assertGreater(reward, 4500 * .995)

  def testModelEncoding(self):
    _, params, _, env_fn = run_test(seed=0, num_timesteps=1000)
    env = env_fn()
    base_params, inference = es.make_params_and_inference_fn(
        env.observation_size, env.action_size, True)
    byte_encoding = serialization.to_bytes(params)
    decoded_params = serialization.from_bytes(base_params, byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params, state.obs, state.rng)
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
