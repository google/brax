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

"""Training tests dor direct trajectory optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

# Dependency imports

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import dto


def run_test(seed):
  env_name = 'reacherangle'
  eval_frequency = 200
  episode_length = 100
  action_repeat = 4
  learning_rate = 3e-3
  num_envs = 1024
  normalize_observations = True
  env_fn = envs.create_fn(env_name)

  inference, params, metrics = dto.train(
      environment_fn=env_fn,
      log_frequency=eval_frequency,
      episode_length=episode_length,
      action_repeat=action_repeat,
      normalize_observations=normalize_observations,
      learning_rate=learning_rate,
      num_envs=num_envs,
      seed=seed)

  return inference, params, metrics, env_fn


class TrainingTest(parameterized.TestCase):

  def testTraining(self):
    _, _, metrics, _ = run_test(seed=0)
    logging.info(metrics)
    assert metrics['eval/episode_reward'] > -2


if __name__ == '__main__':
  absltest.main()
