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

"""Analytic policy gradient tests."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import apg


class APGTest(parameterized.TestCase):
  """Tests for APG module."""

  def testTraining(self):
    _, _, metrics = apg.train(
        environment_fn=envs.create_fn('reacherangle'),
        episode_length=100,
        action_repeat=4,
        num_envs=16,
        learning_rate=3e-3,
        normalize_observations=True,
        log_frequency=200,
        truncation_length=10,
    )
    self.assertGreater(metrics['eval/episode_reward'], -2)


if __name__ == '__main__':
  absltest.main()
