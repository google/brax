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

"""PPO tests."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import ppo
from flax import serialization
import jax


class PPOTest(parameterized.TestCase):
  """Tests for PPO module."""

  def testTrain(self):
    """Test PPO with a simple env."""
    _, _, metrics = ppo.train(
        envs.create_fn('fast'),
        num_timesteps=32768,
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_update_epochs=4,
        log_frequency=5,
        normalize_observations=True,
        reward_scaling=10)
    self.assertGreater(metrics['eval/episode_reward'], 100 * 0.995)

  @parameterized.parameters(True, False)
  def testModelEncoding(self, normalize_observations):
    env_fn = envs.create_fn('fast')
    _, params, _ = ppo.train(
        env_fn, num_timesteps=128, episode_length=128, num_envs=128)
    env = env_fn()
    base_params, inference = ppo.make_params_and_inference_fn(
        env.observation_size, env.action_size, normalize_observations)
    byte_encoding = serialization.to_bytes(params)
    decoded_params = serialization.from_bytes(base_params, byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    action = inference(decoded_params, state.obs, jax.random.PRNGKey(0))
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
