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
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.apg import train as apg
import jax


class APGTest(parameterized.TestCase):
  """Tests for APG module."""

  def testTrain(self):
    """Test APG with a simple env."""
    _, _, metrics = apg.train(
        envs.get_environment('fast_differentiable'),
        episode_length=128,
        num_envs=64,
        num_evals=200,
        learning_rate=3e-3,
        normalize_observations=True,
    )
    self.assertGreater(metrics['eval/episode_reward'], 135)

  @parameterized.parameters(True, False)
  def testNetworkEncoding(self, normalize_observations):
    env = envs.get_environment('fast')
    original_inference, params, _ = apg.train(
        envs.get_environment('fast'),
        episode_length=100,
        action_repeat=4,
        num_envs=16,
        learning_rate=3e-3,
        normalize_observations=normalize_observations,
        num_evals=200,
        truncation_length=10)
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    apg_network = apg_networks.make_apg_networks(env.observation_size,
                                                 env.action_size, normalize_fn)
    inference = apg_networks.make_inference_fn(apg_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    original_action = original_inference(decoded_params)(
        state.obs, jax.random.PRNGKey(0))[0]
    action = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))[0]
    self.assertSequenceEqual(original_action, action)
    env.step(state, action)


if __name__ == '__main__':
  absltest.main()
