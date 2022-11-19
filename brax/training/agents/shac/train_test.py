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

"""SHAC tests."""
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.shac import networks as shac_networks
from brax.training.agents.shac import train as shac
import jax


class SHACTest(parameterized.TestCase):
  """Tests for SHAC module."""


  def testTrain(self):
    """Test SHAC with a simple env."""
    _, _, metrics = shac.train(
        envs.get_environment('fast_differentiable'),
        num_timesteps=2**15,
        episode_length=128,
        num_envs=64,
        actor_learning_rate=1.5e-2,
        critic_learning_rate=1e-3,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=10,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=1,
        normalize_observations=True,
        seed=2,
        reward_scaling=10)
    self.assertGreater(metrics['eval/episode_reward'], 135)

  @parameterized.parameters(True, False)
  def testNetworkEncoding(self, normalize_observations):
    env = envs.get_environment('fast')
    original_inference, params, _ = shac.train(
        env,
        num_timesteps=128,
        episode_length=128,
        num_envs=128,
        normalize_observations=normalize_observations)
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    shac_network = shac_networks.make_shac_networks(env.observation_size,
                                                 env.action_size, normalize_fn)
    inference = shac_networks.make_inference_fn(shac_network)
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
