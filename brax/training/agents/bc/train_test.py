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

"""BC tests."""

import functools
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.bc import train as bc
import jax


class BCTest(parameterized.TestCase):
  """Tests for BC module."""

  def _create_teacher_policy(self, num_envs):
    """Helper to create a dummy teacher policy."""

    def teacher_policy(obs, rng):
      return (
          jax.numpy.ones((num_envs, 1)),
          {'loc': 1 * jax.numpy.ones((num_envs, 1))},
      )

    return teacher_policy

  def testTrain(self):
    """Test BC with a simple env."""
    teacher_policy = self._create_teacher_policy(num_envs=50)
    fast = envs.get_environment('fast', obs_mode='dict_state')
    fast = envs.training.wrap(fast, episode_length=128, action_repeat=1)
    network_factory = functools.partial(
        bc_networks.make_bc_networks,
        policy_hidden_layer_sizes=(32,),
        policy_obs_key='state',
        vision=False,
    )
    _, _, metrics = bc.train(
        env=fast,
        teacher_inference_fn=teacher_policy,
        demo_length=6,
        num_envs=50,
        eval_length=128,
        num_evals=3,
        num_eval_envs=10,
        dagger_steps=20,
        batch_size=50,
        learning_rate=3e-3,
        epochs=4,
        network_factory=network_factory,
        reset=False,
    )
    self.assertGreater(metrics['eval/episode_reward'], 135)

  def testNetworkEncoding(self):
    """Test network encoding and decoding."""
    teacher_policy = self._create_teacher_policy(num_envs=50)
    fast = envs.get_environment('fast', obs_mode='dict_state')
    w_fast = envs.training.wrap(fast, episode_length=128, action_repeat=1)

    # Train a BC agent to get parameters
    network_factory = functools.partial(
        bc_networks.make_bc_networks,
        policy_hidden_layer_sizes=(32,),
        policy_obs_key='state',
        vision=False,
    )

    original_inference, params, _ = bc.train(
        env=w_fast,
        teacher_inference_fn=teacher_policy,
        demo_length=6,
        num_envs=50,
        eval_length=128,
        num_evals=1,
        dagger_steps=2,
        batch_size=50,
        learning_rate=3e-3,
        epochs=2,
        tanh_squash=True,
        network_factory=network_factory,
        reset=False,
    )

    # Create a new inference function with the same network structure
    normalize_fn = running_statistics.normalize

    bc_network = network_factory(
        fast.observation_size, fast.action_size, normalize_fn
    )
    make_inference = bc_networks.make_inference_fn(bc_network)

    # Test serialization and deserialization
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action with both the original and the reconstructed parameters
    state = fast.reset(jax.random.PRNGKey(0))
    original_action = original_inference(decoded_params)(
        state.obs, jax.random.PRNGKey(0)
    )[0]
    action = make_inference(decoded_params, tanh_squash=True)(
        state.obs, jax.random.PRNGKey(0)
    )[0]

    # Verify that the actions are the same
    self.assertSequenceEqual(original_action, action)
    fast.step(state, action)

  def testPixelsBC(self):
    """Test BC with pixel observations."""
    teacher_policy = self._create_teacher_policy(num_envs=50)
    fast = envs.get_environment('fast', obs_mode='dict_latent_state')
    fast = envs.training.wrap(fast, episode_length=128, action_repeat=1)
    network_factory = functools.partial(
        bc_networks.make_bc_networks,
        policy_hidden_layer_sizes=(32,),
        policy_obs_key='state',
        latent_vision=True,
    )
    _, _, metrics = bc.train(
        env=fast,
        teacher_inference_fn=teacher_policy,
        demo_length=6,
        num_envs=50,
        eval_length=128,
        num_evals=3,
        num_eval_envs=10,
        dagger_steps=20,
        batch_size=50,
        learning_rate=3e-3,
        epochs=4,
        network_factory=network_factory,
        reset=False,
        augment_pixels=True,
    )
    self.assertGreater(metrics['eval/episode_reward'], 135)


if __name__ == '__main__':
  absltest.main()
