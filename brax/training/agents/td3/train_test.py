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

"""TD3 tests."""

import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
import jax
import networks as td3_networks
import train as td3
import jax.numpy as jnp


class TD3Test(parameterized.TestCase):
    """Tests for TD3 module."""

    def testTrain(self):
        """Test TD3 with a simple env."""
        fast = envs.get_environment('fast')
        _, _, metrics = td3.train(
            fast,
            num_timesteps=2 ** 15,
            episode_length=128,
            num_envs=64,
            learning_rate=3e-4,
            discounting=0.99,
            batch_size=64,
            normalize_observations=True,
            reward_scaling=10,
            grad_updates_per_step=64,
            num_evals=3,
            seed=0)
        self.assertGreater(metrics['eval/episode_reward'], 140 * 0.995)
        self.assertEqual(fast.reset_count, 2)  # type: ignore
        # once for prefill, once for train, once for eval
        self.assertEqual(fast.step_count, 3)  # type: ignore

    @parameterized.parameters(True, False)
    def testNetworkEncoding(self, normalize_observations):
        env = envs.get_environment('fast')
        original_inference, params, _ = td3.train(
            env,
            num_timesteps=128,
            episode_length=128,
            num_envs=128,
            normalize_observations=normalize_observations)
        normalize_fn = lambda x, y: x
        if normalize_observations:
            normalize_fn = running_statistics.normalize
        td3_network = td3_networks.make_td3_networks(env.observation_size,
                                                     env.action_size, normalize_fn)
        inference = td3_networks.make_inference_fn(td3_network)
        byte_encoding = pickle.dumps(params)
        decoded_params = pickle.loads(byte_encoding)

        # Compute one action.
        state = env.reset(jax.random.PRNGKey(0))
        original_action = original_inference(decoded_params, exploration_noise=0.1, noise_clip=0.1)(state.obs, jax.random.PRNGKey(0))[0]
        action = inference(decoded_params, exploration_noise=0.1, noise_clip=0.1)(state.obs, jax.random.PRNGKey(0))[0]
        self.assertSequenceEqual(original_action, action)
        env.step(state, action)

    def testTrainDomainRandomize(self):
        """Test with domain randomization."""

        def rand_fn(sys, rng):
            @jax.vmap
            def get_offset(rng):
                offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
                pos = sys.link.transform.pos.at[0].set(offset)
                return pos

            sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
            in_axes = jax.tree.map(lambda x: None, sys)
            in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
            return sys_v, in_axes

        _, _, _ = td3.train(
            envs.get_environment('inverted_pendulum', backend='spring'),
            num_timesteps=1280,
            num_envs=128,
            episode_length=128,
            randomization_fn=rand_fn,
        )

    def test_td3_networks_output_shapes(self):
        obs_size = 24
        action_size = 4

        dummy_observation = jnp.ones((obs_size,))

        td3_nets = td3_networks.make_td3_networks(obs_size, action_size)

        key = jax.random.PRNGKey(0)
        key_policy, key_q = jax.random.split(key)
        policy_params = td3_nets.policy_network.init(key_policy)
        q_params = td3_nets.q_network.init(key_q)
        dummy_preprocess_params = {}

        inference = td3_networks.make_inference_fn(td3_nets)
        policy_fn = inference((dummy_preprocess_params, policy_params), exploration_noise=0.1, noise_clip=0.1)

        action_output, _ = policy_fn(dummy_observation, jax.random.PRNGKey(0))
        self.assertEqual(action_output.shape, (action_size,),
                         f"Expected action shape {(action_size,)}, but got {action_output.shape}")

        q_output = td3_nets.q_network.apply(dummy_preprocess_params, q_params, dummy_observation, action_output)
        self.assertEqual(q_output.shape, (2,), f"Expected Q-value shape {(2,)}, but got {q_output.shape}")


if __name__ == '__main__':
    absltest.main()
