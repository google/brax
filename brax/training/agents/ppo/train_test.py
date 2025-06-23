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

"""PPO tests."""

import functools
import pickle
from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
import jax


class PPOTest(parameterized.TestCase):
  """Tests for PPO module."""


  @parameterized.parameters('ndarray', 'dict_state')
  def testTrain(self, obs_mode):
    """Test PPO with a simple env."""
    fast = envs.get_environment('fast', obs_mode=obs_mode)
    _, _, metrics = ppo.train(
        fast,
        num_timesteps=2**15,
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        num_evals=3,
        reward_scaling=10,
        normalize_advantage=False,
    )
    self.assertGreater(metrics['eval/episode_reward'], 135)
    self.assertEqual(fast.reset_count, 2)  # type: ignore
    self.assertEqual(fast.step_count, 2)  # type: ignore

  @parameterized.parameters(
      ('normal', 'scalar'),
      ('normal', 'log'),
      ('tanh_normal', 'log'),
  )
  def testTrainWithNetworkParams(self, distribution_type, noise_std_type):
    """Test PPO runs with different network params."""
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
    )

    _, _, _ = ppo.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**13,
        episode_length=50,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
    )

  def testTrainAsymmetricActorCritic(self):
    """Test PPO with asymmetric actor critic."""
    env = envs.get_environment(
        'fast', asymmetric_obs=True, obs_mode='dict_state'
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32,),
        value_hidden_layer_sizes=(32,),
        policy_obs_key='state',
        value_obs_key='privileged_state',
    )

    _, (_, policy_params, value_params), _ = ppo.train(
        env,
        num_timesteps=2**15,
        episode_length=1000,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=False,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
    )

    self.assertEqual(
        policy_params['params']['hidden_0']['kernel'].shape,
        (env.observation_size['state'], 32),
    )
    self.assertEqual(
        value_params['params']['hidden_0']['kernel'].shape,
        (env.observation_size['privileged_state'], 32),
    )

  @parameterized.parameters(True, False)
  def testNetworkEncoding(self, normalize_observations):
    env = envs.get_environment('fast')
    original_inference, params, _ = ppo.train(
        env,
        num_timesteps=128,
        episode_length=128,
        num_envs=128,
        normalize_observations=normalize_observations,
    )
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, normalize_fn
    )
    inference = ppo_networks.make_inference_fn(ppo_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action.
    state = env.reset(jax.random.PRNGKey(0))
    original_action = original_inference(decoded_params)(
        state.obs, jax.random.PRNGKey(0)
    )[0]
    action = inference(decoded_params)(state.obs, jax.random.PRNGKey(0))[0]
    self.assertSequenceEqual(original_action, action)
    env.step(state, action)

  def testTrainDomainRandomize(self):
    """Test PPO with domain randomization."""

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

    _, _, _ = ppo.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**15,
        episode_length=1000,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        randomization_fn=rand_fn,
    )

  @parameterized.parameters(
      {'asymmetric_obs': True, 'obs_mode': 'dict_pixels_state'},
      {'asymmetric_obs': False, 'obs_mode': 'dict_pixels_state'},
      {'asymmetric_obs': False, 'obs_mode': 'dict_pixels'},
  )
  def testPixelsPPO(self, asymmetric_obs, obs_mode):
    """Test PPO with pixel observations."""
    env = envs.get_environment(
        'fast',
        pixel_obs=True,
        asymmetric_obs=asymmetric_obs,
        obs_mode=obs_mode,
    )
    if obs_mode == 'dict_pixels':
      policy_obs_key = ''
      value_obs_key = ''
    else:
      policy_obs_key = 'state'
      value_obs_key = 'privileged_state' if asymmetric_obs else 'state'

    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=(32,),
        value_hidden_layer_sizes=(32,),
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )

    _, (_, policy_params, value_params), _ = ppo.train(
        env,
        num_timesteps=2**15,
        episode_length=1000,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
        augment_pixels=True,
    )
    num_views = 2
    cnn_features = 64

    if asymmetric_obs:
      self.assertEqual(
          policy_params['params']['MLP_0']['hidden_0']['kernel'].shape,
          (num_views * cnn_features + env.observation_size['state'], 32),
      )
      self.assertEqual(
          value_params['params']['MLP_0']['hidden_0']['kernel'].shape,
          (
              num_views * cnn_features
              + env.observation_size['privileged_state'],
              32,
          ),
      )
    if obs_mode == 'dict_pixels':
      self.assertEqual(
          policy_params['params']['MLP_0']['hidden_0']['kernel'].shape,
          (num_views * cnn_features, 32),
      )


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
