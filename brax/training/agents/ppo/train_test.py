# Copyright 2026 The Brax Authors.
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
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
import jax
from jax import numpy as jnp
import numpy as np
import optax


class PPOTest(parameterized.TestCase):
  """Tests for PPO module."""

  @parameterized.parameters(
      (0.25, [0.25, 0.25, 0.25, 0.25, 0.25]),
      (optax.constant_schedule(0.25), [0.25, 0.25, 0.25, 0.25, 0.25]),
      (optax.linear_schedule(1.0, 0.0, 100), [1.0, 0.75, 0.5, 0.0, 0.0]),
      (
          optax.cosine_decay_schedule(1.0, 100),
          [1.0, (2 + 2**0.5) / 4, 0.5, 0.0, 0.0],
      ),
      (
          lambda step: jnp.where(step < 50, 1.0, 0.25),
          [1.0, 1.0, 0.25, 0.25, 0.25],
      ),
  )
  def testEntropyCostSchedule(self, entropy_cost, expected):
    compute = jax.jit(
        functools.partial(ppo._compute_entropy_cost, entropy_cost)
    )
    actual = jnp.array(
        [compute(types.UInt64(hi=0, lo=step)) for step in [0, 25, 50, 100, 150]]
    )
    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-7)

  def testEntropyCostScheduleUInt64(self):
    compute = jax.jit(
        functools.partial(
            ppo._compute_entropy_cost,
            lambda step: 1.0 - step / float(2**33),
        )
    )
    # Use representable float32 counts on both sides of the low-word rollover.
    steps = types.UInt64(
        hi=jnp.array([0, 1, 1]),
        lo=jnp.array([2**32 - 4096, 0, 4096], dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        compute(steps), [0.5 + 2**-21, 0.5, 0.5 - 2**-21]
    )

  @parameterized.parameters(False, True)
  def testEntropyCostScheduleFloat64(self, scheduled):
    coefficient = 0.123456789012345
    entropy_cost = (
        (lambda _: jnp.asarray(coefficient, dtype=jnp.float64))
        if scheduled
        else coefficient
    )
    self.addCleanup(jax.config.update, 'jax_enable_x64', jax.config.x64_enabled)
    jax.config.update('jax_enable_x64', True)
    compute = jax.jit(
        functools.partial(ppo._compute_entropy_cost, entropy_cost)
    )
    actual = compute(types.UInt64(hi=0, lo=0))
    self.assertEqual(actual.dtype, jnp.float64)
    np.testing.assert_array_equal(actual, coefficient)
    loss = jax.jit(lambda cost, entropy: cost * -entropy)
    entropy = jnp.asarray(1.25, dtype=jnp.float64)
    np.testing.assert_array_equal(
        loss(actual, entropy), loss(coefficient, entropy)
    )

  def testTrainConstantEntropySchedule(self):
    train = functools.partial(
        ppo.train,
        envs.get_environment('fast'),
        num_timesteps=48,
        episode_length=16,
        num_envs=4,
        batch_size=4,
        unroll_length=2,
        num_minibatches=2,
        num_updates_per_batch=2,
        max_devices_per_host=1,
        run_evals=False,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(8,),
            value_hidden_layer_sizes=(8,),
        ),
    )
    _, scalar_params, scalar_metrics = train(entropy_cost=0.01)
    _, schedule_params, schedule_metrics = train(
        entropy_cost=optax.constant_schedule(0.01)
    )
    self.assertEqual(
        jax.tree.structure(scalar_params), jax.tree.structure(schedule_params)
    )
    for scalar, scheduled in zip(
        jax.tree.leaves(scalar_params), jax.tree.leaves(schedule_params)
    ):
      np.testing.assert_array_equal(scalar, scheduled)
    self.assertEqual(scalar_metrics.keys(), schedule_metrics.keys())
    for key in scalar_metrics:
      if key not in ('training/sps', 'training/walltime'):
        np.testing.assert_array_equal(
            scalar_metrics[key], schedule_metrics[key]
        )
    self.assertAlmostEqual(float(scalar_metrics['training/entropy_cost']), 0.01)

  @parameterized.parameters((1, 1), (2, 3))
  def testTrainEntropyScheduleClock(self, action_repeat, num_updates_per_batch):
    schedule_steps = []

    def entropy_schedule(step):
      jax.debug.callback(
          lambda value: schedule_steps.append(float(value)), step
      )
      return step / 1000.0

    steps_per_rollout = 4 * 2 * 2 * action_repeat
    _, _, metrics = ppo.train(
        envs.get_environment('fast'),
        num_timesteps=3 * steps_per_rollout,
        episode_length=16,
        num_envs=4,
        batch_size=4,
        unroll_length=2,
        num_minibatches=2,
        num_updates_per_batch=num_updates_per_batch,
        action_repeat=action_repeat,
        entropy_cost=entropy_schedule,
        max_devices_per_host=1,
        run_evals=False,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(8,),
            value_hidden_layer_sizes=(8,),
        ),
    )
    jax.effects_barrier()
    self.assertEqual(
        sorted(schedule_steps), [0, steps_per_rollout, 2 * steps_per_rollout]
    )
    self.assertAlmostEqual(
        float(metrics['training/entropy_cost']), steps_per_rollout / 1000.0
    )

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

  @parameterized.product(
      (
          dict(distribution_type='normal', noise_std_type='scalar'),
          dict(distribution_type='normal', noise_std_type='log'),
          dict(distribution_type='tanh_normal', noise_std_type='log'),
      ),
      normalize_mode=['welford', 'ema'],
      bootstrap_on_timeout=[True, False],
      clipping_epsilon_value=[None, 0.1],
  )
  def testTrainWithNetworkParams(
      self,
      distribution_type,
      noise_std_type,
      normalize_mode,
      bootstrap_on_timeout,
      clipping_epsilon_value,
  ):
    """Test PPO runs with different network params."""
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=0.8,
        activation=jax.nn.elu,
        policy_network_kernel_init_fn=jax.nn.initializers.orthogonal,
        policy_network_kernel_init_kwargs={'scale': jnp.sqrt(2.0)},
        value_network_kernel_init_fn=jax.nn.initializers.orthogonal,
        value_network_kernel_init_kwargs={'scale': jnp.sqrt(2.0)},
        mean_clip_scale=5.0,
        mean_kernel_init_fn=jax.nn.initializers.orthogonal,
        mean_kernel_init_kwargs={'scale': 0.001},
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
        max_grad_norm=1.0,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
        learning_rate_schedule='ADAPTIVE_KL',
        normalize_observations_mode=normalize_mode,
        bootstrap_on_timeout=bootstrap_on_timeout,
        clipping_epsilon_value=clipping_epsilon_value,
    )

  def testTrainWithDistributionalCritic(self):
    """Test PPO runs with distributional critic and adaptive KL."""
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        distribution_type='normal',
        noise_std_type='log',
        init_noise_std=0.8,
        activation=jax.nn.elu,
        use_distributional_critic=True,
    )

    _, _, metrics = ppo.train(
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
        max_grad_norm=1.0,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
        learning_rate_schedule='ADAPTIVE_KL',
        clipping_epsilon_value=1.0,
        use_distributional_critic=True,
    )
    # Verify training produced finite results.
    self.assertTrue(
        jnp.isfinite(metrics['eval/episode_reward']),
        f'Reward is not finite: {metrics["eval/episode_reward"]}',
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
        (env.observation_size['state'], 32),  # pyrefly: ignore[bad-index]
    )
    self.assertEqual(
        value_params['params']['hidden_0']['kernel'].shape,
        (env.observation_size['privileged_state'], 32),  # pyrefly: ignore[bad-index]
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
        env.observation_size, env.action_size, normalize_fn  # pyrefly: ignore[bad-argument-type]
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
        network_factory=network_factory,  # pyrefly: ignore[bad-argument-type]
        augment_pixels=True,
    )
    num_views = 2
    cnn_features = 64

    if asymmetric_obs:
      self.assertEqual(
          policy_params['params']['MLP_0']['hidden_0']['kernel'].shape,
          (num_views * cnn_features + env.observation_size['state'], 32),  # pyrefly: ignore[bad-index, unsupported-operation]
      )
      self.assertEqual(
          value_params['params']['MLP_0']['hidden_0']['kernel'].shape,
          (
              num_views * cnn_features  # pyrefly: ignore[unsupported-operation]
              + env.observation_size['privileged_state'],  # pyrefly: ignore[bad-index]
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
