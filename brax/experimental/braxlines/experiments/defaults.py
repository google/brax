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

"""Common default parameters."""
import copy

DEFAULT_PPO_PARAMS = {
    'ant':
        dict(
            num_timesteps=30000000,
            log_frequency=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024),
    'humanoid':
        dict(
            num_timesteps=50000000,
            log_frequency=20,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=1024),
    'fetch':
        dict(
            num_timesteps=100_000_000,
            log_frequency=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.997,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=256),
    'grasp':
        dict(
            num_timesteps=600_000_000,
            log_frequency=10,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=2,
            discounting=0.99,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=256),
    'halfcheetah':
        dict(
            num_timesteps=100_000_000,
            log_frequency=10,
            reward_scaling=1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=512),
    'ur5e':
        dict(
            num_timesteps=20_000_000,
            log_frequency=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.95,
            learning_rate=2e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            max_devices_per_host=8),
    'reacher':
        dict(
            num_timesteps=100_000_000,
            log_frequency=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=4,
            unroll_length=50,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=256,
            max_devices_per_host=8),
}
DEFAULT_PPO_PARAMS.update(
    dict(
        hopper=DEFAULT_PPO_PARAMS['halfcheetah'],
        walker2d=DEFAULT_PPO_PARAMS['halfcheetah']))


def get_ppo_params(env_name: str,
                   timesteps_multiplier: float = 1,
                   default: str = None,
                   **kwargs):
  """Get Brax Training default ppo params."""
  if default:
    ppo_params = DEFAULT_PPO_PARAMS.get(env_name, DEFAULT_PPO_PARAMS[default])
  else:
    ppo_params = DEFAULT_PPO_PARAMS[env_name]
  ppo_params = copy.deepcopy(ppo_params)
  ppo_params['num_timesteps'] = int(ppo_params['num_timesteps'] *
                                    timesteps_multiplier)
  ppo_params.update(kwargs)
  return ppo_params
