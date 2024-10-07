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

"""RL training with an environment running entirely on an accelerator."""

import functools
import os

from absl import app
from absl import flags
from absl import logging
from brax import envs
from brax.io import html
from brax.io import metrics
from brax.io import model
from brax.training.agents.apg import train as apg
from brax.training.agents.ars import train as ars
from brax.training.agents.es import train as es
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.v1 import envs as envs_v1
from brax.v1.io import html as html_v1
from brax.v1.io import npy_file
import jax


FLAGS = flags.FLAGS

flags.DEFINE_enum('learner', 'ppo', ['ppo', 'apg', 'es', 'sac', 'ars'],
                  'Which algorithm to run.')
flags.DEFINE_string('env', 'ant', 'Name of environment to train.')

# TODO move npy_file to v2.

flags.DEFINE_bool('use_v2', True, 'Use Brax v2.')
flags.DEFINE_enum(
    'backend',
    'mjx',
    ['mjx', 'spring', 'generalized', 'positional'],
    'The physics backend to use.',
)
flags.DEFINE_bool('legacy_spring', False, 'Brax v1 backend.')
flags.DEFINE_integer('total_env_steps', 50000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('num_evals', 10, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 4, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('unroll_length', 30, 'Unroll length.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_minibatches', 1, 'Number')
flags.DEFINE_integer(
    'num_updates_per_batch', 1,
    'Number of times to reuse each transition for gradient '
    'computation.')
flags.DEFINE_float('reward_scaling', 10.0, 'Reward scale.')
flags.DEFINE_float('entropy_cost', 3e-4, 'Entropy cost.')
flags.DEFINE_integer('episode_length', 1000, 'Episode length.')
flags.DEFINE_float('discounting', 0.99, 'Discounting.')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 1e9,
                   'Maximal norm of a gradient update.')
flags.DEFINE_string('logdir', '', 'Logdir.')
flags.DEFINE_bool('normalize_observations', True,
                  'Whether to apply observation normalization.')
flags.DEFINE_integer(
    'max_devices_per_host', None,
    'Maximum number of devices to use per host. If None, '
    'defaults to use as much as it can.')
flags.DEFINE_integer('num_videos', 1,
                     'Number of videos to record after training.')
flags.DEFINE_integer('num_trajectories_npy', 0,
                     'Number of rollouts to write to disk as raw QP states.')
# Evolution Strategy related flags
flags.DEFINE_integer(
    'population_size', 1,
    'Number of environments in ES. The actual number is 2x '
    'larger (used for antithetic sampling.')
flags.DEFINE_float('perturbation_std', 0.1,
                   'Std of a random noise added by ES.')
flags.DEFINE_enum('fitness_shaping', 'original',
                  ['original', 'centered_rank', 'wierstra'],
                  'Defines a type of fitness shaping to apply.')
flags.DEFINE_bool('center_fitness', False,
                  'Whether to normalize fitness after the shaping.')
flags.DEFINE_float('l2coeff', 0,
                   'L2 regularization coefficient for model params.')
# SAC hps.
flags.DEFINE_integer('min_replay_size', 8192,
                     'Minimal replay buffer size before the training starts.')
flags.DEFINE_integer('max_replay_size', 1048576, 'Maximal replay buffer size.')
flags.DEFINE_integer(
    'grad_updates_per_step', 1,
    'How many SAC gradient updates to run per one step in the '
    'environment.')
# PPO hps.
flags.DEFINE_float('gae_lambda', .95, 'General advantage estimation lambda.')
flags.DEFINE_float('clipping_epsilon', .3, 'Policy loss clipping epsilon.')
flags.DEFINE_integer('num_resets_per_eval', 10, 'Number of resets per eval.')
# ARS hps.
flags.DEFINE_integer(
    'number_of_directions', 60,
    'Number of directions to explore. The actual number is 2x '
    'larger (used for antithetic sampling.')
flags.DEFINE_integer('top_directions', 20,
                     'Number of top directions to select.')
flags.DEFINE_float('exploration_noise_std', 0.1,
                   'Std of a random noise added by ARS.')
flags.DEFINE_float('reward_shift', 0.,
                   'A reward shift to get rid of "stay alive" bonus.')

# ARS hps.
flags.DEFINE_integer('policy_updates', None,
                     'Number of policy updates in APG.')


def main(unused_argv):

  if FLAGS.use_v2:
    get_environment = functools.partial(
        envs.get_environment, backend=FLAGS.backend
    )
  else:
    get_environment = functools.partial(
        envs_v1.get_environment, legacy_spring=FLAGS.legacy_spring
    )

  with metrics.Writer(FLAGS.logdir) as writer:
    writer.write_hparams({
        'num_evals': FLAGS.num_evals,
        'num_envs': FLAGS.num_envs,
        'total_env_steps': FLAGS.total_env_steps
    })
    if FLAGS.learner == 'sac':
      make_policy, params, _ = sac.train(
          environment=get_environment(FLAGS.env),
          num_envs=FLAGS.num_envs,
          action_repeat=FLAGS.action_repeat,
          normalize_observations=FLAGS.normalize_observations,
          num_timesteps=FLAGS.total_env_steps,
          num_evals=FLAGS.num_evals,
          batch_size=FLAGS.batch_size,
          min_replay_size=FLAGS.min_replay_size,
          max_replay_size=FLAGS.max_replay_size,
          learning_rate=FLAGS.learning_rate,
          discounting=FLAGS.discounting,
          max_devices_per_host=FLAGS.max_devices_per_host,
          seed=FLAGS.seed,
          reward_scaling=FLAGS.reward_scaling,
          grad_updates_per_step=FLAGS.grad_updates_per_step,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'es':
      make_policy, params, _ = es.train(
          environment=get_environment(FLAGS.env),
          num_timesteps=FLAGS.total_env_steps,
          fitness_shaping=es.FitnessShaping[FLAGS.fitness_shaping.upper()],
          population_size=FLAGS.population_size,
          perturbation_std=FLAGS.perturbation_std,
          normalize_observations=FLAGS.normalize_observations,
          action_repeat=FLAGS.action_repeat,
          num_evals=FLAGS.num_evals,
          center_fitness=FLAGS.center_fitness,
          l2coeff=FLAGS.l2coeff,
          learning_rate=FLAGS.learning_rate,
          seed=FLAGS.seed,
          max_devices_per_host=FLAGS.max_devices_per_host,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'ppo':
      make_policy, params, _ = ppo.train(
          environment=get_environment(FLAGS.env),
          num_timesteps=FLAGS.total_env_steps,
          episode_length=FLAGS.episode_length,
          action_repeat=FLAGS.action_repeat,
          num_envs=FLAGS.num_envs,
          max_devices_per_host=FLAGS.max_devices_per_host,
          learning_rate=FLAGS.learning_rate,
          entropy_cost=FLAGS.entropy_cost,
          discounting=FLAGS.discounting,
          seed=FLAGS.seed,
          unroll_length=FLAGS.unroll_length,
          batch_size=FLAGS.batch_size,
          num_minibatches=FLAGS.num_minibatches,
          normalize_observations=FLAGS.normalize_observations,
          num_updates_per_batch=FLAGS.num_updates_per_batch,
          num_evals=FLAGS.num_evals,
          reward_scaling=FLAGS.reward_scaling,
          gae_lambda=FLAGS.gae_lambda,
          clipping_epsilon=FLAGS.clipping_epsilon,
          num_resets_per_eval=FLAGS.num_resets_per_eval,
          progress_fn=writer.write_scalars,
      )
    if FLAGS.learner == 'apg':
      make_policy, params, _ = apg.train(
          environment=get_environment(FLAGS.env),
          policy_updates=FLAGS.policy_updates,
          num_envs=FLAGS.num_envs,
          action_repeat=FLAGS.action_repeat,
          num_evals=FLAGS.num_evals,
          learning_rate=FLAGS.learning_rate,
          seed=FLAGS.seed,
          max_devices_per_host=FLAGS.max_devices_per_host,
          normalize_observations=FLAGS.normalize_observations,
          max_gradient_norm=FLAGS.max_gradient_norm,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'ars':
      make_policy, params, _ = ars.train(
          environment=get_environment(FLAGS.env),
          number_of_directions=FLAGS.number_of_directions,
          max_devices_per_host=FLAGS.max_devices_per_host,
          action_repeat=FLAGS.action_repeat,
          normalize_observations=FLAGS.normalize_observations,
          num_timesteps=FLAGS.total_env_steps,
          exploration_noise_std=FLAGS.exploration_noise_std,
          num_evals=FLAGS.num_evals,
          seed=FLAGS.seed,
          step_size=FLAGS.learning_rate,
          top_directions=FLAGS.top_directions,
          reward_shift=FLAGS.reward_shift,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)

  # Save to flax serialized checkpoint.
  filename = f'{FLAGS.env}_{FLAGS.learner}.pkl'
  path = os.path.join(FLAGS.logdir, filename)
  model.save_params(path, params)

  # Output an episode trajectory.
  if FLAGS.use_v2:
    env = envs.create(FLAGS.env, backend=FLAGS.backend)
  else:
    env = envs_v1.create(FLAGS.env, legacy_spring=FLAGS.legacy_spring)

  @jax.jit
  def jit_next_state(state, key):
    new_key, tmp_key = jax.random.split(key)
    act = make_policy(params)(state.obs, tmp_key)[0]
    return env.step(state, act), act, new_key

  def do_rollout(rng):
    rng, env_key = jax.random.split(rng)
    state = env.reset(env_key)
    states = []
    while not state.done:
      if isinstance(env, envs.Env):
        states.append(state.pipeline_state)
      else:
        states.append(state.qp)
      state, _, rng = jit_next_state(state, rng)
    return states, rng

  trajectories = []
  rng = jax.random.PRNGKey(FLAGS.seed)
  for _ in range(max(FLAGS.num_videos, FLAGS.num_trajectories_npy)):
    qps, rng = do_rollout(rng)
    trajectories.append(qps)

  if hasattr(env, 'sys'):
    for i in range(FLAGS.num_videos):
      html_path = f'{FLAGS.logdir}/saved_videos/trajectory_{i:04d}.html'
      if isinstance(env, envs.Env):
        html.save(html_path, env.sys.tree_replace({'opt.timestep': env.dt}), trajectories[i])
      else:
        html_v1.save_html(html_path, env.sys, trajectories[i], make_dir=True)
  elif FLAGS.num_videos > 0:
    logging.warn('Cannot save videos for non physics environments.')

  for i in range(FLAGS.num_trajectories_npy):
    qp_path = f'{FLAGS.logdir}/saved_qps/trajectory_{i:04d}.npy'
    npy_file.save(qp_path, trajectories[i], make_dir=True)



if __name__ == '__main__':
  app.run(main)
