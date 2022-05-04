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

"""RL training with an environment running entirely on an accelerator."""

import os
import uuid

from absl import app
from absl import flags
from brax import envs
from brax.io import html
from brax.io import metrics
from brax.io import model
from brax.training import apg
from brax.training import ars
from brax.training import es
from brax.training import ppo
from brax.training import sac
import jax

FLAGS = flags.FLAGS

flags.DEFINE_enum('learner', 'ppo', ['ppo', 'apg', 'es', 'sac', 'ars'],
                  'Which algorithm to run.')
flags.DEFINE_string('env', 'ant', 'Name of environment to train.')
flags.DEFINE_integer('total_env_steps', 50000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_frequency', 10, 'How many times to run an eval.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_envs', 4, 'Number of envs to run in parallel.')
flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
flags.DEFINE_integer('unroll_length', 30, 'Unroll length.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_minibatches', 1, 'Number')
flags.DEFINE_integer('num_update_epochs', 1,
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
flags.DEFINE_integer('max_devices_per_host', None,
                     'Maximum number of devices to use per host. If None, '
                     'defaults to use as much as it can.')
flags.DEFINE_integer('num_videos', 1,
                     'Number of videos to record after training.')

# Evolution Strategy related flags
flags.DEFINE_integer('population_size', 1,
                     'Number of environments in ES. The actual number is 2x '
                     'larger (used for antithetic sampling.')
flags.DEFINE_float('perturbation_std', 0.1,
                   'Std of a random noise added by ES.')
flags.DEFINE_integer('fitness_shaping', 0,
                     'Defines a type of fitness shaping to apply.'
                     'Just check the code in es to figure out what '
                     'numbers mean.')
flags.DEFINE_bool('center_fitness', False,
                  'Whether to normalize fitness after the shaping.')
flags.DEFINE_float('l2coeff', 0,
                   'L2 regularization coefficient for model params.')
# SAC hps.
flags.DEFINE_integer('min_replay_size', 8192,
                     'Minimal replay buffer size before the training starts.')
flags.DEFINE_integer('max_replay_size', 1048576, 'Maximal replay buffer size.')
flags.DEFINE_float('grad_updates_per_step', 1.0,
                   'How many SAC gradient updates to run per one step in the '
                   'environment.')
# ARS hps.
flags.DEFINE_integer('number_of_directions', 60,
                     'Number of directions to explore. The actual number is 2x '
                     'larger (used for antithetic sampling.')
flags.DEFINE_integer('top_directions', 20,
                     'Number of top directions to select.')
flags.DEFINE_float('exploration_noise_std', 0.1,
                   'Std of a random noise added by ARS.')
flags.DEFINE_float('reward_shift', 0.,
                   'A reward shift to get rid of "stay alive" bonus.')
flags.DEFINE_enum('head_type', '', ['', 'clip', 'tanh'],
                  'Which policy head to use.')
# ARS hps.
flags.DEFINE_integer('truncation_length', None,
                     'Truncation for gradient propagation in APG.')


def main(unused_argv):

  env_fn = envs.create_fn(FLAGS.env)

  with metrics.Writer(FLAGS.logdir) as writer:
    writer.write_hparams({'log_frequency': FLAGS.eval_frequency,
                          'num_envs': FLAGS.num_envs,
                          'total_env_steps': FLAGS.total_env_steps})
    if FLAGS.learner == 'sac':
      inference_fn, params, _ = sac.train(
          environment_fn=env_fn,
          num_envs=FLAGS.num_envs,
          action_repeat=FLAGS.action_repeat,
          normalize_observations=FLAGS.normalize_observations,
          num_timesteps=FLAGS.total_env_steps,
          log_frequency=FLAGS.eval_frequency,
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
      inference_fn, params, _ = es.train(
          environment_fn=env_fn,
          num_timesteps=FLAGS.total_env_steps,
          fitness_shaping=FLAGS.fitness_shaping,
          population_size=FLAGS.population_size,
          perturbation_std=FLAGS.perturbation_std,
          normalize_observations=FLAGS.normalize_observations,
          action_repeat=FLAGS.action_repeat,
          log_frequency=FLAGS.eval_frequency,
          center_fitness=FLAGS.center_fitness,
          l2coeff=FLAGS.l2coeff,
          learning_rate=FLAGS.learning_rate,
          seed=FLAGS.seed,
          max_devices_per_host=FLAGS.max_devices_per_host,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'apg':
      inference_fn, params, _ = apg.train(
          environment_fn=env_fn,
          num_envs=FLAGS.num_envs,
          action_repeat=FLAGS.action_repeat,
          log_frequency=FLAGS.eval_frequency,
          learning_rate=FLAGS.learning_rate,
          seed=FLAGS.seed,
          max_devices_per_host=FLAGS.max_devices_per_host,
          normalize_observations=FLAGS.normalize_observations,
          max_gradient_norm=FLAGS.max_gradient_norm,
          episode_length=FLAGS.episode_length,
          truncation_length=FLAGS.truncation_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'ppo':
      inference_fn, params, _ = ppo.train(
          environment_fn=env_fn,
          num_envs=FLAGS.num_envs,
          max_devices_per_host=FLAGS.max_devices_per_host,
          action_repeat=FLAGS.action_repeat,
          normalize_observations=FLAGS.normalize_observations,
          num_timesteps=FLAGS.total_env_steps,
          log_frequency=FLAGS.eval_frequency,
          batch_size=FLAGS.batch_size,
          unroll_length=FLAGS.unroll_length,
          num_minibatches=FLAGS.num_minibatches,
          num_update_epochs=FLAGS.num_update_epochs,
          learning_rate=FLAGS.learning_rate,
          entropy_cost=FLAGS.entropy_cost,
          discounting=FLAGS.discounting,
          seed=FLAGS.seed,
          reward_scaling=FLAGS.reward_scaling,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'ars':
      inference_fn, params, _ = ars.train(
          environment_fn=env_fn,
          number_of_directions=FLAGS.number_of_directions,
          max_devices_per_host=FLAGS.max_devices_per_host,
          action_repeat=FLAGS.action_repeat,
          normalize_observations=FLAGS.normalize_observations,
          num_timesteps=FLAGS.total_env_steps,
          exploration_noise_std=FLAGS.exploration_noise_std,
          log_frequency=FLAGS.eval_frequency,
          seed=FLAGS.seed,
          step_size=FLAGS.learning_rate,
          top_directions=FLAGS.top_directions,
          reward_shift=FLAGS.reward_shift,
          head_type=FLAGS.head_type,
          episode_length=FLAGS.episode_length,
          progress_fn=writer.write_scalars)

  # Save to flax serialized checkpoint.
  filename = f'{FLAGS.env}_{FLAGS.learner}.pkl'
  path = os.path.join(FLAGS.logdir, filename)
  model.save_params(path, params)

  # Output an episode trajectory.
  env = env_fn()

  @jax.jit
  def jit_next_state(state, key):
    new_key, tmp_key = jax.random.split(key)
    act = inference_fn(params, state.obs, tmp_key)
    return env.step(state, act), new_key

  for i in range(FLAGS.num_videos):
    rng = jax.random.PRNGKey(FLAGS.seed + i)
    rng, env_key = jax.random.split(rng)
    state = env.reset(env_key)
    qps = []
    while not state.done:
      qps.append(state.qp)
      state, rng = jit_next_state(state, rng)

    html_path = f'{FLAGS.logdir}/trajectory_{uuid.uuid4()}.html'
    html.save_html(html_path, env.sys, qps)



if __name__ == '__main__':
  app.run(main)
