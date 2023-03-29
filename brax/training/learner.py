# Copyright 2023 The Brax Authors.
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
from brax.io import model
from brax.training.agents.apg import train as apg
from brax.training.agents.ars import train as ars
from brax.training.agents.es import train as es
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.v1 import envs as envs_v1
from brax.v1.io import html as html_v1
from brax.v1.io import metrics as metrics_v1
from brax.v1.io import npy_file
from brax.v1.io import rlds
import jax


FLAGS = flags.FLAGS

flags.DEFINE_enum('learner', 'ppo', ['ppo', 'apg', 'es', 'sac', 'ars'],
                  'Which algorithm to run.')
flags.DEFINE_string('env', 'ant', 'Name of environment to train.')

# TODO move npy_file to v2.


  # TODO move metrics writer to v2.
  with metrics_v1.Writer(FLAGS.logdir) as writer:
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
          progress_fn=writer.write_scalars)
    if FLAGS.learner == 'apg':
      make_policy, params, _ = apg.train(
          environment=get_environment(FLAGS.env),
          num_envs=FLAGS.num_envs,
          action_repeat=FLAGS.action_repeat,
          num_evals=FLAGS.num_evals,
          learning_rate=FLAGS.learning_rate,
          seed=FLAGS.seed,
          max_devices_per_host=FLAGS.max_devices_per_host,
          normalize_observations=FLAGS.normalize_observations,
          max_gradient_norm=FLAGS.max_gradient_norm,
          episode_length=FLAGS.episode_length,
          truncation_length=FLAGS.truncation_length,
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
        html.save(html_path, env.sys.replace(dt=env.dt), trajectories[i])
      else:
        html_v1.save_html(html_path, env.sys, trajectories[i], make_dir=True)
  elif FLAGS.num_videos > 0:
    logging.warn('Cannot save videos for non physics environments.')

  for i in range(FLAGS.num_trajectories_npy):
    qp_path = f'{FLAGS.logdir}/saved_qps/trajectory_{i:04d}.npy'
    npy_file.save(qp_path, trajectories[i], make_dir=True)



if __name__ == '__main__':
  app.run(main)
