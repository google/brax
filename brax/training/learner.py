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

"""RL training with an environment running entirely on an accelerator."""

import functools
import json
import os

os.environ['XLA_FLAGS'] = '--xla_gpu_graph_min_graph_size=1'

from absl import app
from absl import flags
from absl import logging
from brax import envs
from brax.io import metrics
from brax.io import model
from brax.training.agents.apg import train as apg
from brax.training.agents.ars import train as ars
from brax.training.agents.es import train as es
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
import jax
import mediapy as media
try:
  import mujoco_playground as mjp
except ImportError as e:
  print(
      'MuJoCo Playground is not available. Install it via `pip install'
      ' playground`.\n'
      + str(e)
  )
  mjp = None

FLAGS = flags.FLAGS
_LEARNER = flags.DEFINE_enum(
    'learner',
    'ppo',
    ['ppo', 'apg', 'es', 'sac', 'ars'],
    'Which algorithm to run.',
)
_ENV = flags.DEFINE_string('env', 'ant', 'Name of environment to train.')

_BACKEND = flags.DEFINE_enum(
    'backend',
    'mjx',
    ['mjx', 'spring', 'generalized', 'positional'],
    'The physics backend to use.',
)
_TOTAL_ENV_STEPS = flags.DEFINE_integer(
    'total_env_steps', 50000000, 'Number of env steps to run training for.'
)
_NUM_EVALS = flags.DEFINE_integer(
    'num_evals', 10, 'How many times to run an eval.'
)
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_NUM_ENVS = flags.DEFINE_integer(
    'num_envs', 4, 'Number of envs to run in parallel.'
)
_ACTION_REPEAT = flags.DEFINE_integer('action_repeat', 1, 'Action repeat.')
_UNROLL_LENGTH = flags.DEFINE_integer('unroll_length', 30, 'Unroll length.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 4, 'Batch size.')
_NUM_MINIBATCHES = flags.DEFINE_integer('num_minibatches', 1, 'Number')
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    'num_updates_per_batch',
    1,
    'Number of times to reuse each transition for gradient computation.',
)
_REWARD_SCALING = flags.DEFINE_float('reward_scaling', 10.0, 'Reward scale.')
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', 3e-4, 'Entropy cost.')
_EPISODE_LENGTH = flags.DEFINE_integer(
    'episode_length', 1000, 'Episode length.'
)
_DISCOUNTING = flags.DEFINE_float('discounting', 0.99, 'Discounting.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate.')
_MAX_GRADIENT_NORM = flags.DEFINE_float(
    'max_gradient_norm', 1e9, 'Maximal norm of a gradient update.'
)
_LOGDIR = flags.DEFINE_string('logdir', '', 'Logdir.')
_NORMALIZE_OBSERVATIONS = flags.DEFINE_bool(
    'normalize_observations',
    True,
    'Whether to apply observation normalization.',
)
_MAX_DEVICES_PER_HOST = flags.DEFINE_integer(
    'max_devices_per_host',
    None,
    'Maximum number of devices to use per host. If None, '
    'defaults to use as much as it can.',
)
_NUM_VIDEOS = flags.DEFINE_integer(
    'num_videos', 1, 'Number of videos to record after training.'
)
# Evolution Strategy related flags
_POPULATION_SIZE = flags.DEFINE_integer(
    'population_size',
    1,
    'Number of environments in ES. The actual number is 2x '
    'larger (used for antithetic sampling.',
)
_PERTURBATION_STD = flags.DEFINE_float(
    'perturbation_std', 0.1, 'Std of a random noise added by ES.'
)
_FITNESS_SHAPING = flags.DEFINE_enum(
    'fitness_shaping',
    'original',
    ['original', 'centered_rank', 'wierstra'],
    'Defines a type of fitness shaping to apply.',
)
_CENTER_FITNESS = flags.DEFINE_bool(
    'center_fitness', False, 'Whether to normalize fitness after the shaping.'
)
_L2COEFF = flags.DEFINE_float(
    'l2coeff', 0, 'L2 regularization coefficient for model params.'
)
# SAC hps.
_MIN_REPLAY_SIZE = flags.DEFINE_integer(
    'min_replay_size',
    8192,
    'Minimal replay buffer size before the training starts.',
)
_MAX_REPLAY_SIZE = flags.DEFINE_integer(
    'max_replay_size', 1048576, 'Maximal replay buffer size.'
)
_GRAD_UPDATES_PER_STEP = flags.DEFINE_integer(
    'grad_updates_per_step',
    1,
    'How many SAC gradient updates to run per one step in the environment.',
)
_Q_NETWORK_LAYER_NORM = flags.DEFINE_bool(
    'q_network_layer_norm', False, 'Critic network layer norm.'
)
# PPO hps.
_GAE_LAMBDA = flags.DEFINE_float(
    'gae_lambda', 0.95, 'General advantage estimation lambda.'
)
_CLIPPING_EPSILON = flags.DEFINE_float(
    'clipping_epsilon', 0.3, 'Policy loss clipping epsilon.'
)
_NUM_RESETS_PER_EVAL = flags.DEFINE_integer(
    'num_resets_per_eval', 10, 'Number of resets per eval.'
)
_PPO_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_string(
    'ppo_policy_hidden_layer_sizes', None, 'PPO policy hidden layer sizes.'
)
_PPO_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_string(
    'ppo_value_hidden_layer_sizes', None, 'PPO value hidden layer sizes.'
)
_PPO_POLICY_OBS_KEY = flags.DEFINE_string(
    'ppo_policy_obs_key', None, 'PPO policy obs key.'
)
_PPO_VALUE_OBS_KEY = flags.DEFINE_string(
    'ppo_value_obs_key', None, 'PPO value obs key.'
)
# ARS hps.
_NUMBER_OF_DIRECTIONS = flags.DEFINE_integer(
    'number_of_directions',
    60,
    'Number of directions to explore. The actual number is 2x '
    'larger (used for antithetic sampling.',
)
_TOP_DIRECTIONS = flags.DEFINE_integer(
    'top_directions', 20, 'Number of top directions to select.'
)
_EXPLORATION_NOISE_STD = flags.DEFINE_float(
    'exploration_noise_std', 0.1, 'Std of a random noise added by ARS.'
)
_REWARD_SHIFT = flags.DEFINE_float(
    'reward_shift', 0.0, 'A reward shift to get rid of "stay alive" bonus.'
)
# ARS hps.
_POLICY_UPDATES = flags.DEFINE_integer(
    'policy_updates', None, 'Number of policy updates in APG.'
)
# MuJoCo Playground.
_PLAYGROUND_DM_CONTROL_SUITE = flags.DEFINE_bool(
    'playground_dm_control_suite', False, 'Use the playground dm control suite.'
)
_PLAYGROUND_LOCOMOTION_SUITE = flags.DEFINE_bool(
    'playground_locomotion_suite', False, 'Use the playground locomotion suite.'
)
_PLAYGROUND_MANIPULATION_SUITE = flags.DEFINE_bool(
    'playground_manipulation_suite',
    False,
    'Use the playground manipulation suite.',
)
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    'playground_config_overrides',
    None,
    'Overrides for the playground env config.',
)


def get_env_factory(env_name: str):
  """Returns a function that creates an environment."""
  wrap_fn = None
  randomizer_fn = None
  if mjp:  # MuJoCo Playground environments.
    overrides = {}
    randomizer_fn = None
    if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
      overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
    if _PLAYGROUND_DM_CONTROL_SUITE.value:
      get_environment = lambda *args, **kwargs: mjp.dm_control_suite.load(  # pytype: disable=attribute-error
          *args, **kwargs, config_overrides=overrides
      )
    elif _PLAYGROUND_LOCOMOTION_SUITE.value:
      get_environment = lambda *args, **kwargs: mjp.locomotion.load(  # pytype: disable=attribute-error
          *args, **kwargs, config_overrides=overrides
      )
      randomizer_fn = mjp.locomotion.get_domain_randomizer(env_name)
    elif _PLAYGROUND_MANIPULATION_SUITE.value:
      get_environment = lambda *args, **kwargs: mjp.manipulation.load(  # pytype: disable=attribute-error
          *args, **kwargs, config_overrides=overrides
      )
      randomizer_fn = mjp.manipulation.get_domain_randomizer(env_name)
    else:
      raise ValueError('No playground suite selected.')
    wrap_fn = mjp.wrapper.wrap_for_brax_training
  else:
    get_environment = functools.partial(
        envs.get_environment, backend=_BACKEND.value
    )
  return get_environment, wrap_fn, randomizer_fn


def main(unused_argv):
  logdir = _LOGDIR.value

  ckpt_dir = epath.Path(logdir) / 'checkpoints'
  ckpt_dir.mkdir(exist_ok=True)
  get_environment, wrap_fn, randomizer_fn = get_env_factory(_ENV.value)
  with metrics.Writer(logdir) as writer:
    writer.write_hparams({
        'num_evals': _NUM_EVALS.value,
        'num_envs': _NUM_ENVS.value,
        'total_env_steps': _TOTAL_ENV_STEPS.value,
    })
    if _LEARNER.value == 'sac':
      network_factory = sac_networks.make_sac_networks
      if _Q_NETWORK_LAYER_NORM.value:
        network_factory = functools.partial(
            sac_networks.make_sac_networks, q_network_layer_norm=True
        )
      make_policy, params, _ = sac.train(
          environment=get_environment(_ENV.value),
          eval_env=get_environment(_ENV.value),
          wrap_env_fn=wrap_fn,
          randomization_fn=randomizer_fn,
          num_envs=_NUM_ENVS.value,
          action_repeat=_ACTION_REPEAT.value,
          normalize_observations=_NORMALIZE_OBSERVATIONS.value,
          num_timesteps=_TOTAL_ENV_STEPS.value,
          num_evals=_NUM_EVALS.value,
          batch_size=_BATCH_SIZE.value,
          min_replay_size=_MIN_REPLAY_SIZE.value,
          max_replay_size=_MAX_REPLAY_SIZE.value,
          network_factory=network_factory,
          learning_rate=_LEARNING_RATE.value,
          discounting=_DISCOUNTING.value,
          max_devices_per_host=_MAX_DEVICES_PER_HOST.value,
          seed=_SEED.value,
          reward_scaling=_REWARD_SCALING.value,
          grad_updates_per_step=_GRAD_UPDATES_PER_STEP.value,
          episode_length=_EPISODE_LENGTH.value,
          progress_fn=writer.write_scalars,
      )
    elif _LEARNER.value == 'es':
      make_policy, params, _ = es.train(
          environment=get_environment(_ENV.value),
          eval_env=get_environment(_ENV.value),
          wrap_env_fn=wrap_fn,
          randomization_fn=randomizer_fn,
          num_timesteps=_TOTAL_ENV_STEPS.value,
          fitness_shaping=es.FitnessShaping[_FITNESS_SHAPING.value.upper()],
          population_size=_POPULATION_SIZE.value,
          perturbation_std=_PERTURBATION_STD.value,
          normalize_observations=_NORMALIZE_OBSERVATIONS.value,
          action_repeat=_ACTION_REPEAT.value,
          num_evals=_NUM_EVALS.value,
          center_fitness=_CENTER_FITNESS.value,
          l2coeff=_L2COEFF.value,
          learning_rate=_LEARNING_RATE.value,
          seed=_SEED.value,
          max_devices_per_host=_MAX_DEVICES_PER_HOST.value,
          episode_length=_EPISODE_LENGTH.value,
          progress_fn=writer.write_scalars,
      )
    elif _LEARNER.value == 'ppo':
      network_factory = ppo_networks.make_ppo_networks
      if _PPO_POLICY_HIDDEN_LAYER_SIZES.value is not None:
        policy_hidden_layer_sizes = [
            int(x) for x in _PPO_POLICY_HIDDEN_LAYER_SIZES.value.split(',')
        ]
        network_factory = functools.partial(
            network_factory,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        )
      if _PPO_VALUE_HIDDEN_LAYER_SIZES.value is not None:
        value_hidden_layer_sizes = [
            int(x) for x in _PPO_VALUE_HIDDEN_LAYER_SIZES.value.split(',')
        ]
        network_factory = functools.partial(
            network_factory,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
        )
      if _PPO_POLICY_OBS_KEY.value is not None:
        network_factory = functools.partial(
            network_factory,
            policy_obs_key=_PPO_POLICY_OBS_KEY.value,
        )
      if _PPO_VALUE_OBS_KEY.value is not None:
        network_factory = functools.partial(
            network_factory,
            value_obs_key=_PPO_VALUE_OBS_KEY.value,
        )
      make_policy, params, _ = ppo.train(
          environment=get_environment(_ENV.value),
          eval_env=get_environment(_ENV.value),
          wrap_env_fn=wrap_fn,
          randomization_fn=randomizer_fn,
          num_timesteps=_TOTAL_ENV_STEPS.value,
          episode_length=_EPISODE_LENGTH.value,
          network_factory=network_factory,
          action_repeat=_ACTION_REPEAT.value,
          num_envs=_NUM_ENVS.value,
          max_devices_per_host=_MAX_DEVICES_PER_HOST.value,
          learning_rate=_LEARNING_RATE.value,
          entropy_cost=_ENTROPY_COST.value,
          discounting=_DISCOUNTING.value,
          seed=_SEED.value,
          unroll_length=_UNROLL_LENGTH.value,
          batch_size=_BATCH_SIZE.value,
          num_minibatches=_NUM_MINIBATCHES.value,
          normalize_observations=_NORMALIZE_OBSERVATIONS.value,
          num_updates_per_batch=_NUM_UPDATES_PER_BATCH.value,
          num_evals=_NUM_EVALS.value,
          reward_scaling=_REWARD_SCALING.value,
          gae_lambda=_GAE_LAMBDA.value,
          clipping_epsilon=_CLIPPING_EPSILON.value,
          num_resets_per_eval=_NUM_RESETS_PER_EVAL.value,
          progress_fn=writer.write_scalars,
          save_checkpoint_path=ckpt_dir.as_posix(),
      )
    elif _LEARNER.value == 'apg':
      make_policy, params, _ = apg.train(
          environment=get_environment(_ENV.value),
          eval_env=get_environment(_ENV.value),
          wrap_env_fn=wrap_fn,
          randomization_fn=randomizer_fn,
          policy_updates=_POLICY_UPDATES.value,
          num_envs=_NUM_ENVS.value,
          action_repeat=_ACTION_REPEAT.value,
          num_evals=_NUM_EVALS.value,
          learning_rate=_LEARNING_RATE.value,
          seed=_SEED.value,
          max_devices_per_host=_MAX_DEVICES_PER_HOST.value,
          normalize_observations=_NORMALIZE_OBSERVATIONS.value,
          max_gradient_norm=_MAX_GRADIENT_NORM.value,
          episode_length=_EPISODE_LENGTH.value,
          progress_fn=writer.write_scalars,
      )
    elif _LEARNER.value == 'ars':
      make_policy, params, _ = ars.train(
          environment=get_environment(_ENV.value),
          eval_env=get_environment(_ENV.value),
          wrap_env_fn=wrap_fn,
          randomization_fn=randomizer_fn,
          number_of_directions=_NUMBER_OF_DIRECTIONS.value,
          max_devices_per_host=_MAX_DEVICES_PER_HOST.value,
          action_repeat=_ACTION_REPEAT.value,
          normalize_observations=_NORMALIZE_OBSERVATIONS.value,
          num_timesteps=_TOTAL_ENV_STEPS.value,
          exploration_noise_std=_EXPLORATION_NOISE_STD.value,
          num_evals=_NUM_EVALS.value,
          seed=_SEED.value,
          step_size=_LEARNING_RATE.value,
          top_directions=_TOP_DIRECTIONS.value,
          reward_shift=_REWARD_SHIFT.value,
          episode_length=_EPISODE_LENGTH.value,
          progress_fn=writer.write_scalars,
      )
    else:
      raise ValueError(f'Unknown learner: {_LEARNER.value}')

  # Save to flax serialized checkpoint.
  filename = f'{_ENV.value}_{_LEARNER.value}.pkl'
  path = os.path.join(logdir, filename)
  model.save_params(path, params)

  # Output an episode trajectory.
  get_environment, *_ = get_env_factory(_ENV.value)
  env = get_environment(_ENV.value)

  def do_rollout(rng, state):
    data_attr_name = 'pipeline_state' if hasattr(env, 'sys') else 'data'
    empty_data = getattr(state, data_attr_name).__class__(
        **{k: None for k in getattr(state, data_attr_name).__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(**{data_attr_name: empty_data})

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = make_policy(params)(state.obs, act_key)[0]
      state = env.step(state, act)
      if hasattr(state, 'data'):
        # select a sub-set of the data for playground envs
        traj_data = empty_traj.tree_replace({
            'data.qpos': state.data.qpos,
            'data.qvel': state.data.qvel,
            'data.time': state.data.time,
            'data.ctrl': state.data.ctrl,
            'data.mocap_pos': state.data.mocap_pos,
            'data.mocap_quat': state.data.mocap_quat,
            'data.xfrc_applied': state.data.xfrc_applied,
        })
      elif hasattr(state, 'pipeline_state'):
        # select the entire state for brax envs
        traj_data = empty_traj.replace(
            **{data_attr_name: getattr(state, data_attr_name)}
        )
      else:
        raise ValueError(
            f'Unknown data attribute name: {data_attr_name} on state: {state}.'
        )
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(env.reset))(rng)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  video_path = ''
  if hasattr(env, 'render'):
    for i in range(_NUM_VIDEOS.value):
      path_ = epath.Path(f'{logdir}/saved_videos/trajectory_{i:04d}.mp4')
      path_.parent.mkdir(parents=True, exist_ok=True)
      frames = env.render(trajectories[i])
      media.write_video(path_, frames, fps=1.0 / env.dt)
      video_path = path_.as_posix()
  elif _NUM_VIDEOS.value > 0:
    logging.warn('Cannot save videos for non physics environments.')



if __name__ == '__main__':
  app.run(main)
