# Copyright 2021 The Brax Authors.
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

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from absl import logging
import flax
from flax import linen
import jax
import jax.numpy as jnp
import numpy as onp
from brax import envs
from brax.training import distribution
from brax.training import env
from brax.training import networks
from brax.training import normalization

Params = Mapping[str, Mapping[str, jnp.ndarray]]
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Transition:
  """Contains data for one environment step."""
  o_tm1: jnp.ndarray
  a_tm1: jnp.ndarray
  r_t: jnp.ndarray
  o_t: jnp.ndarray
  d_t: jnp.ndarray  # discount (1-done)


# The rewarder allows to change the reward of before the learner trains.
RewarderState = Any
RewarderInit = Callable[[int, PRNGKey], RewarderState]
ComputeReward = Callable[[RewarderState, Transition, PRNGKey],
                         Tuple[RewarderState, jnp.ndarray]]
Rewarder = Tuple[RewarderInit, ComputeReward]


@flax.struct.dataclass
class ReplayBuffer:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  current_position: jnp.ndarray
  current_size: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer: flax.optim.Optimizer
  q_optimizer: flax.optim.Optimizer
  target_q_params: Params
  key: PRNGKey
  steps: jnp.ndarray
  alpha_optimizer: flax.optim.Optimizer
  normalizer_params: Any
  # The is passed to the rewarder to update the reward.
  rewarder_state: Any


def make_sac_networks(
    param_size: int,
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> Tuple[networks.FeedForwardModel, networks.FeedForwardModel]:
  """Creates a policy and a value networks for SAC."""
  policy_module = networks.MLP(
      layer_sizes=hidden_layer_sizes + (param_size,),
      activation=linen.relu,
      kernel_init=jax.nn.initializers.lecun_uniform())

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int = 2

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = networks.MLP(
            layer_sizes=hidden_layer_sizes + (1,),
            activation=linen.relu,
            kernel_init=jax.nn.initializers.lecun_uniform())(hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule()

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  policy = networks.FeedForwardModel(
      init=lambda key: policy_module.init(key, dummy_obs),
      apply=policy_module.apply)
  value = networks.FeedForwardModel(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action),
      apply=q_module.apply)
  return policy, value


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    log_frequency: int = 10000,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.,
    tau: float = 0.005,
    min_replay_size: int = 8192,
    max_replay_size: int = 1048576,
    grad_updates_per_step: float = 1,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    # The rewarder is an init function and a compute_reward function.
    # It is used to change the reward before the learner trains on it.
    make_rewarder: Optional[Callable[[], Rewarder]] = None,
):
  """SAC training."""
  assert min_replay_size % num_envs == 0
  assert max_replay_size % min_replay_size == 0
  # jax.config.update('jax_log_compiles', True)

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(), process_count, process_id, local_device_count,
      local_devices_to_use)

  assert max_replay_size % local_devices_to_use == 0
  assert min_replay_size % local_devices_to_use == 0
  assert num_envs % local_devices_to_use == 0

  max_replay_size = max_replay_size // local_devices_to_use
  min_replay_size = min_replay_size // local_devices_to_use

  key = jax.random.PRNGKey(seed)
  key, key_models, key_env, key_eval, key_rewarder = jax.random.split(key, 5)

  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs // local_devices_to_use // process_count,
      episode_length=episode_length)
  key_envs = jax.random.split(key_env, local_devices_to_use)
  tmp_env_states = []
  for key in key_envs:
    first_state, step_fn = env.wrap(core_env, key)
    tmp_env_states.append(first_state)
  first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                  *tmp_env_states)

  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = env.wrap(core_eval_env, key_eval)

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)

  _, obs_size = eval_first_state.core.obs.shape

  policy_model, value_model = make_sac_networks(
      parametric_action_distribution.param_size,
      obs_size, core_env.action_size)

  log_alpha = jnp.asarray(0., dtype=jnp.float32)
  alpha_optimizer_def = flax.optim.Adam(learning_rate=3e-4)
  alpha_optimizer = alpha_optimizer_def.create(log_alpha)

  policy_optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  q_optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  key_policy, key_q = jax.random.split(key_models)
  policy_optimizer = policy_optimizer_def.create(policy_model.init(key_policy))
  q_optimizer = q_optimizer_def.create(value_model.init(key_q))

  policy_optimizer = normalization.bcast_local_devices(
      policy_optimizer, local_devices_to_use)
  q_optimizer = normalization.bcast_local_devices(
      q_optimizer, local_devices_to_use)
  alpha_optimizer = normalization.bcast_local_devices(
      alpha_optimizer, local_devices_to_use)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          obs_size, normalize_observations,
          pmap_to_devices=local_devices_to_use))

  if make_rewarder is not None:
    init, compute_reward = make_rewarder()
    rewarder_state = init(obs_size, key_rewarder)
    rewarder_state = normalization.bcast_local_devices(
        rewarder_state, local_devices_to_use)
  else:
    rewarder_state = None
    compute_reward = None

  key_debug = jax.random.PRNGKey(seed + 666)

  # EVAL
  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params, key = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    logits = policy_model.apply(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions)
    return (nstate, policy_params, normalizer_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params, normalizer_params):
    policy_params, normalizer_params = jax.tree_map(
        lambda x: x[0], (policy_params, normalizer_params))
    (state, _, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, policy_params, normalizer_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  # SAC
  target_entropy = -0.5 * core_env.action_size

  def alpha_loss(alpha: jnp.ndarray,
                 policy_params: Params,
                 transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_model.apply(policy_params, transitions.o_tm1)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

  def critic_loss(q_params: Params,
                  policy_params: Params,
                  target_q_params: Params,
                  alpha: jnp.ndarray,
                  transitions: Transition,
                  key: PRNGKey) -> jnp.ndarray:
    q_old_action = value_model.apply(q_params, transitions.o_tm1,
                                     transitions.a_tm1)
    next_dist_params = policy_model.apply(policy_params, transitions.o_t)
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key)
    next_log_prob = parametric_action_distribution.log_prob(next_dist_params,
                                                            next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = value_model.apply(target_q_params, transitions.o_t, next_action)
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(transitions.r_t * reward_scaling +
                                     transitions.d_t * discounting * next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    q_error *= jnp.expand_dims(transitions.d_t, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def actor_loss(policy_params: Params,
                 q_params: Params,
                 alpha: jnp.ndarray,
                 transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    dist_params = policy_model.apply(policy_params, transitions.o_tm1)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = value_model.apply(q_params, transitions.o_tm1, action)
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q
    return jnp.mean(actor_loss)

  alpha_grad = jax.jit(jax.value_and_grad(alpha_loss))
  critic_grad = jax.jit(jax.value_and_grad(critic_loss))
  actor_grad = jax.jit(jax.value_and_grad(actor_loss))

  @jax.jit
  def update_step(state: TrainingState, transitions: jnp.ndarray,
                  ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    normalized_transitions = Transition(
        o_tm1=obs_normalizer_apply_fn(state.normalizer_params,
                                      transitions[:, :obs_size]),
        o_t=obs_normalizer_apply_fn(state.normalizer_params,
                                    transitions[:, obs_size:2*obs_size]),
        a_tm1=transitions[:, 2 * obs_size:2*obs_size + core_env.action_size],
        r_t=transitions[:, -2],
        d_t=transitions[:, -1])

    (key, key_alpha, key_critic, key_actor,
     key_rewarder) = jax.random.split(state.key, 5)

    if compute_reward is not None:
      new_rewarder_state, rewards = compute_reward(state.rewarder_state,
                                                   normalized_transitions,
                                                   key_rewarder)
      # Assertion prevents building errors.
      assert hasattr(normalized_transitions, 'replace')
      normalized_transitions = normalized_transitions.replace(r_t=rewards)
    else:
      new_rewarder_state = state.rewarder_state

    alpha = jnp.exp(state.alpha_optimizer.target)
    alpha_loss, alpha_grads = alpha_grad(alpha,
                                         state.policy_optimizer.target,
                                         normalized_transitions, key_alpha)
    critic_loss, critic_grads = critic_grad(state.q_optimizer.target,
                                            state.policy_optimizer.target,
                                            state.target_q_params, alpha,
                                            normalized_transitions, key_critic)
    actor_loss, actor_grads = actor_grad(state.policy_optimizer.target,
                                         state.q_optimizer.target, alpha,
                                         normalized_transitions, key_actor)
    alpha_grads = jax.lax.pmean(alpha_grads, axis_name='i')
    critic_grads = jax.lax.pmean(critic_grads, axis_name='i')
    actor_grads = jax.lax.pmean(actor_grads, axis_name='i')

    policy_optimizer = state.policy_optimizer.apply_gradient(actor_grads)
    q_optimizer = state.q_optimizer.apply_gradient(critic_grads)
    new_target_q_params = jax.tree_multimap(
        lambda x, y: x * (1 - tau) + y * tau,
        state.target_q_params, q_optimizer.target)
    alpha_optimizer = state.alpha_optimizer.apply_gradient(alpha_grads)

    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'alpha': jnp.exp(alpha_optimizer.target),
    }

    new_state = TrainingState(
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        target_q_params=new_target_q_params,
        key=key,
        steps=state.steps + 1,
        alpha_optimizer=alpha_optimizer,
        normalizer_params=state.normalizer_params,
        rewarder_state=new_rewarder_state)
    return new_state, metrics

  def collect_data(training_state, state):
    key, key_sample = jax.random.split(training_state.key)
    normalized_obs = obs_normalizer_apply_fn(training_state.normalizer_params,
                                             state.core.obs)
    logits = policy_model.apply(training_state.policy_optimizer.target,
                                normalized_obs)
    actions = parametric_action_distribution.sample_no_postprocessing(
        logits, key_sample)
    postprocessed_actions = parametric_action_distribution.postprocess(actions)
    nstate = step_fn(state, postprocessed_actions)

    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, state.core.obs)

    training_state = training_state.replace(key=key,
                                            normalizer_params=normalizer_params)

    # Concatenating data into a single data blob performs faster than 5
    # separate tensors.
    concatenated_data = jnp.concatenate([
        state.core.obs,
        nstate.core.obs,
        postprocessed_actions,
        jnp.expand_dims(nstate.core.reward, axis=-1),
        jnp.expand_dims(1 - nstate.core.done, axis=-1),
    ], axis=-1)

    return training_state, nstate, concatenated_data

  def collect_and_update_buffer(training_state, state, replay_buffer):
    training_state, state, newdata = collect_data(training_state, state)
    new_replay_data = jax.tree_multimap(
        lambda x, y: jax.lax.dynamic_update_slice_in_dim(
            x, y, replay_buffer.current_position, axis=0),
        replay_buffer.data, newdata)
    new_position = (
        replay_buffer.current_position +
        num_envs // local_devices_to_use) % max_replay_size
    new_size = jnp.minimum(
        replay_buffer.current_size + num_envs // local_devices_to_use,
        max_replay_size)
    return training_state, state, ReplayBuffer(data=new_replay_data,
                                               current_position=new_position,
                                               current_size=new_size)

  def init_replay_buffer(training_state, state, replay_buffer):
    (training_state, state, replay_buffer), _ = jax.lax.scan(
        (lambda a, b: (collect_and_update_buffer(*a), ())),
        (training_state, state, replay_buffer), (),
        length=min_replay_size // (num_envs // local_devices_to_use))
    return training_state, state, replay_buffer
  init_replay_buffer = jax.pmap(init_replay_buffer, axis_name='i')

  num_updates = int(num_envs * grad_updates_per_step)

  def sample_data(training_state, replay_buffer):
    key1, key2 = jax.random.split(training_state.key)
    idx = jax.random.randint(
        key2, (batch_size * num_updates // local_devices_to_use,), minval=0,
        maxval=replay_buffer.current_size)
    transitions = jnp.take(replay_buffer.data, idx, axis=0, mode='clip')
    transitions = jnp.reshape(transitions,
                              [num_updates, -1] + list(transitions.shape[1:]))
    training_state = training_state.replace(key=key1)
    return training_state, transitions

  def run_one_sac_epoch(carry, unused_t):
    training_state, state, replay_buffer = carry

    training_state, state, replay_buffer = collect_and_update_buffer(
        training_state, state, replay_buffer)

    training_state, transitions = sample_data(training_state, replay_buffer)
    training_state, metrics = jax.lax.scan(
        update_step, training_state, transitions, length=num_updates)

    metrics['current_size'] = replay_buffer.current_size
    metrics['current_position'] = replay_buffer.current_position
    return (training_state, state, replay_buffer), metrics

  def run_sac_training(training_state, state, replay_buffer):
    (training_state, state, replay_buffer), metrics = jax.lax.scan(
        run_one_sac_epoch, (training_state, state, replay_buffer), (),
        length=(log_frequency // action_repeat + num_envs - 1) // num_envs)
    metrics = jax.tree_map(jnp.mean, metrics)
    return training_state, state, replay_buffer, metrics
  run_sac_training = jax.pmap(run_sac_training, axis_name='i')

  training_state = TrainingState(
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      target_q_params=q_optimizer.target,
      key=jnp.stack(jax.random.split(key, local_devices_to_use)),
      steps=jnp.zeros((local_devices_to_use,)),
      alpha_optimizer=alpha_optimizer,
      normalizer_params=normalizer_params,
      rewarder_state=rewarder_state)

  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  training_metrics = {}
  state = first_state
  metrics = {}

  while True:
    current_step = int(training_state.normalizer_params[0][0]) * action_repeat
    logging.info('step %s', current_step)

    t = time.time()
    eval_state, key_debug = (
        run_eval(eval_first_state, key_debug,
                 training_state.policy_optimizer.target,
                 training_state.normalizer_params))
    eval_state.total_episodes.block_until_ready()
    eval_walltime += time.time() - t
    eval_sps = (episode_length *
                eval_first_state.core.reward.shape[0] / (time.time() - t))
    metrics = dict(
        dict({
            f'eval/episode_{name}': value / eval_state.total_episodes
            for name, value in eval_state.total_metrics.items()
        }),
        **dict({
            f'training/{name}': onp.mean(value)
            for name, value in training_metrics.items()
        }),
        **dict({
            'eval/total_episodes': eval_state.total_episodes,
            'speed/sps': sps,
            'speed/eval_sps': eval_sps,
            'speed/training_walltime': training_walltime,
            'speed/eval_walltime': eval_walltime,
            'training/grad_updates': training_state.steps[0],
        }),
        )
    logging.info(metrics)
    if progress_fn:
      progress_fn(current_step, metrics)

    if current_step >= num_timesteps:
      break

    # Create an initialize the replay buffer.
    if current_step == 0:
      t = time.time()

      replay_buffer = ReplayBuffer(
          data=jnp.zeros((local_devices_to_use, max_replay_size,
                          obs_size * 2 + core_env.action_size + 1 + 1)),
          current_size=jnp.zeros((local_devices_to_use,), dtype=jnp.int32),
          current_position=jnp.zeros((local_devices_to_use,),
                                     dtype=jnp.int32))

      training_state, state, replay_buffer = init_replay_buffer(
          training_state, state, replay_buffer)
      training_walltime += time.time() - t

    t = time.time()
    # optimization
    training_state, state, replay_buffer, training_metrics = run_sac_training(
        training_state, state, replay_buffer)
    jax.tree_map(lambda x: x.block_until_ready(), training_metrics)
    sps = ((training_state.normalizer_params[0][0] * action_repeat -
            current_step) / (time.time() - t))
    training_walltime += time.time() - t

  normalizer_params = jax.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  policy_params = jax.tree_map(lambda x: x[0],
                               training_state.policy_optimizer.target)

  logging.info('total steps: %s', normalizer_params[0] * action_repeat)

  _, inference = make_params_and_inference_fn(core_env.observation_size,
                                              core_env.action_size,
                                              normalize_observations)
  params = normalizer_params, policy_params

  return (inference, params, metrics)


def make_params_and_inference_fn(observation_size, action_size,
                                 normalize_observations):
  """Creates params and inference function for the SAC agent."""
  obs_normalizer_params, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      normalize_observations)
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_model, _ = make_sac_networks(
      parametric_action_distribution.param_size,
      observation_size, action_size)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(policy_params, obs), key)
    return action

  params = (obs_normalizer_params, policy_model.init(jax.random.PRNGKey(0)))
  return params, inference_fn
