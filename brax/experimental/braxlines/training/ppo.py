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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf

*This is branched from training/ppo.py, and will be folded back later.*
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from brax import envs
from brax.experimental.braxlines.training import env
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training import pmap
from brax.training import ppo
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: Params
  key: PRNGKey
  normalizer_params: Params


def compute_ppo_loss(
    models: Dict[str, Params],
    data: ppo.StepData,
    udata: ppo.StepData,
    rng: PRNGKey,
    parametric_action_distribution: distribution.ParametricDistribution,
    policy_apply: Any,
    value_apply: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    ppo_epsilon: float = 0.3,
    extra_loss_update_ratios: Optional[Dict[str, float]] = None,
    extra_loss_fns: Optional[Dict[str, Callable[[ppo.StepData],
                                                jnp.ndarray]]] = None):
  """Computes PPO loss."""
  policy_params, value_params = models['policy'], models['value']
  extra_params = models.get('extra', {})
  policy_logits = policy_apply(policy_params, data.obs[:-1])
  baseline = value_apply(value_params, data.obs)
  baseline = jnp.squeeze(baseline, axis=-1)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = baseline[-1]
  baseline = baseline[:-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.

  # already removed at data generation time
  # actions = actions[:-1]
  # logits = logits[:-1]

  rewards = data.rewards[1:] * reward_scaling
  truncation = data.truncation[1:]
  termination = data.dones[1:] * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.actions)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      data.logits, data.actions)

  vs, advantages = ppo.compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon,
                             1 + ppo_epsilon) * advantages

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  value_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + value_loss + entropy_loss

  # Additional losses
  extra_losses = {}
  if extra_loss_fns:
    for key, loss_fn in extra_loss_fns.items():
      loss, rng = loss_fn(data=data, udata=udata, rng=rng, params=extra_params)
      if extra_loss_update_ratios and key in extra_loss_update_ratios:
        # enable loss gradient p*100 percent of the time
        rng, key_update = jax.random.split(rng)
        p = extra_loss_update_ratios[key]
        b = jax.random.bernoulli(key_update, p=jnp.array(p))
        loss = jnp.where(b, loss, jax.lax.stop_gradient(loss))
      total_loss += loss
      extra_losses[key] = loss

  return total_loss, dict(
      extra_losses, **{
          'total_loss': total_loss,
          'policy_loss': policy_loss,
          'value_loss': value_loss,
          'entropy_loss': entropy_loss,
      })


def train(environment_fn: Callable[..., envs.Env],
          num_timesteps,
          episode_length: int,
          action_repeat: int = 1,
          num_envs: int = 1,
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 128,
          learning_rate=1e-4,
          entropy_cost=1e-4,
          discounting=0.9,
          seed=0,
          unroll_length=10,
          batch_size=32,
          num_minibatches=16,
          num_update_epochs=2,
          log_frequency=10,
          normalize_observations=False,
          reward_scaling=1.,
          progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
          parametric_action_distribution_fn: Optional[Callable[[
              int,
          ], distribution.ParametricDistribution]] = distribution
          .NormalTanhDistribution,
          make_models_fn: Optional[Callable[
              [int, int],
              Tuple[networks.FeedForwardModel]]] = networks.make_models,
          policy_params: Optional[Dict[str, jnp.ndarray]] = None,
          value_params: Optional[Dict[str, jnp.ndarray]] = None,
          extra_params: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
          extra_step_kwargs: bool = True,
          extra_loss_update_ratios: Optional[Dict[str, float]] = None,
          extra_loss_fns: Optional[Dict[str, Callable[[ppo.StepData],
                                                      jnp.ndarray]]] = None):
  """PPO training."""
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)

  # TODO: check key randomness
  key = jax.random.PRNGKey(seed)
  key, key_models, key_env, key_eval = jax.random.split(key, 4)
  # Make sure every process gets a different random key, otherwise they will be
  # doing identical work.
  key_env = jax.random.split(key_env, process_count)[process_id]
  key = jax.random.split(key, process_count)[process_id]
  # key_models should be the same, so that models are initialized the same way
  # for different processes

  key_envs = jax.random.split(key_env, local_devices_to_use)
  core_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_envs // local_devices_to_use // process_count,
      episode_length=episode_length)

  core_eval_env = environment_fn(
      action_repeat=action_repeat,
      batch_size=num_eval_envs,
      episode_length=episode_length)
  eval_first_state, eval_step_fn = env.wrap(core_eval_env, key_eval,
                                            extra_step_kwargs=extra_step_kwargs)

  parametric_action_distribution = parametric_action_distribution_fn(
      event_size=core_env.action_size)

  policy_model, value_model = make_models_fn(
      parametric_action_distribution.param_size, core_env.observation_size)
  key_policy, key_value = jax.random.split(key_models)

  optimizer = optax.adam(learning_rate=learning_rate)
  init_params = {'policy': policy_params or policy_model.init(key_policy),
                 'value': value_params or value_model.init(key_value),
                 'extra': extra_params}
  optimizer_state = optimizer.init(init_params)
  optimizer_state, init_params = pmap.bcast_local_devices(
      (optimizer_state, init_params), local_devices_to_use)

  tmp_env_states = []
  for key in key_envs:
    first_state, step_fn = env.wrap(
        core_env, key, extra_step_kwargs=extra_step_kwargs)
    tmp_env_states.append(first_state)
  first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                  *tmp_env_states)

  normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
      normalization.create_observation_normalizer(
          core_env.observation_size,
          normalize_observations,
          num_leading_batch_dims=2,
          pmap_to_devices=local_devices_to_use))

  key_debug = jax.random.PRNGKey(seed + 666)

  loss_fn = functools.partial(
      compute_ppo_loss,
      parametric_action_distribution=parametric_action_distribution,
      policy_apply=policy_model.apply,
      value_apply=value_model.apply,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      extra_loss_update_ratios=extra_loss_update_ratios,
      extra_loss_fns=extra_loss_fns)

  grad_loss = jax.grad(loss_fn, has_aux=True)

  def do_one_step_eval(carry, unused_target_t):
    state, policy_params, normalizer_params, extra_params, key = carry
    key, key_sample = jax.random.split(key)
    obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    logits = policy_model.apply(policy_params, obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions, normalizer_params, extra_params)
    return (nstate, policy_params, normalizer_params, extra_params, key), ()

  @jax.jit
  def run_eval(state, key, policy_params, normalizer_params, extra_params):
    policy_params = jax.tree_map(lambda x: x[0], policy_params)
    normalizer_params = jax.tree_map(lambda x: x[0], normalizer_params)
    extra_params = jax.tree_map(lambda x: x[0], extra_params)
    (state, _, _, _, key), _ = jax.lax.scan(
        do_one_step_eval,
        (state, policy_params, normalizer_params, extra_params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def do_one_step(carry, unused_target_t):
    state, normalizer_params, policy_params, extra_params, key = carry
    key, key_sample = jax.random.split(key)
    normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.core.obs)
    logits = policy_model.apply(policy_params, normalized_obs)
    actions = parametric_action_distribution.sample_no_postprocessing(
        logits, key_sample)
    postprocessed_actions = parametric_action_distribution.postprocess(actions)
    nstate = step_fn(state, postprocessed_actions, normalizer_params,
                     extra_params)
    return (nstate, normalizer_params, policy_params, extra_params,
            key), ppo.StepData(
                obs=state.core.obs,
                rewards=state.core.reward,
                dones=state.core.done,
                truncation=state.core.info['truncation'],
                actions=actions,
                logits=logits)

  def generate_unroll(carry, unused_target_t):
    state, normalizer_params, policy_params, extra_params, key = carry
    (state, _, _, _, key), data = jax.lax.scan(
        do_one_step,
        (state, normalizer_params, policy_params, extra_params, key), (),
        length=unroll_length)
    data = data.replace(
        obs=jnp.concatenate([data.obs,
                             jnp.expand_dims(state.core.obs, axis=0)]),
        rewards=jnp.concatenate(
            [data.rewards,
             jnp.expand_dims(state.core.reward, axis=0)]),
        dones=jnp.concatenate(
            [data.dones, jnp.expand_dims(state.core.done, axis=0)]),
        truncation=jnp.concatenate(
            [data.truncation, jnp.expand_dims(state.core.info['truncation'],
                                              axis=0)]))
    return (state, normalizer_params, policy_params, extra_params, key), data

  def update_model(carry, data_tuple):
    optimizer_state, params, key = carry
    data, udata = data_tuple
    key, key_loss = jax.random.split(key)
    loss_grad, metrics = grad_loss(params, data, udata, key_loss)
    loss_grad = jax.lax.pmean(loss_grad, axis_name='i')
    params_update, optimizer_state = optimizer.update(loss_grad,
                                                      optimizer_state)
    params = optax.apply_updates(params, params_update)

    return (optimizer_state, params, key), metrics

  def minimize_epoch(carry, unused_t):
    optimizer_state, params, data, udata, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)
    permutation = jax.random.permutation(key_perm, data.obs.shape[1])

    def convert_data(data, permutation):
      data = jnp.take(data, permutation, axis=1, mode='clip')
      data = jnp.reshape(data, [data.shape[0], num_minibatches, -1] +
                         list(data.shape[2:]))
      data = jnp.swapaxes(data, 0, 1)
      return data

    ndata = jax.tree_map(lambda x: convert_data(x, permutation), data)
    u_ndata = jax.tree_map(lambda x: convert_data(x, permutation), udata)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        update_model, (optimizer_state, params, key_grad), (ndata, u_ndata),
        length=num_minibatches)
    return (optimizer_state, params, data, udata, key), metrics

  def run_epoch(carry, unused_t):
    training_state, state = carry
    key_minimize, key_generate_unroll, new_key = jax.random.split(
        training_state.key, 3)
    (state, _, _, _, _), data = jax.lax.scan(
        generate_unroll,
        (state, training_state.normalizer_params,
         training_state.params['policy'],
         training_state.params.get('extra', {}), key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs)
    # make unroll first
    data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    data = jax.tree_map(
        lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)

    # Update normalization params and normalize observations.
    normalizer_params = obs_normalizer_update_fn(
        training_state.normalizer_params, data.obs[:-1])
    udata = data
    data = data.replace(
        obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

    (optimizer_state, params, _, _, _), metrics = jax.lax.scan(
        minimize_epoch, (training_state.optimizer_state, training_state.params,
                         data, udata, key_minimize),
        (),
        length=num_update_epochs)

    new_training_state = TrainingState(
        optimizer_state=optimizer_state, params=params,
        normalizer_params=normalizer_params, key=new_key)
    return (new_training_state, state), metrics

  num_epochs = num_timesteps // (
      batch_size * unroll_length * num_minibatches * action_repeat)

  def _minimize_loop(training_state, state):
    (training_state, state), losses = jax.lax.scan(
        run_epoch, (training_state, state), (),
        length=num_epochs // log_frequency)
    losses = jax.tree_map(jnp.mean, losses)
    return (training_state, state), losses

  minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

  inference = make_inference_fn(
      core_env.observation_size, core_env.action_size, normalize_observations,
      parametric_action_distribution_fn, make_models_fn)

  training_state = TrainingState(
      optimizer_state=optimizer_state,
      params=init_params,
      key=jnp.stack(jax.random.split(key, local_devices_to_use)),
      normalizer_params=normalizer_params)
  training_walltime = 0
  eval_walltime = 0
  sps = 0
  eval_sps = 0
  losses = {}
  state = first_state
  metrics = {}

  for it in range(log_frequency + 1):
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    if process_id == 0:
      eval_state, key_debug = (
          run_eval(eval_first_state, key_debug,
                   training_state.params['policy'],
                   training_state.normalizer_params,
                   training_state.params.get('extra', {})))
      eval_state.total_episodes.block_until_ready()
      eval_walltime += time.time() - t
      eval_sps = (
          episode_length * eval_first_state.core.reward.shape[0] /
          (time.time() - t))
      metrics = dict(
          **dict({
              f'eval/episode_{name}': value / eval_state.total_episodes
              for name, value in eval_state.total_metrics.items()
          }), **dict({f'losses/{k}': jnp.mean(v) for k, v in losses.items()}),
          **dict({
              'eval/total_episodes': eval_state.total_episodes,
              'speed/sps': sps,
              'speed/eval_sps': eval_sps,
              'speed/training_walltime': training_walltime,
              'speed/eval_walltime': eval_walltime,
              'speed/timestamp': training_walltime
          }))
      logging.info(metrics)
      if progress_fn:
        params = dict(
            normalizer=jax.tree_map(lambda x: x[0],
                                    training_state.normalizer_params),
            policy=jax.tree_map(lambda x: x[0],
                                training_state.params['policy']),
            extra=jax.tree_map(lambda x: x[0], training_state.params['extra']))
        progress_fn(
            int(training_state.normalizer_params[0][0]) * action_repeat,
            metrics, params)

    if it == log_frequency:
      break

    t = time.time()
    previous_step = training_state.normalizer_params[0][0]
    # optimization
    (training_state, state), losses = minimize_loop(training_state, state)
    jax.tree_map(lambda x: x.block_until_ready(), losses)
    sps = ((training_state.normalizer_params[0][0] - previous_step) /
           (time.time() - t)) * action_repeat
    training_walltime += time.time() - t

  # To undo the pmap.
  normalizer_params = jax.tree_map(lambda x: x[0],
                                   training_state.normalizer_params)
  policy_params = jax.tree_map(lambda x: x[0],
                               training_state.params['policy'])
  extra_params = jax.tree_map(lambda x: x[0],
                              training_state.params.get('extra', {}))

  logging.info('total steps: %s', normalizer_params[0] * action_repeat)

  params = dict(
      normalizer=normalizer_params, policy=policy_params, extra=extra_params)

  if process_count > 1:
    # Make sure all processes stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()

  return (inference, params, metrics)


def make_inference_fn(
    observation_size: int,
    action_size: int,
    normalize_observations: bool = False,
    parametric_action_distribution_fn: Optional[Callable[[
        int,
    ], distribution.ParametricDistribution]] = distribution
    .NormalTanhDistribution,
    make_models_fn: Optional[Callable[
        [int, int], Tuple[networks.FeedForwardModel]]] = networks.make_models,
    params: Dict[str, Dict[str, jnp.ndarray]] = None):
  """Creates params and inference function for the PPO agent."""
  _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
      observation_size, normalize_observations=normalize_observations)
  parametric_action_distribution = parametric_action_distribution_fn(
      event_size=action_size)
  policy_model, _ = make_models_fn(parametric_action_distribution.param_size,
                                   observation_size)

  def inference_fn(params, obs, key):
    normalizer_params, policy_params = params['normalizer'], params['policy']
    obs = obs_normalizer_apply_fn(normalizer_params, obs)
    action = parametric_action_distribution.sample(
        policy_model.apply(policy_params, obs), key)
    return action

  return inference_fn
