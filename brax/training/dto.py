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

"""Direct trajectory optimization training.

Note: this module is untested.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from absl import logging
import flax
from flax import linen
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import distribution
from brax.training import env
from brax.training import networks


def train(
    environment_fn: Callable[..., envs.Env],
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    max_gradient_norm: float = 1e9,
    learning_rate=1e-4,
    seed=0,
    log_frequency=10,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
  """Direct trajectory optimization training."""
  xt = time.time()

  key = jax.random.PRNGKey(seed)
  key, key_models, key_env = jax.random.split(key, 3)

  create_env_fn = functools.partial(env.create_env,
                                    environment_fn,
                                    action_repeat=action_repeat,
                                    episode_length=episode_length,
                                    rng=key_env)

  first_state, step_fn, core_env = create_env_fn(num_envs)
  eval_first_state, eval_step_fn, _ = create_env_fn(num_eval_envs)
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=core_env.action_size)

  num_envs, obs_size = first_state.core.obs.shape

  policy_model = make_direct_optimization_model(parametric_action_distribution,
                                                obs_size)

  optimizer_def = flax.optim.GradientDescent(learning_rate=learning_rate)
  optimizer = optimizer_def.create(policy_model.init(key_models))

  key_debug = jax.random.PRNGKey(seed + 666)

  def do_one_step_eval(carry, unused_target_t):
    state, params, key = carry
    key, key_sample = jax.random.split(key)
    logits = policy_model.apply(params, state.core.obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = eval_step_fn(state, actions)
    return (nstate, params, key), ()

  @jax.jit
  def run_eval(params, state, key):
    (state, _, key), _ = jax.lax.scan(
        do_one_step_eval, (state, params, key), (),
        length=episode_length // action_repeat)
    return state, key

  def do_one_step(carry, unused_target_t):
    state, params, key = carry
    key, key_sample = jax.random.split(key)
    logits = policy_model.apply(params, state.core.obs)
    actions = parametric_action_distribution.sample(logits, key_sample)
    nstate = step_fn(state, actions)
    return (nstate, params, key), nstate.core.reward

  def loss(params, state, key):
    _, rewards = jax.lax.scan(do_one_step, (state, params, key), (),
                              length=episode_length // action_repeat)
    return -jnp.sum(rewards)

  loss_grad = jax.grad(loss)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    updates = jax.tree_multimap(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
        updates)
    return updates

  @jax.jit
  def minimize(optimizer, state, key):
    grad = loss_grad(optimizer.target, state, key)
    grad = clip_by_global_norm(grad)
    optimizer = optimizer.apply_gradient(grad)
    metrics = {'grad_norm': optax.global_norm(grad),
               'params_norm': optax.global_norm(optimizer.target)}
    return optimizer, key, metrics

  logging.info('Available devices %s', jax.devices())
  training_walltime = 0
  sps = 0
  eval_sps = 0
  summary = {'params_norm': optax.global_norm(optimizer.target)}

  for it in range(log_frequency + 1):
    logging.info('starting iteration %s %s', it, time.time() - xt)
    t = time.time()

    eval_state, key_debug = run_eval(optimizer.target, eval_first_state,
                                     key_debug)
    eval_state.total_episodes.block_until_ready()
    eval_sps = (
        episode_length * eval_first_state.core.reward.shape[0] /
        (time.time() - t))
    metrics = dict(
        dict({f'eval/episode_{name}': value / eval_state.total_episodes
              for name, value in eval_state.total_metrics.items()}),
        **dict({
            'eval/total_episodes': eval_state.total_episodes,
            'speed/sps': sps,
            'speed/eval_sps': eval_sps,
            'speed/training_walltime': training_walltime,
            'speed/timestamp': training_walltime,
            'train/grad_norm': summary.get('grad_norm', 0),
            'train/params_norm': summary.get('params_norm', 0),
        }))

    if progress_fn:
      progress_fn(it, metrics)

    if it == log_frequency:
      break

    t = time.time()
    # optimization
    optimizer, key, summary = minimize(optimizer, first_state, key)
    jax.tree_map(lambda x: x.block_until_ready(), summary)
    sps = (episode_length * num_envs) / (time.time() - t)
    training_walltime += time.time() - t

  params = optimizer.target
  _, inference = make_params_and_inference_fn(core_env.observation_size,
                                              core_env.action_size)

  return (inference, params, metrics)


def make_direct_optimization_model(parametric_action_distribution, obs_size):
  return networks.make_model(
      [32, parametric_action_distribution.param_size], obs_size,
      activation=linen.tanh)


def make_params_and_inference_fn(observation_size, action_size):
  """Creates params and inference function for the direct optimization agent."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_model = make_direct_optimization_model(parametric_action_distribution,
                                                observation_size)

  def inference_fn(params, obs, key):
    action = parametric_action_distribution.sample(
        policy_model.apply(params, obs), key)
    return action

  return policy_model.init(jax.random.PRNGKey(0)), inference_fn
