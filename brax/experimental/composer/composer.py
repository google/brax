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

"""Composer for environments.

ComponentEnv composes a scene from descriptions of the form below:

   composer = Composer(
    components=dict(
        ant1=dict(component='ant', pos=(0, 1, 0)),
        ant2=dict(component='ant', pos=(0, -1, 0)),
    ),
    edges=dict(ant1__ant2=dict(collide_type='full'),),
   )
   env = ComposeEnv(composer=composer)

 (more examples available in experimental/composer/env_descs.py)

During loading, it:
- creates components: loads and pieces together brax.Config()
    components defined in experimental/composer/components/
    such as ant.py or ground.py
  - support multiple instances of the same component through suffixes
  - each component requires: ROOT=root body, SYS_CONFIG=config in string form,
      TERM_FN=termination function of this component, COLLIDES=bodies that
      are allowed to collide, DEFAULT_OBSERVERS=a list of observers (
      see experimental/composer/observers.py for references)
  - optionally, each component can specify a dictionary of reward functions
      as `reward_fns`. See experimental/composer/reward_functions.py.
- creates edges: automatically create necessary edge information
    between components, such as collide_include's in brax.Config()
  - optionally edge information can be supplied,
      e.g. `collide_type`={'full', 'root', None} specifying full collisons,
      collision only between roots, or no collision between two components
  - optionally, each edge can specify a dictionary of reward functions
      as `reward_fns`. See experimental/composer/reward_functions.py.
- sets reward as sum of all `reward_fns` defined in `components` and `edges`
- sets termination as any(termination_fn of each component)
- sets observation to concatenation of observations of each component defined
    by each component's `observers` argument
"""
import collections
import copy
import functools
import itertools
from typing import Dict, Any, Callable, Tuple, Optional

import brax
from brax import envs
from brax.envs import Env
from brax.envs import State
from brax.envs import wrappers
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.braxlines.envs import wrappers as braxlines_wrappers
from brax.experimental.composer import component_editor
from brax.experimental.composer import env_descs
from brax.experimental.composer import observers
from brax.experimental.composer import reward_functions
from jax import numpy as jnp

MetaData = collections.namedtuple('MetaData', [
    'components', 'edges', 'global_options', 'config_str', 'config_json',
    'extra_observers', 'reward_features', 'reward_fns'
])


class Composer(object):
  """Compose a brax system."""

  def __init__(self,
               components: Dict[str, Dict[str, Any]],
               edges: Dict[str, Dict[str, Any]] = None,
               extra_observers: Tuple[observers.Observer] = (),
               add_ground: bool = True,
               global_options: Dict[str, Any] = None):
    components = copy.deepcopy(components)
    edges = copy.deepcopy(edges or {})
    extra_observers = copy.deepcopy(extra_observers)
    reward_features = []
    reward_fns = collections.OrderedDict()

    # load components
    if add_ground:
      components['ground'] = dict(component='ground')
    components = {
        name: component_editor.load_component(**value)
        for name, value in components.items()
    }
    component_keys = sorted(components.keys())
    components_ = collections.OrderedDict([
        (k, components[k]) for k in component_keys
    ])

    # set global
    v = dict(
        json=component_editor.json_global_options(**(global_options or {})))
    v['message_str'] = component_editor.json2message_str(v['json'])
    global_options_ = v

    for k, v in components_.items():
      # convert to json format for easy editing
      v['json'] = component_editor.message_str2json(v['message_str'])
      # add suffices
      suffix = v.get('suffix', k)
      if suffix:
        rename_fn = functools.partial(
            component_editor.json_add_suffix, suffix=suffix)
        v['json'] = rename_fn(v['json'])
        v['collides'] = rename_fn(v['collides'], force_add=True)
        v['root'] = rename_fn(v['root'], force_add=True)
      v['bodies'] = [b['name'] for b in v['json'].get('bodies', [])]
      v['joints'] = [b['name'] for b in v['json'].get('joints', [])]
      v['suffix'] = suffix
      # convert back to str
      v['message_str'] = component_editor.json2message_str(v['json'])

      # set transform or not
      if 'pos' in v or 'quat' in v:
        v['transform'] = True
        v['pos'] = jnp.array(v.get('pos', [0, 0, 0]), dtype='float')
        v['quat_origin'] = jnp.array(
            v.get('quat_origin', [0, 0, 0]), dtype='float')
        v['quat'] = jnp.array(v.get('quat', [1., 0., 0., 0.]), dtype='float')
      else:
        v['transform'] = False

      # add reward functions
      component_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(component_reward_fns.items()):
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn = reward_functions.get_component_reward_fns(
            v, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            reward_fn)

      component_observers = v.pop('extra_observers', ())
      for observer_kwargs in component_observers:
        extra_observers += (observers.get_component_observers(
            v, **observer_kwargs),)

    edges_ = {}
    for k1, k2 in itertools.combinations(list(components_.keys()), 2):
      if k1 == k2:
        continue
      k1, k2 = sorted([k1, k2])  # ensure the name is always sorted in order
      edge_name = f'{k1}__{k2}'
      v, new_v = edges.pop(edge_name, {}), {}
      v1, v2 = [components_[k] for k in [k1, k2]]

      # add reward functions
      edge_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(edge_reward_fns.items()):
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn = reward_functions.get_edge_reward_fns(
            v1, v2, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            reward_fn)

      # add observers
      edge_observers = v.pop('extra_observers', ())
      for observer_kwargs in edge_observers:
        extra_observers += (observers.get_edge_observers(
            v1, v2, **observer_kwargs),)

      collide_type = v.pop('collide_type', 'full')
      v_json = {}
      # add colliders
      if collide_type == 'full':
        v_json.update(
            component_editor.json_collides(v1['collides'], v2['collides']))
      elif collide_type == 'root':
        v_json.update(
            component_editor.json_collides([v1['root']], [v2['root']]))
      else:
        assert not collide_type, collide_type
      if v_json:
        # convert back to str
        new_v['message_str'] = component_editor.json2message_str(v_json)
      else:
        new_v['message_str'] = ''
      new_v['json'] = v_json
      assert not v, f'unused edges[{edge_name}]: {v}'
      edges_[edge_name] = new_v
    assert not edges, f'unused edges: {edges}'
    edge_keys = sorted(edges_.keys())
    edges_ = collections.OrderedDict([(k, edges_[k]) for k in edge_keys])

    # merge all message strs
    message_str = ''
    for _, v in sorted(components_.items()):
      message_str += v.get('message_str', '')
    for _, v in sorted(edges_.items()):
      message_str += v.get('message_str', '')
    message_str += global_options_.get('message_str', '')
    config_str = message_str
    config_json = component_editor.message_str2json(message_str)
    metadata = MetaData(
        components=components_,
        edges=edges_,
        global_options=global_options_,
        config_str=config_str,
        config_json=config_json,
        extra_observers=extra_observers,
        reward_features=reward_features,
        reward_fns=reward_fns,
    )
    config = component_editor.message_str2message(message_str)
    self.config, self.metadata = config, metadata

  def reset_fn(self, sys, qp: brax.QP):
    """Reset state."""
    # apply translations and rotations
    for _, v in sorted(self.metadata.components.items()):
      if v['transform']:
        _, _, mask = sim_utils.names2indices(sys.config, v['bodies'], 'body')
        qp = sim_utils.transform_qp(qp, mask[..., None], v['quat'],
                                    v['quat_origin'], v['pos'])
    return qp

  def term_fn(self, done: jnp.ndarray, sys, qp: brax.QP, info: brax.Info):
    """Termination."""
    for k, v in self.metadata.components.items():
      term_fn = v['term_fn']
      if term_fn:
        done = term_fn(done, sys, qp, info, k)
    return done

  def obs_fn(self, sys, qp: brax.QP, info: brax.Info):
    """Return observation as OrderedDict."""
    cached_obs_dict = {}
    obs_dict = collections.OrderedDict()
    reward_features = collections.OrderedDict()
    for _, v in self.metadata.components.items():
      for observer in v['observers']:
        obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                           cached_obs_dict, v)
        obs_dict = collections.OrderedDict(
            list(obs_dict.items()) + list(obs_dict_.items()))
    for observer in self.metadata.extra_observers:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      obs_dict = collections.OrderedDict(
          list(obs_dict.items()) + list(obs_dict_.items()))
    for observer in self.metadata.reward_features:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      reward_features = collections.OrderedDict(
          list(reward_features.items()) + list(obs_dict_.items()))
    return obs_dict, reward_features


class ComponentEnv(Env):
  """Make a brax Env fromc config/metadata for training and inference."""

  def __init__(self, composer: Composer, *args, **kwargs):
    self.observer_shapes = None
    self.composer = composer
    super().__init__(*args, config=self.composer.metadata.config_str, **kwargs)

  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    qp = self.composer.reset_fn(self.sys, qp)
    info = self.sys.info(qp)
    obs_dict, _ = self._get_obs(qp, info)
    obs = concat_obs(obs_dict, self.observer_shapes)
    reward, done = jnp.zeros(2)
    state_info = {}
    state_info['rewards'] = collections.OrderedDict([(k, jnp.zeros(
        ())) for k, _ in self.composer.metadata.reward_fns.items()])
    return State(qp=qp, obs=obs, reward=reward, done=done, info=state_info)

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    del normalizer_params, extra_params
    qp, info = self.sys.step(state.qp, action)
    obs_dict, reward_features = self._get_obs(qp, info)
    obs = concat_obs(obs_dict, self.observer_shapes)
    reward = jnp.zeros(())
    done = jnp.array(False)
    reward_done_dict = collections.OrderedDict([
        (k, fn(action, reward_features))
        for k, fn in self.composer.metadata.reward_fns.items()
    ])
    for r, d in reward_done_dict.values():
      reward += r
      done = jnp.logical_or(done, d)
    done = self.composer.term_fn(done, self.sys, qp, info)
    state.info['rewards'] = collections.OrderedDict([
        (k, v[0]) for k, v in reward_done_dict.items()
    ])
    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe."""
    obs_dict, reward_features = self.composer.obs_fn(self.sys, qp, info)
    if self.observer_shapes is None:
      self.observer_shapes = observers.get_obs_dict_shape(
          obs_dict, batch_shape=())
    return obs_dict, reward_features


def concat_obs(obs_dict: Dict[str, jnp.ndarray],
               observer_shapes: Dict[str, Dict[str, Any]]) -> jnp.ndarray:
  """Concatenate observation dictionary to a vector."""
  return jnp.concatenate([
      o.reshape(o.shape[:-len(s['shape'])] + (s['size'],))
      for o, s in zip(obs_dict.values(), observer_shapes.values())
  ],
                         axis=-1)


def split_obs(
    obs: jnp.ndarray,
    observer_shapes: Dict[str, Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
  """Split observation vector to a dictionary."""
  obs_leading_dims = obs.shape[:-1]
  return collections.OrderedDict([
      (k, obs[..., v['start']:v['end']].reshape(obs_leading_dims + v['shape']))
      for k, v in observer_shapes.items()
  ])


def get_env_obs_dict_shape(env: Env):
  """Gets an Env's observation shape(s)."""
  if isinstance(env, ComponentEnv):
    assert env.observation_size  # ensure env.observer_shapes is set
    return env.observer_shapes
  else:
    return (env.observation_size,)


def create(env_name: str = None,
           env_desc: Dict[str, Any] = None,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  assert env_name or env_desc, 'env_name or env_desc must be supplied'
  env_desc = env_desc or {}
  if env_name in env_descs.ENV_DESCS:
    composer = Composer(**env_desc, **env_descs.ENV_DESCS[env_name])
    env = ComponentEnv(composer=composer, **kwargs)
  elif env_desc:
    composer = Composer(**env_desc)
    env = ComponentEnv(composer=composer, **kwargs)
  else:
    env = envs.create(env_name, **kwargs)

  # add wrappers
  env = braxlines_wrappers.ExtraStepArgsWrapper(env)
  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)
  return env  # type: ignore


def create_fn(env_name: str = None,
              env_desc: Dict[str, Any] = None,
              episode_length: int = 1000,
              action_repeat: int = 1,
              auto_reset: bool = True,
              batch_size: Optional[int] = None,
              **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(
      create,
      env_name=env_name,
      env_desc=env_desc,
      episode_length=episode_length,
      action_repeat=action_repeat,
      auto_reset=auto_reset,
      batch_size=batch_size,
      **kwargs)
