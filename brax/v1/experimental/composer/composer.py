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

"""Composer for environments.

ComponentEnv composes a scene from descriptions of the form below:

   composer = Composer(
    components=dict(
        agent1=dict(component='ant', pos=(0, 1, 0)),
        agent2=dict(component='ant', pos=(0, -1, 0)),
    ),
    edges=dict(ant1__ant2=dict(collide_type='full'),),
   )
   env = ComposeEnv(composer=composer)

 (more examples available in envs/ant_descs.py)

During loading an env through create(), it:
- loads an env_descs, a dictionary containing all args to Composer/ComposerEnv
  - pre-defined envs are defined in envs/
  - new envs can be registered through register_env() or register_lib()
  - example env_descs are in envs/ant_descs.py
  - example multi-agent RL envs (through agent_utils.py) are in envs/ma_descs.py
- creates components: loads and pieces together brax.Config()
    components defined in components/
    such as ant.py or ground.py
  - new components can be registered through register_component()
  - support multiple instances of the same component through comp_name's
  - each component requires: ROOT=root body, SYS_CONFIG=config in string form,
      TERM_FN=termination function of this component, COLLIDES=bodies that
      are allowed to collide, DEFAULT_OBSERVERS=a list of observers (
      see observers.py for references)
  - optionally, each component can specify a dictionary of reward functions
      as `reward_fns`. See reward_functions.py.
- creates edges: automatically create necessary edge information
    among 2+ components, such as collide_include's in brax.Config()
  - optionally edge information can be supplied,
      e.g. `collide_type`={'full', 'root', None} specifying full collisons,
      collision only between roots, or no collision between two components
  - optionally, each edge can specify a dictionary of reward functions
      as `reward_fns`. See reward_functions.py.
- sets reward as sum of all `reward_fns` defined in `components` and `edges`
- sets termination as any(termination_fn of each component)
- sets observation to concatenation of observations of each component defined
    by each component's `observers` argument
"""

import collections
from collections import OrderedDict as odict
import copy
import functools
import itertools
from typing import Dict, Any, Callable, Tuple, Optional, Union

import brax.v1 as brax
from brax.v1 import envs
from brax.v1.envs import Env
from brax.v1.envs import State
from brax.v1.envs import Wrapper
from brax.v1.envs import wrappers
from brax.v1.experimental.braxlines.common import sim_utils
from brax.v1.experimental.braxlines.envs import wrappers as braxlines_wrappers
from brax.v1.experimental.composer import agent_utils
from brax.v1.experimental.composer import component_editor
from brax.v1.experimental.composer import composer_utils
from brax.v1.experimental.composer import data_utils
from brax.v1.experimental.composer import envs as composer_envs
from brax.v1.experimental.composer import observers
from brax.v1.experimental.composer import reward_functions
from brax.v1.experimental.composer.components import load_component
from brax.v1.experimental.composer.components import register_default_components
from jax import numpy as jnp

assert_env_params = composer_envs.assert_env_params
inspect_env = composer_envs.inspect_env
list_env = composer_envs.list_env
register_env = composer_envs.register_env
register_lib = composer_envs.register_lib

register_default_components()
composer_envs.register_default_libs()

MetaData = collections.namedtuple('MetaData', [
    'components',
    'edges',
    'global_options',
    'config_str',
    'config_json',
    'extra_observers',
    'reward_features',
    'reward_fns',
    'agent_groups',
])


class Composer(object):
  """Compose a brax system."""

  def __init__(self,
               components: Dict[str, Dict[str, Any]],
               edges: Dict[str, Dict[str, Any]] = None,
               extra_observers: Tuple[observers.Observer] = (),
               add_ground: bool = True,
               agent_groups: Dict[str, Any] = None,
               global_options: Dict[str, Any] = None):
    components = copy.deepcopy(components)
    edges = copy.deepcopy(edges or {})
    global_options = copy.deepcopy(global_options or {})
    extra_observers = copy.deepcopy(extra_observers)
    reward_features = []
    reward_fns = odict()
    agent_groups = agent_groups or {}

    # load components
    if add_ground:
      components['ground'] = dict(component='ground')
    components = {
        name: load_component(**value) for name, value in components.items()
    }
    component_keys = sorted(components)
    components = odict([(k, components[k]) for k in component_keys])

    # set global
    global_options = dict(
        json=component_editor.json_global_options(**global_options))
    global_options['message_str'] = component_editor.json2message_str(
        global_options['json'])

    components_ = odict()
    for k in component_keys:
      v, new_v = components[k], {}
      # [start config editing] convert to json format for easy editing
      new_v['json'] = component_editor.message_str2json(v.pop('message_str'))

      # add comp_name's
      comp_name = k
      rename_fn = functools.partial(
          component_editor.json_concat_name, comp_name=comp_name)
      new_v['json'] = rename_fn(new_v['json'])
      new_v['collides'] = rename_fn(v.pop('collides'), force_add=True)
      new_v['root'] = rename_fn(v.pop('root'), force_add=True)

      # add frozen bodies option
      if v.pop('frozen', False):
        for b in new_v['json'].get('bodies', []):
          b['frozen'] = {'all': True}
        new_v['frozen'] = True

      # cache config properties
      new_v['bodies'] = [b['name'] for b in new_v['json'].get('bodies', [])]
      new_v['joints'] = [b['name'] for b in new_v['json'].get('joints', [])]
      new_v['actuators'] = [
          b['name'] for b in new_v['json'].get('actuators', [])
      ]
      new_v['comp_name'] = comp_name

      # [end config editing] convert back to str
      new_v['message_str'] = component_editor.json2message_str(new_v['json'])

      # set transform or not
      if 'pos' in v or 'quat' in v:
        new_v['transform'] = True
        new_v['pos'] = jnp.array(v.pop('pos', [0, 0, 0]), dtype='float')
        new_v['quat_origin'] = jnp.array(
            v.pop('quat_origin', [0, 0, 0]), dtype='float')
        new_v['quat'] = jnp.array(
            v.pop('quat', [1., 0., 0., 0.]), dtype='float')
      else:
        new_v['transform'] = False

      # add reward functions
      component_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(component_reward_fns.items()):
        name = component_editor.concat_name(name, comp_name)
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn, unwrapped_reward_fn = reward_functions.get_reward_fns(
            new_v, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            unwrapped_reward_fn)

      # add extra observers
      component_observers = v.pop('extra_observers', ())
      for observer_kwargs in component_observers:
        extra_observers += (observers.get_component_observers(
            new_v, **observer_kwargs),)

      assert all(kk in ('term_fn', 'observers', 'collide')
                 for kk in v), f'unused kwargs in components[{k}]: {v}'
      new_v.update(v)
      components_[k] = new_v
    del components

    edges_ = {}
    for k1, k2 in itertools.combinations(component_keys, 2):
      if k1 == k2:
        continue
      k1, k2 = sorted([k1, k2])  # ensure the name is always sorted in order
      edge_name = component_editor.concat_comps(k1, k2)
      v, new_v = edges.pop(edge_name, {}), {}
      v1, v2 = [components_[k] for k in [k1, k2]]

      # add reward functions
      edge_reward_fns = v.pop('reward_fns', {})
      for name, reward_kwargs in sorted(edge_reward_fns.items()):
        name = component_editor.concat_name(name, edge_name)
        assert name not in reward_fns, f'duplicate reward_fns {name}'
        reward_fn, unwrapped_reward_fn = reward_functions.get_reward_fns(
            v1, v2, **reward_kwargs)
        reward_fns[name] = reward_fn
        reward_features += reward_functions.get_observers_from_reward_fns(
            unwrapped_reward_fn)

      # add observers
      edge_observers = v.pop('extra_observers', ())
      for observer_kwargs in edge_observers:
        extra_observers += (observers.get_edge_observers(
            v1, v2, **observer_kwargs),)

      # [start config editing]
      collide_type = v.pop('collide_type', 'full')
      v_json = {}
      # add colliders
      # TODO: add conaffinity-based collide definitions
      if not all([vv.get('collide', True) for vv in [v1, v2]]):
        pass
      elif collide_type == 'full':
        v_json.update(
            component_editor.json_collides(v1['collides'], v2['collides']))
      elif collide_type == 'root':
        v_json.update(
            component_editor.json_collides([v1['root']], [v2['root']]))
      else:
        assert not collide_type, collide_type
      # [end config editing]
      if v_json:
        # convert back to str
        new_v['message_str'] = component_editor.json2message_str(v_json)
      else:
        new_v['message_str'] = ''
      new_v['json'] = v_json
      assert not v, f'unused kwargs in edges[{edge_name}]: {v}'
      edges_[edge_name] = new_v
    assert not edges, f'unused edges: {edges}'
    edge_keys = sorted(edges_.keys())
    edges_ = odict([(k, edges_[k]) for k in edge_keys])

    # merge all message strs
    message_str = ''
    for _, v in sorted(components_.items()):
      message_str += v.get('message_str', '')
    for _, v in sorted(edges_.items()):
      message_str += v.get('message_str', '')
    message_str += global_options.get('message_str', '')
    config_str = message_str
    config_json = component_editor.message_str2json(message_str)
    metadata = MetaData(
        components=components_,
        edges=edges_,
        global_options=global_options,
        config_str=config_str,
        config_json=config_json,
        extra_observers=extra_observers,
        reward_features=reward_features,
        reward_fns=reward_fns,
        agent_groups=agent_groups,
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
    for _, v in self.metadata.components.items():
      term_fn = v['term_fn']
      if term_fn:
        done = term_fn(done, sys, qp, info, v)
    return done

  def obs_fn(self, sys, qp: brax.QP, info: brax.Info):
    """Return observation as OrderedDict."""
    cached_obs_dict = {}
    obs_dict = odict()
    reward_features = odict()
    for _, v in self.metadata.components.items():
      for observer in v['observers']:
        obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                           cached_obs_dict, v)
        obs_dict = odict(list(obs_dict.items()) + list(obs_dict_.items()))
    for observer in self.metadata.extra_observers:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      obs_dict = odict(list(obs_dict.items()) + list(obs_dict_.items()))
    for observer in self.metadata.reward_features:
      obs_dict_ = observers.get_obs_dict(sys, qp, info, observer,
                                         cached_obs_dict, None)
      reward_features = odict(
          list(reward_features.items()) + list(obs_dict_.items()))
    return obs_dict, reward_features


class ComponentEnv(Env):
  """Make a brax Env fromc config/metadata for training and inference."""

  def __init__(self, composer: Composer, env_desc: Dict[str, Any]):
    self.observer_shapes = None
    self.composer = composer
    self.env_desc = env_desc
    self.metadata = composer.metadata
    super().__init__(config=self.composer.metadata.config_str)
    self.action_shapes = get_action_shapes(self.sys)
    # reward_shape = (num_agents,) if multi-agent else ()
    self.reward_shape = (len(
        self.metadata.agent_groups),) if self.metadata.agent_groups else ()
    assert self.observation_size  # ensure self.observer_shapes is set
    self.group_action_shapes = agent_utils.set_agent_groups(
        self.metadata, self.action_shapes, self.observer_shapes)

  @property
  def is_multiagent(self):
    return bool(self.metadata.agent_groups)

  @property
  def agent_names(self):
    if not self.is_multiagent:
      return ()
    else:
      return tuple(sorted(self.metadata.agent_groups))

  def reset(self, rng: jnp.ndarray) -> State:
    """Resets the environment to an initial state."""
    qpos = self.sys.default_angle()
    qvel = jnp.zeros((self.sys.num_joint_dof,))
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    qp = self.composer.reset_fn(self.sys, qp)
    info = self.sys.info(qp)
    obs_dict, _ = self._get_obs(qp, info)
    obs = data_utils.concat_array(obs_dict, self.observer_shapes)
    reward, done, score = jnp.zeros((3,) + self.reward_shape)
    reward_names = tuple(self.composer.metadata.reward_fns)
    if self.is_multiagent:  # multi-agent
      done = jnp.any(done, axis=-1)  # ensure done is a scalar
      reward_names += self.agent_names
    state_info = {}
    state_info['score'] = score
    state_info['rewards'] = odict([(k, jnp.zeros(())) for k in reward_names])
    state_info['scores'] = odict([(k, jnp.zeros(())) for k in reward_names])
    return State(
        qp=qp,
        obs=obs,
        reward=reward,
        done=done.astype(jnp.float32),
        info=state_info)

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    del normalizer_params, extra_params
    qp, info = self.sys.step(state.qp, action)
    obs_dict, reward_features = self._get_obs(qp, info)
    obs = data_utils.concat_array(obs_dict, self.observer_shapes)
    reward_tuple_dict = odict([
        (k, fn(action, reward_features))
        for k, fn in self.composer.metadata.reward_fns.items()
    ])
    state.info['rewards'] = odict([
        (k, v[0]) for k, v in reward_tuple_dict.items()
    ])
    state.info['scores'] = odict([
        (k, v[1]) for k, v in reward_tuple_dict.items()
    ])
    if self.is_multiagent:  # multi-agent
      reward, score, done = agent_utils.process_agent_rewards(
          self.metadata, reward_tuple_dict)
      state.info['rewards'] = odict(
          list(state.info['rewards'].items()) +
          [(k, v) for k, v in zip(self.agent_names, reward)])
      state.info['scores'] = odict(
          list(state.info['scores'].items()) +
          [(k, v) for k, v in zip(self.agent_names, score)])
    else:
      reward, done, score = jnp.zeros((3,))
      for r, s, d in reward_tuple_dict.values():
        reward += r
        score += s
        done = jnp.logical_or(done, d)
    done = self.composer.term_fn(done, self.sys, qp, info)
    state.info['score'] = score
    return state.replace(
        qp=qp, obs=obs, reward=reward, done=done.astype(jnp.float32))

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe."""
    obs_dict, reward_features = self.composer.obs_fn(self.sys, qp, info)
    if self.observer_shapes is None:
      self.observer_shapes = data_utils.get_array_shapes(
          obs_dict, batch_shape=())
    return obs_dict, reward_features


def get_action_shapes(sys):
  """Get action shapes."""
  actuator_names = sim_utils.get_names(sys.config, 'actuator')
  actuator_action_shapes = sim_utils.names2indices(
      sys.config, names=actuator_names, datatype='actuator')[1]
  force_names = sim_utils.get_names(sys.config, 'force')
  force_action_shapes = sim_utils.names2indices(
      sys.config, names=force_names, datatype='force')[1]
  action_shapes = odict([
      (k, dict(start=v[0], end=v[-1] + 1, size=len(v), shape=(len(v),)))
      for k, v in itertools.chain(actuator_action_shapes.items(), force_action_shapes.items())
  ])
  return action_shapes


def unwrap(env: Env):
  """Unwrap all Env wrappers."""
  while isinstance(env, Wrapper):  # unwrap wrappers
    env = env.unwrapped
  return env


def is_multiagent(env: Env):
  """If it supports multiagent RL."""
  env = unwrap(env)
  return env.is_multiagent if hasattr(env, 'is_multiagent') else False


def get_obs_dict_shape(env: Env):
  """Get observation (dictionary) shape."""
  env = unwrap(env)
  if isinstance(env, ComponentEnv):
    assert env.observation_size  # ensure env.observer_shapes is set
    return env.observer_shapes
  else:
    return (env.observation_size,)


def create(env_name: str = None,
           env_desc: Union[Dict[str, Any], Callable[..., Dict[str,
                                                              Any]]] = None,
           desc_edits: Dict[str, Any] = None,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  assert env_name or env_desc, 'env_name or env_desc must be supplied'
  env_desc = env_desc or {}
  desc_edits = desc_edits or {}
  if env_name in composer_envs.ENV_DESCS:
    desc = composer_envs.ENV_DESCS[env_name]
    if callable(desc):
      desc = desc(**kwargs)
    else:
      assert not kwargs, f'unused kwargs: {kwargs}'
    env_desc = dict(**env_desc, **desc)
    env_desc = composer_utils.edit_desc(env_desc, desc_edits)
    composer = Composer(**env_desc)
    env = ComponentEnv(composer=composer, env_desc=env_desc)
  elif env_desc:
    if callable(env_desc):
      env_desc = env_desc(**kwargs)
    else:
      assert not kwargs, f'unused kwargs: {kwargs}'
    env_desc = composer_utils.edit_desc(env_desc, desc_edits)
    composer = Composer(**env_desc)
    env = ComponentEnv(composer=composer, env_desc=env_desc)
  else:
    env = envs.create(
        env_name,
        episode_length=episode_length,
        action_repeat=action_repeat,
        auto_reset=auto_reset,
        batch_size=batch_size,
        **kwargs)
    return env  # type: ignore

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
              env_desc: Union[Dict[str, Any], Callable[..., Dict[str,
                                                                 Any]]] = None,
              desc_edits: Dict[str, Any] = None,
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
      desc_edits=desc_edits,
      episode_length=episode_length,
      action_repeat=action_repeat,
      auto_reset=auto_reset,
      batch_size=batch_size,
      **kwargs)
