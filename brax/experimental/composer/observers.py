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

"""Observation functions.

get_obs_dict() supports modular observation specs, with observer=
  an Observer(): get the specified observation
    e.g.1. SimObserver(sim_datatype='body', sim_datacomp='vel',
      sim_dataname='root')
    e.g.2. LambdaObserver(observers=[obs1, obs2], fn=lambda x, y: x-y)
  'qp': includes all body info (pos, rot, vel, ang) of the component
  'root_joints': includes root body info and all joint info (pos, vel) of the
    component
  'root_z_joints': the same as 'root_joints' but remove root body's pos[:2]

index_preprocess() converts special specs, e.g. ('key1', 2), into int indices
  for index_obs()

index_obs() allows indexing with a list of indices
"""
import abc
import collections
from typing import Any, Dict, List, Tuple, Union
import brax
from brax.envs import Env
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.composer import component_editor
from brax.experimental.composer import composer
from jax import numpy as jnp


class Observer(abc.ABC):
  """Observer."""

  def __init__(self, name: str = None, indices: Tuple[int] = None):
    assert name
    self.name = name
    if isinstance(indices, int):
      indices = (indices,)
    self.indices = indices
    self.initialized = False

  def initialize(self, sys):
    del sys
    self.initialized = True

  def index_obs(self, obs: jnp.ndarray):
    if self.indices is not None:
      obs = obs[..., self.indices]
    return obs

  def get_obs(self, sys, qp: brax.QP, info: brax.Info,
              cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                       Any]):
    if not self.initialized:
      self.initialize(sys)
    if self.name in cached_obs_dict:
      return cached_obs_dict[self.name].copy()
    obs = self._get_obs(sys, qp, info, cached_obs_dict, component)
    obs = self.index_obs(obs)
    return obs

  @abc.abstractmethod
  def _get_obs(self, sys, qp: brax.QP, info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                        Any]):
    raise NotImplementedError

  def __str__(self) -> str:
    return self.name

  def __repr__(self) -> str:
    return self.name


LAMBDA_FN_MAPPING = {
    '-': lambda x, y: x - y,
}


class LambdaObserver(Observer):
  """LambdaObserver."""

  def __init__(self, observers: List[Observer], fn, **kwargs):
    self.observers = observers
    if isinstance(fn, str):
      fn = LAMBDA_FN_MAPPING[fn]
    self.fn = fn
    super().__init__(**kwargs)

  def initialize(self, sys):
    for o in self.observers:
      o.initialize(sys)
    super().initialize(sys)

  def _get_obs(self, sys, qp: brax.QP, info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                        Any]):
    obses = [
        o.get_obs(sys, qp, info, cached_obs_dict, component)
        for o in self.observers
    ]
    return self.fn(*obses)


class SimObserver(Observer):
  """SimObserver."""

  def __init__(self,
               sdtype: str = 'body',
               sdcomp: str = 'pos',
               sdname: str = '',
               comp_name: str = '',
               name: str = None,
               indices: Tuple[int] = None,
               **kwargs):
    sdname = component_editor.concat_name(sdname, comp_name)
    if not name:
      name = f'{sdtype}_{sdcomp}:{sdname}'
      if indices:
        name = f'{name}[{indices}]'
    self.sdtype = sdtype
    self.sdcomp = sdcomp
    self.sdname = sdname
    super().__init__(name=name, indices=indices, **kwargs)

  def initialize(self, sys):
    self.sim_indices, self.sim_info, self.sim_mask = sim_utils.names2indices(
        sys.config, names=[self.sdname], datatype=self.sdtype)
    self.sim_indices = self.sim_indices[0]  # list -> int
    super().initialize(sys)

  def _get_obs(self,
               sys,
               qp: brax.QP,
               info: brax.Info,
               cached_obs_dict: Dict[str, jnp.ndarray],
               component: Dict[str, Any] = None):
    """Get observation."""
    if self.sdtype == 'body':
      assert self.sdcomp in ('pos', 'rot', 'ang', 'vel'), self.sdcomp
      obs = getattr(qp, self.sdcomp)[self.sim_indices]
    elif self.sdtype == 'joint':
      joint_obs_dict = sim_utils.get_joint_value(sys, qp, self.sim_info)
      obs = list(joint_obs_dict.values())[0]
    elif self.sdtype == 'contact':
      assert self.sdcomp in ('vel', 'ang'), self.sdcomp
      v = getattr(info.contact, self.sdcomp)[self.sim_indices]
      v = jnp.clip(v, -1, 1)
      obs = jnp.reshape(v, v.shape[:-2] + (-1,))
    else:
      raise NotImplementedError(self.sdtype)
    return obs


def index_preprocess(indices: List[Any], env: Env = None) -> List[int]:
  """Preprocess indices to a list of ints and a list of str labels."""
  if indices is None:
    return None
  int_indices = []
  labels = []
  for index in indices:
    if isinstance(index, int):
      int_indices += [index]
      labels += [f'obs[{index}]']
    elif isinstance(index, tuple):
      assert len(index) == 2, 'tuple indexing is of form: (obs_dict_key, index)'
      key, i = index
      assert isinstance(env, composer.ComponentEnv), env
      assert env.observation_size  # ensure env.observer_shapes is set
      obs_shape = env.observer_shapes
      assert key in obs_shape, f'{key} not in {tuple(obs_shape.keys())}'
      int_indices += [obs_shape[key]['start'] + i]
      labels += [f'{key}[{i}]']
    else:
      raise NotImplementedError(index)
  return int_indices, labels


def index_obs(obs: jnp.ndarray, indices: List[Any], env: Env = None):
  """Index observation vector."""
  int_indices = index_preprocess(indices, env)
  return obs.take(int_indices, axis=-1)


def initialize_observers(observers: List[Union[Observer, str]], sys):
  """Initialize observers."""
  for o in observers:
    if isinstance(o, Observer):
      o.initialize(sys)


STRING_OBSERVERS = ('qp', 'root_joints', 'root_z_joints', 'cfrc')


def get_obs_dict(sys, qp: brax.QP, info: brax.Info, observer: Union[str,
                                                                    Observer],
                 cached_obs_dict: Dict[str, jnp.ndarray], component: Dict[str,
                                                                          Any]):
  """Observe."""
  obs_dict = collections.OrderedDict()
  if isinstance(observer, Observer):
    obs_dict[observer.name] = observer.get_obs(sys, qp, info, cached_obs_dict,
                                               component)
  elif observer == 'qp':
    # get all positions/orientations/velocities/ang velocities of all bodies
    bodies = component['bodies']
    indices = sim_utils.names2indices(sys.config, bodies, 'body')[0]
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      for index, b in zip(indices, bodies):
        v = getattr(qp, type_)[index]
        key = f'body_{type_}:{b}'
        obs_dict[key] = v
  elif observer in ('root_joints', 'root_z_joints'):
    # get all positions/orientations/velocities/ang velocities of root bodies
    root = component['root']
    index = sim_utils.names2indices(sys.config, root, 'body')[0][0]
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      v = getattr(qp, type_)[index]
      if observer == 'root_z_joints' and type_ == 'pos':
        # remove xy position
        v = v[2:]
      obs_dict[f'body_{type_}:{root}'] = v
    # get all joints
    joints = component['joints']
    _, joint_info, _ = sim_utils.names2indices(sys.config, joints, 'joint')
    joint_obs_dict = sim_utils.get_joint_value(sys, qp, joint_info)
    obs_dict = collections.OrderedDict(
        list(obs_dict.items()) + list(joint_obs_dict.items()))
  elif observer == 'cfrc':
    # external contact forces:
    # delta velocity (3,), delta ang (3,) * N bodies in the system
    bodies = component['bodies']
    indices = sim_utils.names2indices(sys.config, bodies, 'body')[0]
    for i, b in zip(indices, bodies):
      for type_ in ('vel', 'ang'):
        v = getattr(info.contact, type_)[i]
        v = jnp.clip(v, -1, 1)
        v = jnp.reshape(v, v.shape[:-2] + (-1,))
        key = f'contact_{type_}:{b}'
        obs_dict[key] = v
  else:
    raise NotImplementedError(observer)
  return obs_dict


def get_component_observers(component: Dict[str, Any],
                            observer_type: str = 'qp',
                            **observer_kwargs):
  """Get component-based Observers."""
  del component, observer_kwargs
  raise NotImplementedError(observer_type)


def get_edge_observers(component1: Dict[str, Any],
                       component2: Dict[str, Any],
                       observer_type: str = 'root_vec',
                       **observer_kwargs):
  """Get edge-based Observers."""
  if observer_type == 'root_vec':
    root1 = component1['root']
    root2 = component2['root']
    return LambdaObserver(
        name=f'dist__{root1}__{root2}',
        fn='-',
        observers=[
            SimObserver(sdname=component1['root'], **observer_kwargs),
            SimObserver(sdname=component2['root'], **observer_kwargs)
        ],
    )
  else:
    raise NotImplementedError(observer_type)
