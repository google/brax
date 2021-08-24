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

"""Observation functions.

Support modular observation space specifications:
  'qp': includes all body info (pos, rot, vel, ang) of the component
  'root_joints': includes root body info and all joint info (pos, vel) of the
    component
  'root_z_joints': the same as 'root_joints' but remove root body's pos[:2]

get_obs_dict_shape() returns shape info in the form:
  dict(key1=dict(shape=(10,), start=40, end=50), ...)

index_obs() allows indexing with a list of indices:
  index_obs(obs, [('key1', 2)], env) == obs[40+2:40+3]
"""
import collections
from typing import Any, Dict, List
import brax
from brax.envs import Env
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.composer import composer
from jax import numpy as jnp


def index_preprocess(indices: List[Any], env: Env = None) -> List[int]:
  """Preprocess indices to a list of ints."""
  if indices is None:
    return None
  int_indices = []
  for index in indices:
    if isinstance(index, int):
      int_indices += [index]
    elif isinstance(index, tuple):
      assert len(index) == 2, 'tuple indexing is of form: (obs_dict_key, index)'
      key, i = index
      assert isinstance(env, composer.ComponentEnv), env
      assert env.observation_size  # ensure env.observer_shapes is set
      obs_shape = env.observer_shapes
      int_indices += [obs_shape[key]['start'] + i]
    else:
      raise NotImplementedError(index)
  return int_indices


def index_obs(obs: jnp.ndarray, indices: List[Any], env: Env = None):
  """Index observation vector."""
  int_indices = index_preprocess(indices, env)
  return obs.take(int_indices, axis=-1)


def get_obs_dict(sys,
                 qp: brax.QP,
                 info: brax.Info,
                 observer: str = None,
                 component: Dict[str, Any] = None):
  """Observe."""
  obs_dict = collections.OrderedDict()
  if observer == 'qp':
    # get all positions/orientations/velocities/ang velocities of all bodies
    bodies = component['bodies']
    indices = sim_utils.names2indices(sys.config, bodies, 'body')[0]
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      for index, b in zip(indices, bodies):
        v = getattr(qp, type_)[index]
        key = f'{type_}:{b}'
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
      obs_dict[f'{type_}:{root}'] = v
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


def get_obs_dict_shape(obs_dict: Dict[str, jnp.ndarray]):
  observer_shapes = collections.OrderedDict()
  i = 0
  for k, v in obs_dict.items():
    assert v.ndim == 1, v.shape
    observer_shapes[k] = dict(shape=v.shape, start=i, end=i + v.shape[0])
    i += v.shape[0]
  return observer_shapes
