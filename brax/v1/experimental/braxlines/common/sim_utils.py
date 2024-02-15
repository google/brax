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

"""Simulator utilities."""

import collections
import functools
from typing import Any, List, Tuple

import brax.v1 as brax
from brax.v1 import jumpy as jp
import jax
from jax import numpy as jnp

lim_to_dof = {0: 1, 1: 1, 2: 2, 3: 3}


@functools.partial(jax.vmap, in_axes=[0, 0, None, None, None])
def transform_qp(qp, mask: jnp.ndarray, rot: jnp.ndarray, rot_vec: jnp.ndarray,
                 offset_vec: jnp.ndarray):
  """Rotates a qp by some rot around some ref_vec and translates it.

  Args:
    qp: QPs to be rotated
    mask: whether to transform this qp or not
    rot: Quaternion to rotate by
    rot_vec: point around which to rotate.
    offset_vec: relative displacement vector to translate qp by

  Returns:
    transformed QP
  """
  relative_pos = qp.pos - rot_vec
  new_pos = brax.math.rotate(relative_pos, rot) + rot_vec + offset_vec
  new_rot = brax.math.quat_mul(rot, qp.rot)
  return brax.physics.base.QP(
      pos=jnp.where(mask, new_pos, qp.pos),
      vel=qp.vel,
      ang=qp.ang,
      rot=jnp.where(mask, new_rot, qp.rot))


def get_names(config, datatype: str = 'body'):
  objs = {
      'body': config.bodies,
      'joint': config.joints,
      'actuator': config.actuators,
      'force': config.forces,
  }[datatype]
  return [b.name for b in objs]


def _legacy_angle_vel(joint, qp) -> Tuple[Any, Any]:
  """Returns joint angle and velocity using the legacy format.

  Args:
    joint: Joint to operate over
    qp: State data for system

  Returns:
    angle: n-tuple of joint angles where n = # DoF of the joint
    vel: n-tuple of joint velocities where n = # DoF of the joint
  """

  @jax.vmap
  def op(joint, qp_p, qp_c):
    axes, angles = joint.axis_angle(qp_p, qp_c)
    vels = tuple([jnp.dot(qp_p.ang - qp_c.ang, axis) for axis in axes])
    return angles, vels

  qp_p = jp.take(qp, joint.body_p.idx)
  qp_c = jp.take(qp, joint.body_c.idx)
  angles, vels = op(joint, qp_p, qp_c)

  return angles, vels


def get_joint_value(sys, qp, info: collections.OrderedDict):
  """Get joint values."""
  values = collections.OrderedDict()
  angles_vels = {j.dof: _legacy_angle_vel(j, qp) for j in sys.joints}
  for k, v in info.items():
    for i, type_ in zip((0, 1), ('pos', 'vel')):
      vvv = jnp.array([vv[v['index']] for vv in angles_vels[v['dof']][i]])
      values[f'joint_{type_}:{k}'] = vvv
  return values


def _check_active_dofs(joint):
  active_dof = 0
  for l in joint.angle_limit:
    if not (l.min == 0 and l.max == 0):
      active_dof += 1
  return active_dof


def names2indices(config, names: List[str], datatype: str = 'body'):
  """Convert name string to indices for indexing simulator states."""

  if isinstance(names, str):
    names = [names]

  indices = {}
  info = {}

  objs = {
      'body': config.bodies,
      'joint': config.joints,
      'actuator': config.actuators,
      'force': config.forces,
  }[datatype]
  joint_counters = [0, 0, 0]
  actuator_counter = 0
  force_counter = 0
  for i, b in enumerate(objs):
    if datatype == 'joint':
      if config.dynamics_mode == 'pbd':
        dof = _check_active_dofs(b)
      else:
        dof = lim_to_dof[len(b.angle_limit)]
    elif datatype == 'actuator':
      joint = [j for j in config.joints if j.name == b.joint][0]
      if config.dynamics_mode == 'pbd':
        dof = _check_active_dofs(joint)
      else:
        dof = lim_to_dof[len(joint.angle_limit)]
    elif datatype == 'force':
      dof = 3
    if b.name in names:
      indices[b.name] = i
      if datatype in ('joint',):
        info[b.name] = dict(dof=dof, index=joint_counters[dof - 1])
      if datatype in ('actuator',):
        info[b.name] = tuple(range(actuator_counter, actuator_counter + dof))
      if datatype in ('force',):
        info[b.name] = tuple(range(force_counter, force_counter + dof))
    if datatype in ('joint',):
      joint_counters[dof - 1] += 1
    if datatype in ('actuator',):
      actuator_counter += dof
    if datatype in ('force',):
      force_counter += dof

  indices = [indices[n] for n in names]
  mask = jnp.array([b.name in names for b in objs])

  if datatype in ('actuator', 'force', 'joint'):
    info = collections.OrderedDict([(k, info[k]) for k in names])

  return indices, info, mask
