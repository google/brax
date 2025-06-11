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

"""Physics pipeline for fully articulated dynamics and collisiion."""
# pylint:disable=g-multiple-import
# pylint:disable=g-importing-member
from typing import Optional
from brax.base import Contact, Motion, System, Transform
from brax.mjx.base import State
import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src.types import Contact as MJXContact


def _reformat_contact(sys: System, contact: MJXContact) -> Contact:
  """Reformats the mjx.Contact into a brax.base.Contact."""
  if contact is None:
    return

  elasticity = jp.zeros(contact.pos.shape[0])
  body1 = jp.array(sys.geom_bodyid)[contact.geom1] - 1
  body2 = jp.array(sys.geom_bodyid)[contact.geom2] - 1
  link_idx = (body1, body2)
  return Contact(link_idx=link_idx, elasticity=elasticity, **contact.__dict__)


def init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
    act: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    unused_debug: bool = False,
) -> State:
  """Initializes physics data.

  Args:
    sys: a brax System
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    act: actuator activations
    ctrl: actuator controls
    unused_debug: ignored

  Returns:
    data: initial physics data
  """

  data = mjx.make_data(sys)
  data = data.replace(qpos=q, qvel=qd)
  if act is not None:
    data = data.replace(act=act)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)

  data = mjx.forward(sys, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)

  brax_contact = _reformat_contact(sys, data.contact)
  data_args = data.__dict__
  data_args['contact'] = brax_contact

  return State(q=q, qd=qd, x=x, xd=xd, **data_args)


def step(
    sys: System, state: State, act: jax.Array, unused_debug: bool = False
) -> State:
  """Performs a single physics step using position-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: a brax System
    state: physics data prior to step
    act: (act_size,) actuator input vector
    unused_debug: ignored

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  data = state.replace(ctrl=act)
  data = mjx.step(sys, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)

  if data.ncon > 0:
    mjx_contact = data._impl.contact if hasattr(data, '_impl') else data.contact
    data = data.replace(contact=_reformat_contact(sys, mjx_contact))

  return data.replace(q=q, qd=qd, x=x, xd=xd)
