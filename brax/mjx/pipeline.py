# Copyright 2023 The Brax Authors.
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
from brax.base import Motion, Transform
from brax.mjx.base import State
import jax
from mujoco import mjx


def init(
    model: mjx.Model, q: jax.Array, qd: jax.Array, debug: bool = False
) -> State:
  """Initializes physics state.

  Args:
    model: an mjx.Model
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    debug: if True, adds contact to the state for debugging

  Returns:
    state: initial physics state
  """
  del debug  # ignored in mjx pipeline

  data = mjx.make_data(model)
  data = data.replace(qpos=q, qvel=qd)
  data = mjx.forward(model, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)
  contact = None

  return State(q, qd, x, xd, contact, data)


def step(
    model: mjx.Model, state: State, act: jax.Array, debug: bool = False
) -> State:
  """Performs a single physics step using position-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    model: an mjx.Model
    state: physics state prior to step
    act: (act_size,) actuator input vector
    debug: if True, adds contact to the state for debugging

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  del debug  # ignored in mjx pipeline

  data = state.data.replace(ctrl=act)
  data = mjx.step(model, data)

  q, qd = data.qpos, data.qvel
  x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
  cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
  offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
  offset = Transform.create(pos=offset)
  xd = offset.vmap().do(cvel)
  contact = None

  return State(q, qd, x, xd, contact, data)
