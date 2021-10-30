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

# pylint:disable=g-multiple-import
"""A brax system."""

from typing import Dict, Optional, Tuple

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import actuators
from brax.physics import bodies
from brax.physics import colliders
from brax.physics import config_pb2
from brax.physics import forces
from brax.physics import integrators
from brax.physics import joints
from brax.physics import tree
from brax.physics.base import Info, P, QP, validate_config, vec_to_arr


@pytree.register
class System:
  """A brax system."""

  __pytree_ignore__ = ('config', 'num_bodies', 'num_joints', 'num_joint_dof',
                       'num_actuators', 'num_forces_dof')

  def __init__(self, config: config_pb2.Config):
    self.config = validate_config(config)
    self.num_bodies = len(config.bodies)
    self.body = bodies.Body(config)
    self.integrator = integrators.Euler(self.config)
    self.colliders = colliders.get(self.config, self.body)
    self.num_joints = len(config.joints)
    self.joints = joints.get(self.config, self.body)
    self.num_actuators = len(config.actuators)
    self.num_joint_dof = sum(len(j.angle_limit) for j in config.joints)
    self.actuators = actuators.get(self.config, self.joints)
    self.forces = forces.get(self.config, self.body)
    self.num_forces_dof = sum(f.act_index.shape[-1] for f in self.forces)

  def default_angle(self, default_index: int = 0) -> jp.ndarray:
    """Returns the default joint angles for the system."""
    if not self.config.joints:
      return jp.array([])

    dofs = {j.name: len(j.angle_limit) for j in self.config.joints}
    angles = {}

    # check overrides in config defaults
    if default_index < len(self.config.defaults):
      defaults = self.config.defaults[default_index]
      for ja in defaults.angles:
        arr = vec_to_arr(ja.angle)[:dofs[ja.name]] * jp.pi / 180
        angles[ja.name] = arr

    # set remaining joint angles set from angle limits, and add jitter
    for joint in self.config.joints:
      if joint.name not in angles:
        angle = [(l.min + l.max) * jp.pi / 360 for l in joint.angle_limit]
        angles[joint.name] = jp.array(angle)

    return jp.concatenate([angles[j.name] for j in self.config.joints])

  def default_qp(
      self,
      default_index: int = 0,
      joint_angle: Optional[jp.ndarray] = None,
      joint_velocity: Optional[jp.ndarray] = None) -> QP:
    """Returns a default state for the system."""
    qps = {}
    if joint_angle is None:
      joint_angle = self.default_angle(default_index)
    if joint_velocity is None:
      joint_velocity = jp.zeros_like(joint_angle)

    # build dof lookup for each joint
    dofs_idx = {}
    dof_beg = 0
    for joint in self.config.joints:
      dof = len(joint.angle_limit)
      dofs_idx[joint.name] = (dof_beg, dof_beg + dof)
      dof_beg += dof

    # check overrides in config defaults
    default_qps = {}
    if default_index < len(self.config.defaults):
      defaults = self.config.defaults[default_index]
      default_qps = {qp.name for qp in defaults.qps}
      for qp in defaults.qps:
        qps[qp.name] = QP(
            pos=vec_to_arr(qp.pos),
            rot=math.euler_to_quat(vec_to_arr(qp.rot)),
            vel=vec_to_arr(qp.vel),
            ang=vec_to_arr(qp.ang))

    # make the kinematic tree of bodies
    root = tree.Node.from_config(self.config)

    parent_joint = {j.child: j for j in self.config.joints}
    rot_axes = jp.vmap(math.rotate, [True, False])

    for body in root.depth_first():
      if body.name in qps:
        continue
      if body.name not in parent_joint:
        qps[body.name] = QP.zero()
        continue
      # qp can be determined if the body is the child of a joint
      joint = parent_joint[body.name]
      dof_beg, dof_end = dofs_idx[joint.name]
      rot = math.euler_to_quat(vec_to_arr(joint.rotation))
      axes = rot_axes(jp.eye(3), rot)[:dof_end - dof_beg]
      angles = joint_angle[dof_beg:dof_end]
      velocities = joint_velocity[dof_beg:dof_end]
      # for each joint angle, rotate by that angle.
      # these are euler intrinsic rotations, so the axes are rotated too
      local_rot = math.euler_to_quat(vec_to_arr(joint.reference_rotation))
      for axis, angle in zip(axes, angles):
        local_axis = math.rotate(axis, local_rot)
        next_rot = math.quat_rot_axis(local_axis, angle)
        local_rot = math.quat_mul(next_rot, local_rot)
      local_offset = math.rotate(vec_to_arr(joint.child_offset), local_rot)
      local_pos = vec_to_arr(joint.parent_offset) - local_offset
      parent_qp = qps[joint.parent]
      rot = math.quat_mul(parent_qp.rot, local_rot)
      pos = parent_qp.pos + math.rotate(local_pos, parent_qp.rot)
      # TODO: propagate ang through tree and account for joint offset
      ang = jp.dot(axes.T, velocities).T
      qps[body.name] = QP(pos=pos, rot=rot, vel=jp.zeros(3), ang=ang)

    # any trees that have no body qp overrides in the config are set just above
    # the xy plane.  this convenience operation may be removed in the future.
    body_map = {body.name: body for body in self.config.bodies}
    for node in root.children:
      children = [node.name] + [n.name for n in node.depth_first()]
      if any(c in default_qps for c in children):
        continue  # ignore a tree if some part of it is overriden

      zs = jp.array([bodies.min_z(qps[c], body_map[c]) for c in children])
      min_z = jp.amin(zs)
      for body in children:
        qp = qps[body]
        pos = qp.pos - min_z * jp.array([0., 0., 1.])
        qps[body] = qp.replace(pos=pos)

    qps = [qps[body.name] for body in self.config.bodies]
    return jp.tree_map(lambda *args: jp.stack(args), *qps)

  def info(self, qp: QP) -> Info:
    """Return info about a system state."""
    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))

    dp_c = sum([c.apply(qp) for c in self.colliders], zero)
    dp_j = sum([j.apply(qp) for j in self.joints], zero)
    info = Info(dp_c, dp_j, zero)
    return info

  def step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
    """Calculates a physics step for a system, returns next state and info."""

    def substep(carry, _):
      qp, info = carry

      # apply kinetic step
      qp = self.integrator.kinetic(qp)

      # apply impulses arising from joints and actuators
      zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
      dp_j = sum([j.apply(qp) for j in self.joints], zero)
      dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
      dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
      qp = self.integrator.potential(qp, dp_j + dp_a + dp_f)

      # apply collision velocity updates
      dp_c = sum([c.apply(qp) for c in self.colliders], zero)
      qp = self.integrator.potential_collision(qp, dp_c)

      info = Info(info.contact + dp_c, info.joint + dp_j, info.actuator + dp_a)
      return (qp, info), ()

    # update collider statistics for culling
    for c in self.colliders:
      c.cull.update(qp)

    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
    info = Info(contact=zero, joint=zero, actuator=zero)
    (qp, info), _ = jp.scan(substep, (qp, info), (), self.config.substeps)
    return qp, info
