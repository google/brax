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

# pylint:disable=g-multiple-import
"""A brax system."""

from typing import Optional, Sequence, Tuple

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
from brax.physics import spring_joints
from brax.physics.base import Info, P, Q, QP, validate_config, vec_to_arr


@pytree.register
class System:
  """A brax system."""

  __pytree_ignore__ = ('config', 'num_bodies', 'num_joints', 'num_joint_dof',
                       'num_actuators', 'num_forces_dof')

  def __init__(self,
               config: config_pb2.Config,
               resource_paths: Optional[Sequence[str]] = None):
    config = validate_config(config, resource_paths=resource_paths)
    self.config = config
    self.num_bodies = len(config.bodies)
    self.body = bodies.Body(config)
    self.colliders = colliders.get(config, self.body)
    self.num_joints = len(config.joints)
    self.joints = joints.get(config, self.body) + spring_joints.get(
        config, self.body)
    self.num_actuators = len(config.actuators)
    self.num_joint_dof = sum(len(j.angle_limit) for j in config.joints)
    self.actuators = actuators.get(config, self.joints)
    self.forces = forces.get(config, self.body)
    self.num_forces_dof = sum(f.act_index.shape[-1] for f in self.forces)
    self.integrator = integrators.Euler(config)

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

  def default_qp(self,
                 default_index: int = 0,
                 joint_angle: Optional[jp.ndarray] = None,
                 joint_velocity: Optional[jp.ndarray] = None) -> QP:
    """Returns a default state for the system."""
    qp = QP.zero(shape=(self.num_bodies,))

    # set any default qps from the config
    default = None
    if default_index < len(self.config.defaults):
      default = self.config.defaults[default_index]
      for dqp in default.qps:
        body_i = self.body.index[dqp.name]
        pos = jp.index_update(qp.pos, body_i, vec_to_arr(dqp.pos))
        rot = jp.index_update(qp.rot, body_i,
                              math.euler_to_quat(vec_to_arr(dqp.rot)))
        vel = jp.index_update(qp.vel, body_i, vec_to_arr(dqp.vel))
        ang = jp.index_update(qp.ang, body_i, vec_to_arr(dqp.ang))
        qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)

    # build joints and joint -> array lookup, and order by depth
    if joint_angle is None:
      joint_angle = self.default_angle(default_index)
    if joint_velocity is None:
      joint_velocity = jp.zeros_like(joint_angle)
    joint_idxs = []
    for j in self.config.joints:
      beg = joint_idxs[-1][1][1] if joint_idxs else 0
      dof = len(j.angle_limit)
      joint_idxs.append((j, (beg, beg + dof)))
    lineage = {j.child: j.parent for j in self.config.joints}
    depth = {}
    for child, parent in lineage.items():
      depth[child] = 1
      while parent in lineage:
        parent = lineage[parent]
        depth[child] += 1
    joint_idxs = sorted(joint_idxs, key=lambda x: depth.get(x[0].parent, 0))
    joint = [j for j, _ in joint_idxs]

    if joint:
      # convert joint_angle and joint_vel to 3dof
      takes = []
      beg = 0
      for j, (beg, end) in joint_idxs:
        arr = list(range(beg, end))
        arr.extend([self.num_joint_dof] * (3 - len(arr)))
        takes.extend(arr)
      takes = jp.array(takes, dtype=int)

      def to_3dof(a):
        a = jp.concatenate([a, jp.array([0.0])])
        a = jp.take(a, takes)
        a = jp.reshape(a, (self.num_joints, 3))
        return a

      joint_angle = to_3dof(joint_angle)
      joint_velocity = to_3dof(joint_velocity)

      # build local rot and ang per joint
      joint_rot = jp.array(
          [math.euler_to_quat(vec_to_arr(j.rotation)) for j in joint])
      joint_ref = jp.array(
          [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joint])

      def local_rot_ang(_, x):
        angles, vels, rot, ref = x
        axes = jp.vmap(math.rotate, [True, False])(jp.eye(3), rot)
        ang = jp.dot(axes.T, vels).T
        rot = ref
        for axis, angle in zip(axes, angles):
          # these are euler intrinsic rotations, so the axes are rotated too:
          axis = math.rotate(axis, rot)
          next_rot = math.quat_rot_axis(axis, angle)
          rot = math.quat_mul(next_rot, rot)
        return (), (rot, ang)

      xs = (joint_angle, joint_velocity, joint_rot, joint_ref)
      _, (local_rot, local_ang) = jp.scan(local_rot_ang, (), xs, len(joint))

      # update qp in depth order
      joint_body = jp.array([
          (self.body.index[j.parent], self.body.index[j.child]) for j in joint
      ])
      joint_off = jp.array([(vec_to_arr(j.parent_offset),
                             vec_to_arr(j.child_offset)) for j in joint])

      def set_qp(carry, x):
        qp, = carry
        (body_p, body_c), (off_p, off_c), local_rot, local_ang = x
        world_rot = math.quat_mul(qp.rot[body_p], local_rot)
        local_pos = off_p - math.rotate(off_c, local_rot)
        world_pos = qp.pos[body_p] + math.rotate(local_pos, qp.rot[body_p])
        world_ang = math.rotate(local_ang, qp.rot[body_p])
        pos = jp.index_update(qp.pos, body_c, world_pos)
        rot = jp.index_update(qp.rot, body_c, world_rot)
        ang = jp.index_update(qp.ang, body_c, world_ang)
        qp = qp.replace(pos=pos, rot=rot, ang=ang)
        return (qp,), ()

      xs = (joint_body, joint_off, local_rot, local_ang)
      (qp,), () = jp.scan(set_qp, (qp,), xs, len(joint))

    # any trees that have no body qp overrides in the config are moved above
    # the xy plane.  this convenience operation may be removed in the future.
    fixed = {j.child for j in joint}
    if default:
      fixed |= {qp.name for qp in default.qps}
    root_idx = {
        b.name: [i]
        for i, b in enumerate(self.config.bodies)
        if b.name not in fixed
    }
    for j in joint:
      parent = j.parent
      while parent in lineage:
        parent = lineage[parent]
      if parent in root_idx:
        root_idx[parent].append(self.body.index[j.child])

    for children in root_idx.values():
      zs = jp.array([
          bodies.min_z(jp.take(qp, c), self.config.bodies[c]) for c in children
      ])
      min_z = jp.amin(zs)
      children = jp.array(children)
      pos = jp.take(qp.pos, children) - min_z * jp.array([0., 0., 1.])
      pos = jp.index_update(qp.pos, children, pos)
      qp = qp.replace(pos=pos)

    return qp

  def step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
    """Generic step function.  Overridden with appropriate step at init."""
    step_funs = {'pbd': self._pbd_step, 'legacy_spring': self._spring_step}
    return step_funs[self.config.dynamics_mode](qp, act)

  def info(self, qp: QP) -> Info:
    """Return info about a system state."""
    info_funs = {'pbd': self._pbd_info, 'legacy_spring': self._spring_info}
    return info_funs[self.config.dynamics_mode](qp)

  def _pbd_step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
    """Position based dynamics stepping scheme."""
    # Just like XPBD except performs two physics substeps per collision update.

    def substep(carry, _):
      qp, info = carry

      # first substep without collisions
      qprev = qp

      # apply acceleration updates for actuators, and forces
      zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
      zero_q = Q(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 4)))
      dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
      dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
      dp_j = sum([j.damp(qp) for j in self.joints], zero)
      qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

      # apply kinetic step
      qp = self.integrator.kinetic(qp)

      # apply joint position update
      dq_j = sum([j.apply(qp) for j in self.joints], zero_q)
      qp = self.integrator.update(qp, pos_q=dq_j)

      # apply pbd velocity projection
      qp = self.integrator.velocity_projection(qp, qprev)

      qprev = qp
      # second substep with collisions

      # apply acceleration updates for actuators, and forces
      dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
      dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
      dp_j = sum([j.damp(qp) for j in self.joints], zero)
      qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

      # apply kinetic step
      qp = self.integrator.kinetic(qp)

      # apply joint position update
      dq_j = sum([j.apply(qp) for j in self.joints], zero_q)
      qp = self.integrator.update(qp, pos_q=dq_j)

      collide_data = [c.position_apply(qp, qprev) for c in self.colliders]
      dq_c = sum([c[0] for c in collide_data], zero_q)
      dlambda = [c[1] for c in collide_data]
      contact = [c[2] for c in collide_data]
      qp = self.integrator.update(qp, pos_q=dq_c)

      # apply pbd velocity updates
      qp_right_before = qp
      qp = self.integrator.velocity_projection(qp, qprev)
      # apply collision velocity updates
      dp_c = sum([
          c.velocity_apply(qp, dlambda[i], qp_right_before, contact[i])
          for i, c in enumerate(self.colliders)
      ], zero)
      qp = self.integrator.update(qp, vel_p=dp_c)

      info = Info(info.contact + dp_c, info.joint, info.actuator + dp_a)
      return (qp, info), ()

    # update collider statistics for culling
    for c in self.colliders:
      c.cull.update(qp)

    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
    info = Info(contact=zero, joint=zero, actuator=zero)

    (qp, info), _ = jp.scan(substep, (qp, info), (), self.config.substeps // 2)
    return qp, info

  def _pbd_info(self, qp: QP) -> Info:
    """Return info about a system state."""
    zero_q = Q(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 4)))
    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))

    # TODO: sort out a better way to get first-step collider data
    dq_c = sum([c.apply(qp) for c in self.colliders], zero)
    dq_j = sum([j.apply(qp) for j in self.joints], zero_q)
    info = Info(dq_c, dq_j, zero)
    return info

  def _spring_step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
    """Spring-based dynamics stepping scheme."""
    # Resolves actuator forces, joints, and forces at acceleration level, and
    # resolves collisions at velocity level with baumgarte stabilization.

    def substep(carry, _):
      qp, info = carry

      # apply kinetic step
      qp = self.integrator.kinetic(qp)

      # apply acceleration level updates for joints, actuators, and forces
      zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
      dp_j = sum([j.apply(qp) for j in self.joints], zero)
      dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
      dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
      qp = self.integrator.update(qp, acc_p=dp_j + dp_a + dp_f)

      # apply velocity level updates for collisions
      dp_c = sum([c.apply(qp) for c in self.colliders], zero)
      qp = self.integrator.update(qp, vel_p=dp_c)

      info = Info(info.contact + dp_c, info.joint + dp_j, info.actuator + dp_a)
      return (qp, info), ()

    # update collider statistics for culling
    for c in self.colliders:
      c.cull.update(qp)

    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))
    info = Info(contact=zero, joint=zero, actuator=zero)

    (qp, info), _ = jp.scan(substep, (qp, info), (), self.config.substeps)
    return qp, info

  def _spring_info(self, qp: QP) -> Info:
    """Return info about a system state."""
    zero = P(jp.zeros((self.num_bodies, 3)), jp.zeros((self.num_bodies, 3)))

    dp_c = sum([c.apply(qp) for c in self.colliders], zero)
    dp_j = sum([j.apply(qp) for j in self.joints], zero)
    info = Info(dp_c, dp_j, zero)
    return info
