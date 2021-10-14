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

from typing import Tuple

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import actuators
from brax.physics import bodies
from brax.physics import colliders
from brax.physics import config_pb2
from brax.physics import integrators
from brax.physics import joints
from brax.physics import tree
from brax.physics.base import Info, P, QP, validate_config, vec_to_arr


@pytree.register
class System:
  """A brax system."""

  __pytree_ignore__ = ('config', 'num_bodies', 'num_joints', 'num_joint_dof',
                       'num_actuators')

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

  def default_qp(self, default_index: int = 0) -> QP:
    """Returns a default state for the system."""
    qps = {}

    # do we have default overrides in the config?
    default_angles, default_qps = {}, {}
    if default_index < len(self.config.defaults):
      defaults = self.config.defaults[default_index]
      default_qps = {qp.name: qp for qp in defaults.qps}
      default_angles = {angle.name: angle for angle in defaults.angles}

    # make the kinematic tree of bodies
    root = tree.Node.from_config(self.config)

    parent_joint = {j.child: j for j in self.config.joints}
    rot_axes = jp.vmap(math.rotate, [True, False])

    for body in root.depth_first():
      if body.name in default_qps:
        # set qp if found in defaults.qps override
        qp = default_qps[body.name]
        qps[body.name] = QP(
            pos=vec_to_arr(qp.pos),
            rot=math.euler_to_quat(vec_to_arr(qp.rot)),
            vel=vec_to_arr(qp.vel),
            ang=vec_to_arr(qp.ang))
      elif body.name in parent_joint:
        # pos/rot can be determined if the body is the child of a joint
        joint = parent_joint[body.name]
        rot = math.euler_to_quat(vec_to_arr(joint.rotation))
        axes = rot_axes(jp.eye(3), rot)
        if joint.name in default_angles:
          angle = default_angles[joint.name].angle
          angles = vec_to_arr(angle) * jp.pi / 180
        else:
          angles = [(l.min + l.max) * jp.pi / 360 for l in joint.angle_limit]

        # for each joint angle, rotate by that angle.
        # these are euler intrinsic rotations, so the axes are rotated too
        local_rot = jp.array([1., 0., 0., 0.])
        for axis, angle in zip(axes, angles):
          axis_rotated = math.rotate(axis, local_rot)
          next_rot = math.quat_rot_axis(axis_rotated, angle)
          local_rot = math.quat_mul(next_rot, local_rot)
        base_rot = math.euler_to_quat(vec_to_arr(joint.reference_rotation))
        local_rot = math.quat_mul(base_rot, local_rot)
        local_offset = math.rotate(vec_to_arr(joint.child_offset), local_rot)
        local_pos = vec_to_arr(joint.parent_offset) - local_offset

        parent_qp = qps[joint.parent]
        rot = math.quat_mul(parent_qp.rot, local_rot)
        pos = parent_qp.pos + math.rotate(local_pos, parent_qp.rot)
        qps[body.name] = QP(
            pos=pos, rot=rot, vel=jp.zeros(3), ang=jp.zeros(3))
      else:
        qps[body.name] = QP.zero()

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
      qp = self.integrator.potential(qp, dp_j + dp_a)

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
