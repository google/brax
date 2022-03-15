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

"""URDF Model importer for Brax."""
# pytype: disable=attribute-error
# pytype: disable=unsupported-operands
from typing import AnyStr, Optional
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from brax.physics import config_pb2
from transforms3d import euler
from transforms3d import quaternions

Vector = np.ndarray
Quaternion = np.ndarray


def _rpy_to_ax_angle(rpy):
  if rpy:
    rpy = rpy.split()
    rpy = [float(a) for a in rpy]
    ax, ang = euler.euler2axangle(rpy[0], rpy[1], rpy[2], 'sxyz')
  else:
    ax = [1, 0., 0.]
    ang = 0.
  return ax, ang


def _relative_quat_from_parent(rpy, quat):
  ax, ang = _rpy_to_ax_angle(rpy)
  new_ax = quaternions.rotate_vector(ax, quat)
  new_quat = quaternions.axangle2quat(new_ax, ang, is_normalized=True)
  return new_quat


def _rpy_to_quat(rpy):
  if rpy:
    rpy = rpy.split()
    rpy = [float(a) for a in rpy]
    rpy = euler.euler2quat(*rpy, axes='sxyz')
  else:
    rpy = [1., 0., 0., 0.]
  return np.array(rpy)


def _srpy_to_rrpy(rpy):
  if rpy:
    rpy = rpy.split()
    rpy = [float(a) for a in rpy]
    rpy = euler.euler2quat(*rpy, axes='sxyz')
    rpy = euler.quat2euler(rpy, 'rxyz')
  else:
    rpy = [0., 0., 0.]
  return np.array(rpy) * 180 / np.pi


def _xyz_to_vec(xyz):
  if xyz:
    xyz = xyz.split()
    xyz = [float(a) for a in xyz]
  else:
    xyz = [0., 0., 0.]
  return np.array(xyz)


def _rot_quat(u: Vector, v: Vector) -> Quaternion:
  """Returns the quaternion performing rotation from u to v."""
  dot_p = np.dot(v, u)
  axis = np.cross(v, u)
  if not np.any(axis):
    return quaternions.qeye()
  norm_axis = np.linalg.norm(axis)
  angle = np.arctan2(norm_axis, dot_p)
  return quaternions.axangle2quat(axis, angle)


def _vec(v, scale=1.):
  """Converts (numpy) array to Vector3."""
  x, y, z = v
  return config_pb2.Vector3(x=x * scale, y=y * scale, z=z * scale)


def _construct_capsule(geom, pos, rot):
  """Converts a cylinder geometry to a collider."""

  radius = float(geom.get('radius'))
  length = float(geom.get('length'))
  length = length + 2 * radius
  return config_pb2.Collider(
      capsule=config_pb2.Collider.Capsule(radius=radius, length=length),
      rotation=_vec(euler.quat2euler(rot, 'rxyz'), scale=180 / np.pi),
      position=_vec(pos))


def _construct_sphere(geom, pos, rot):
  """Converts a sphere geometry to a collider."""
  radius = float(geom.get('radius'))
  return config_pb2.Collider(
      sphere=config_pb2.Collider.Sphere(radius=radius),
      rotation=_vec(euler.quat2euler(rot, 'rxyz'), scale=180 / np.pi),
      position=_vec(pos))


def _construct_box(geom, pos, rot):
  """Converts a box geometry to a collider."""
  size = _xyz_to_vec(geom.get('size'))
  return config_pb2.Collider(
      box=config_pb2.Collider.Box(halfsize=_vec(size / 2)),
      rotation=_vec(euler.quat2euler(rot, 'rxyz'), scale=180 / np.pi),
      position=_vec(pos))


def _construct_mesh(geom, pos, rot):
  """Converts a mesh stl file to a collider."""
  return config_pb2.Collider(
      mesh=config_pb2.Collider.Mesh(name=geom.get('filename'), scale=1.0),
      rotation=_vec(euler.quat2euler(rot, 'rxyz'), scale=180 / np.pi),
      position=_vec(pos))


_joint_type_to_limit = {'revolute': 1, 'universal': 2, 'spherical': 3}


class UrdfConverter(object):
  """Converts a URDF model to a Brax config."""

  def __init__(self, xml_string: AnyStr, add_collision_pairs: bool = False):
    ghum_xml = ET.fromstring(xml_string)
    self.body_tree = {}
    self.links = {}
    self.joints = {}
    self.config = config_pb2.Config()

    # construct link-name-to-link helper dictionary
    for link in ghum_xml.findall('link'):
      name = link.get('name')
      if name not in self.links:
        self.links[name] = link

    # construct body tree
    for joint in ghum_xml.findall('joint'):
      name = joint.get('name')
      self.joints[name] = joint
      parent = joint.find('parent').get('link')
      child = joint.find('child').get('link')
      if parent not in self.body_tree:
        self.body_tree[parent] = {'joints': [], 'parent': None}

      self.body_tree[parent]['joints'].append({'joint': name, 'child': child})

      if child not in self.body_tree:
        self.body_tree[child] = {'joints': [], 'parent': parent}
      else:
        self.body_tree[child]['parent'] = parent

    # find all roots and expand them to populate the config
    for node in self.body_tree:
      if self.body_tree[node]['parent'] is None:
        self.expand_node(node)

    # add actuators to all joints
    # TODO: make more general for multiple actuator types
    for j in self.config.joints:
      # note: this is a sensible default, and is not ingested from the urdf
      new_act = self.config.actuators.add()
      new_act.name = j.name
      new_act.joint = j.name
      new_act.strength = 100.0
      new_act.torque.SetInParent()

    # if no collision pairs, add an empty one to prevent auto-population
    if not add_collision_pairs:
      self.config.collide_include.add()

  def expand_node(self,
                  node: str,
                  cur_quat: Quaternion = np.array([1., 0., 0., 0.]),
                  fuse_to_parent: Optional[config_pb2.Body] = None,
                  cur_offset: Vector = np.array([0., 0., 0.])):
    """Traverses a tree of links connected by joints and adds them to a config.

    Args:
      node: The node currently being expanded
      cur_quat: The global rotation of the current node
      fuse_to_parent: Whether to fuse this link to its parent
      cur_offset: An optional positional offset for this node
    """
    colliders = self.links[node].findall('collision')
    if fuse_to_parent:
      body = fuse_to_parent
    else:
      body = self.config.bodies.add()
      body.name = node

    for c in colliders:

      try:
        col_rotation = _relative_quat_from_parent(
            c.find('origin').get('rpy'), cur_quat)
      except AttributeError:
        col_rotation = np.array([1., 0., 0., 0.])

      try:
        col_offset = _xyz_to_vec(c.find('origin').get('xyz'))
      except AttributeError:
        col_offset = np.array([0., 0., 0.])

      c_geom = c.find('geometry')

      col_total_rot = quaternions.qmult(col_rotation, cur_quat)
      col_global_offset = quaternions.rotate_vector(col_offset, cur_quat)

      if c_geom.find('sphere') is not None:
        body.colliders.append(
            _construct_sphere(
                c_geom.find('sphere'), col_global_offset + cur_offset,
                col_total_rot))
      elif c_geom.find('cylinder') is not None:
        body.colliders.append(
            _construct_capsule(
                c_geom.find('cylinder'), col_global_offset + cur_offset,
                col_total_rot))
      elif c_geom.find('capsule') is not None:
        body.colliders.append(
            _construct_capsule(
                c_geom.find('capsule'), col_global_offset + cur_offset,
                col_total_rot))
      elif c_geom.find('box') is not None:
        body.colliders.append(
            _construct_box(
                c_geom.find('box'), col_global_offset + cur_offset,
                col_total_rot))
      elif c_geom.find('mesh') is not None:
        body.colliders.append(
            _construct_mesh(
                c_geom.find('mesh'), col_global_offset + cur_offset,
                col_total_rot))
        filename = c_geom.find('mesh').get('filename')
        self.config.mesh_geometries.add(name=filename, path=filename)

      else:
        warnings.warn(f'No collider found on link {node}.')
    # TODO: load real mass and inertia
    body.mass += 1.
    body.inertia.x, body.inertia.y, body.inertia.z = 1., 1., 1.

    if self.body_tree[node]['joints']:
      for j in self.body_tree[node]['joints']:

        rpy_rotation = self.joints[j['joint']].find('origin').get('rpy')
        relative_rotation = _srpy_to_rrpy(rpy_rotation)
        rotation = np.array([1., 0., 0., 0.])

        offset = self.joints[j['joint']].find('origin').get('xyz')
        offset = _xyz_to_vec(offset)
        global_offset = quaternions.rotate_vector(offset, cur_quat)
        joint_type = self.joints[j['joint']].get('type')

        axis = self.joints[j['joint']].find('axis')
        axis = _xyz_to_vec(axis.get('xyz')) if axis is not None else np.array(
            [1., 0., 0.])
        axis_rotation = np.array(
            euler.quat2euler(_rot_quat(axis, np.array([1., 0., 0.])), 'rxyz'))
        axis_rotation = axis_rotation * 180 / np.pi

        if joint_type != 'fixed':
          joint = self.config.joints.add()
          joint.name = j['joint']
          if fuse_to_parent:
            joint.parent = fuse_to_parent.name
          else:
            joint.parent = node
          joint.parent_offset.x, joint.parent_offset.y, joint.parent_offset.z = global_offset[
              0], global_offset[1], global_offset[2],
          joint.child = j['child']

          # Note: These are sensible defaults and are not ingested from the urdf
          joint.stiffness = 10000.
          joint.rotation.x, joint.rotation.y, joint.rotation.z = axis_rotation
          joint.limit_strength = 300.
          joint.spring_damping = 50.
          joint.angular_damping = 10.
          joint.reference_rotation.x, joint.reference_rotation.y, joint.reference_rotation.z = relative_rotation

          # TODO: Load joint limit metadata
          if joint_type in _joint_type_to_limit:
            num_limits = _joint_type_to_limit[joint_type]
            for _ in range(num_limits):
              joint.angle_limit.add()

          self.expand_node(
              j['child'], cur_quat=quaternions.qmult(rotation, cur_quat))
        else:
          self.expand_node(
              j['child'],
              cur_quat=quaternions.qmult(rotation, cur_quat),
              fuse_to_parent=body,
              cur_offset=cur_offset + global_offset)


# pytype: enable=attribute-error
