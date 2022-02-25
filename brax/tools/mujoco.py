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

"""Mujoco XML model loader for Brax."""

import dataclasses
from typing import Any, AnyStr, Dict, List, Optional

from absl import logging
from brax.physics import config_pb2
from dm_control import mjcf
from dm_control.mjcf import constants
import numpy as np
from transforms3d import euler
from transforms3d import quaternions
from transforms3d import utils as transform_utils

Vector = Any
Vector3d = Any
Quaternion = Any

# TODO: In Mujoco, the inertial of a body may be specified using an
# <inertial> element. We currently use a unit diagonal matrix instead.
DEFAULT_INERTIA = [1.0, 1.0, 1.0]
DEFAULT_STIFFNESS = 5000

# Tell the typechecker where to actually look for generic Element properties.
MjcfElement = mjcf.element._ElementImpl


@dataclasses.dataclass
class GeomCollider:
  """Represent a collider for a Mujoco geometry."""
  collider: config_pb2.Collider
  # Volume of the geometry in m^3.
  volume: float = 0
  # Mass of the geometry in kg.
  mass: float = 0


@dataclasses.dataclass
class Collision:
  """Collision information for a body."""
  # See http://www.mujoco.org/book/computation.html#Collision
  contype: int
  conaffinity: int
  # Name of the parent body.
  parent_name: str


def _is_worldbody(mujoco_body: MjcfElement) -> bool:
  """Returns true if the Mujoco body is the worldbody."""
  return mujoco_body.tag == constants.WORLDBODY


def _vec(v: Vector3d) -> config_pb2.Vector3:
  """Converts (numpy) array to Vector3."""
  x, y, z = v
  return config_pb2.Vector3(x=x, y=y, z=z)


def _np_vec(v: config_pb2.Vector3) -> Vector3d:
  """Converts a Vector3 to a numpy array."""
  return np.array([v.x, v.y, v.z])


def _euler(q: Optional[Quaternion]) -> config_pb2.Vector3:
  """Converts the quaternion to Euler angles in degrees."""
  if q is None:
    q = quaternions.qeye()
  return _vec(np.degrees(euler.quat2euler(q)))


def _maybe_qmult(q1: Optional[Quaternion],
                 q2: Optional[Quaternion]) -> Quaternion:
  """Returns the multiplication of two quaternions."""
  if q1 is not None:
    return quaternions.qmult(q1, q2) if q2 is not None else q1
  return q2


def _rot_quat(u: Vector3d, v: Vector3d) -> Quaternion:
  """Returns the quaternion performing rotation from u to v."""
  dot_p = np.dot(v, u)
  axis = np.cross(v, u)
  if not np.any(axis):
    return quaternions.qeye()
  norm_axis = np.linalg.norm(axis)
  angle = np.arctan2(norm_axis, dot_p)
  return quaternions.axangle2quat(axis, angle)


def _create_joint(
    name: str,
    parent_body_name: str,
    child_body_name: str,
    parent_offset: Optional[Vector3d] = None,
    child_offset: Optional[Vector3d] = None,
    axis: Optional[Vector3d] = None,
    stiffness: float = DEFAULT_STIFFNESS,
    joint_range: Optional[Vector] = None,
    rotation: Optional[Quaternion] = None,
    reference_rotation: Optional[Quaternion] = None) -> config_pb2.Joint:
  """Returns a (revolute) joint with the specified properties."""
  if axis is None:
    # Default axis of rotation.
    axis = [0, 0, 1]
  if rotation is not None:
    axis = quaternions.rotate_vector(axis, rotation)
  axis = transform_utils.normalized_vector(axis)
  angle_limit = (
      config_pb2.Joint.Range(min=joint_range[0], max=joint_range[1])
      if joint_range is not None else config_pb2.Joint.Range())
  rotation = _rot_quat(axis, [1.0, 0, 0])
  return config_pb2.Joint(
      name=name,
      parent=parent_body_name,
      child=child_body_name,
      parent_offset=_vec(parent_offset) if parent_offset is not None else None,
      child_offset=_vec(child_offset) if child_offset is not None else None,
      stiffness=stiffness,
      rotation=_euler(rotation),
      angle_limit=[angle_limit],
      reference_rotation=_euler(reference_rotation))


def _create_fixed_joint(name: str,
                        parent_body_name: str,
                        child_body_name: str,
                        parent_offset: Optional[Vector3d] = None,
                        rotation: Optional[Quaternion] = None,
                        reference_rotation: Optional[Quaternion] = None):
  """Returns a fixed joint."""
  # Brax does not support such joints. Instead we use a revolute joint with a
  # high stiffness and zero angle range.
  return _create_joint(
      name,
      parent_body_name,
      child_body_name,
      stiffness=DEFAULT_STIFFNESS,
      parent_offset=parent_offset,
      rotation=rotation,
      reference_rotation=reference_rotation)


class MujocoConverter(object):
  """Converts a Mujoco model to a Brax config."""

  def __init__(self,
               xml_string: AnyStr,
               add_collision_pairs: bool = False,
               ignore_unsupported_joints: bool = False,
               add_joint_to_nearest_body: bool = False):
    """Creates a MujocoConverter.

    Args:
      xml_string: XML string containing an Mujoco model description.
      add_collision_pairs: If true, then the collision pairs between bodies will
        be added automatically based on the structure of the model and Mujoco
        collision mask settings. See
        http://www.mujoco.org/book/computation.html#Collision.
      ignore_unsupported_joints: If true, then unsupported joints, e.g. slide,
        will be ignored, otherwise they raise an exception.
      add_joint_to_nearest_body: Adds a joint to the nearest (child)body when
        multiple geometries of a Mujoco body are represented as separate bodies.
    """
    mjcf_model = mjcf.from_xml_string(xml_string, escape_separators=True)
    self._mjcf_model = mjcf_model
    config = config_pb2.Config()
    self._config = config
    self._ignore_unsupported_joints = ignore_unsupported_joints
    self._add_joint_to_nearest_body = add_joint_to_nearest_body
    # Brax uses local coordinates. If global coordinates are used in the
    # Mujoco model, we convert them to local ones.
    self._uses_global = mjcf_model.compiler.coordinate == 'global'
    self._uses_radian = mjcf_model.compiler.angle == 'radian'
    default = mjcf_model.default
    geom = default.geom
    # See http://www.mujoco.org/book/XMLreference.html#geom. Mujoco uses SI
    # units, i.e. m(eter) for size, kg for mass and kg/m^3 for density.
    self._default_density = (
        geom.density if geom.density is not None else 1000.0)
    self._default_contype = geom.contype if geom.contype is not None else 1
    self._default_conaffinity = (
        geom.conaffinity if geom.conaffinity is not None else 1)
    joint = default.joint
    self._default_stiffness = (
        joint.stiffness if joint.stiffness is not None else DEFAULT_STIFFNESS)
    if joint.damping is not None:
      self._config.velocity_damping = joint.damping
    option = mjcf_model.option
    self._config.gravity.CopyFrom(
        _vec(option.gravity if option.gravity is not None else [0, 0, -9.81]))
    self._collisions: Dict[str, Collision] = {}

    # Worldbody is the root of the scene tree. We add the bodies starting from
    # the world body in a depth-first manner.
    self._add_body(mjcf_model.worldbody, None)
    # Add the actuators and the collision pairs.
    self._add_actuators()
    if add_collision_pairs:
      self._add_collision_pairs()

  @property
  def config(self) -> config_pb2.Config:
    """Returns the Brax config for the Mujoco model."""
    return self._config

  def _maybe_to_local(self, pos: Vector3d,
                      mujoco_body: MjcfElement) -> Vector3d:
    """Converts position to local coordinates."""
    if self._uses_global and mujoco_body and not _is_worldbody(mujoco_body):
      return pos - mujoco_body.pos
    return pos

  def _get_position(self, elem: MjcfElement) -> Vector3d:
    """Returns the local position of the Mujoco element, a geom or joint."""
    if elem.pos is None:
      return np.zeros(3)
    return self._maybe_to_local(elem.pos, elem.parent)

  def _maybe_to_radian(self, a: float) -> float:
    """Converts the angle to radian."""
    return a if self._uses_radian else np.radians(a)

  def _get_rotation(self, elem: MjcfElement) -> Quaternion:
    """Returns the rotation quaternion of the Mujoco element."""
    if _is_worldbody(elem):
      return quaternions.qeye()
    if elem.euler is not None:
      return euler.euler2quat(
          self._maybe_to_radian(elem.euler[0]),
          self._maybe_to_radian(elem.euler[1]),
          self._maybe_to_radian(elem.euler[2]))
    if elem.axisangle is not None:
      axisangle = elem.axisangle
      return quaternions.axangle2quat(axisangle[0:3],
                                      self._maybe_to_radian(axisangle[3]))
    return quaternions.qeye()

  def _get_mass(self, geom: MjcfElement, volume: float) -> float:
    """Returns the mass of the geometry based on its volume."""
    if geom.mass:
      return geom.mass
    density = geom.density if geom.density else self._default_density
    return volume * density

  def _get_contype(self, geom: MjcfElement) -> int:
    """Returns the contype of the geometry."""
    return geom.contype if geom.contype is not None else self._default_contype

  def _get_conaffinity(self, geom: MjcfElement) -> int:
    """Returns the conaffinity of the geometry."""
    return (geom.conaffinity
            if geom.conaffinity is not None else self._default_conaffinity)

  def _convert_capsule(self,
                       geom: MjcfElement,
                       add_radius: bool = True) -> GeomCollider:
    """Converts a capsule geometry to a collider."""
    size = geom.size
    radius = size[0]
    if geom.fromto is not None:
      start = geom.fromto[0:3]
      end = geom.fromto[3:6]
      direction = end - start
      length = transform_utils.vector_norm(direction)
      rotation = _rot_quat(direction, [0, 0, 1.0])
      position = self._maybe_to_local((start + end) / 2.0, geom.parent)
    else:
      if len(size) != 2:
        raise ValueError(f'Length is missing for {geom}')
      length = size[1] * 2
      position = self._get_position(geom)
      rotation = quaternions.qeye()
    rotation = _maybe_qmult(rotation, self._get_rotation(geom))
    if add_radius:
      length += 2 * radius
    volume = (np.pi * radius * radius * (length - 2.0 * radius / 3.0))
    mass = self._get_mass(geom, volume)
    return GeomCollider(
        collider=config_pb2.Collider(
            capsule=config_pb2.Collider.Capsule(radius=radius, length=length),
            rotation=_euler(rotation),
            position=_vec(position)),
        volume=volume,
        mass=mass)

  def _convert_cylinder(self, geom: MjcfElement) -> GeomCollider:
    """Converts a cylinder geometry to a collider."""
    return self._convert_capsule(geom, add_radius=False)

  def _convert_sphere(self, geom: MjcfElement) -> GeomCollider:
    """Converts a sphere geometry to a collider."""
    radius = geom.size[0]
    position = self._get_position(geom)
    volume = 4.0 * np.pi * np.power(radius, 3) / 3.0
    mass = self._get_mass(geom, volume)
    return GeomCollider(
        collider=config_pb2.Collider(
            sphere=config_pb2.Collider.Sphere(radius=radius),
            position=_vec(position)),
        volume=volume,
        mass=mass)

  def _convert_box(self, geom: MjcfElement) -> GeomCollider:
    """Converts a box geometry to a collider."""
    position = self._get_position(geom)
    rotation = self._get_rotation(geom)
    size = geom.size
    volume = 8.0 * np.prod(size)
    mass = self._get_mass(geom, volume)
    return GeomCollider(
        collider=config_pb2.Collider(
            box=config_pb2.Collider.Box(halfsize=_vec(size)),
            rotation=_euler(rotation),
            position=position),
        volume=volume,
        mass=mass)

  def _geom_to_collider(self, geom: MjcfElement) -> GeomCollider:
    """Converts a geometry to a collider."""
    if geom.type == 'capsule':
      geom_collider = self._convert_capsule(geom)
    elif geom.type == 'cylinder':
      geom_collider = self._convert_cylinder(geom)
    elif geom.type == 'sphere':
      geom_collider = self._convert_sphere(geom)
    elif geom.type == 'box':
      geom_collider = self._convert_box(geom)
    elif geom.type == 'plane':
      geom_collider = GeomCollider(
          collider=config_pb2.Collider(plane=config_pb2.Collider.Plane()))
    else:
      raise ValueError(f'Unsupported geom {geom.type}.')
    return geom_collider

  def _add_body(self, mujoco_body: MjcfElement, parent_body: Optional[config_pb2.Body]):
    """Adds the body, its children bodies and joints to the config."""
    config = self._config
    body_idx = len(config.bodies)
    body = config.bodies.add()
    if not parent_body:
      body.name = constants.WORLDBODY
      body.frozen.position.x, body.frozen.position.y, body.frozen.position.z = (
          1, 1, 1)
      body.frozen.rotation.x, body.frozen.rotation.y, body.frozen.rotation.z = (
          1, 1, 1)
    else:
      body.name = mujoco_body.name if mujoco_body.name else f'$body{body_idx}'
    logging.info('Body %s', body.name)
    geoms = mujoco_body.geom if hasattr(mujoco_body, 'geom') else []
    if not geoms:
      # Except worldbody, we expect the bodies to have a geometry. We add a
      # dummy one if that is not the case.
      geom = mujoco_body.add('geom', pos=[0, 0, 0], type='sphere', size=[0.01])
    else:
      # We add the first geometry to the body itself.
      geom = geoms[0]
    if geom.name and body.name == constants.WORLDBODY:
      # Using the name of the geometry is less confusing for worldbody. It won't
      # use referred to in joints or actuators.
      body.name = geom.name
    reference_rotation = self._get_rotation(mujoco_body)
    geom_collider = self._geom_to_collider(geom)
    body.mass = geom_collider.mass
    body.inertia.CopyFrom(_vec(DEFAULT_INERTIA))
    body.colliders.append(geom_collider.collider)
    geom_colliders = [geom_collider]
    self._collisions[body.name] = Collision(
        contype=self._get_contype(geom),
        conaffinity=self._get_conaffinity(geom),
        parent_name=parent_body.name if parent_body else '')
    # The remaining geometries are represented as child bodies.
    for idx, geom in enumerate(geoms[1:]):
      geom_name = geom.name if geom.name else f'${body.name}.{idx}'
      child_geom_collider = self._geom_to_collider(geom)
      config.bodies.append(
          config_pb2.Body(
              name=geom_name,
              colliders=[child_geom_collider.collider],
              mass=child_geom_collider.mass,
              inertia=_vec(DEFAULT_INERTIA)))
      config.joints.append(
          _create_fixed_joint(
              geom_name,
              body.name,
              geom_name,
              reference_rotation=reference_rotation))
      geom_colliders.append(child_geom_collider)
      # We use the same parent body name to ensure that multiple geometries of a
      # body do not collide with each other.
      self._collisions[geom_name] = Collision(
          contype=self._get_contype(geom),
          conaffinity=self._get_conaffinity(geom),
          parent_name=parent_body.name if parent_body else '')

    if not parent_body:
      # Worldbody cannot have joints.
      pass
    elif mujoco_body.joint:
      self._add_joints(mujoco_body, parent_body, geom_colliders,
                       config.bodies[body_idx:])
    else:
      # If there are no joints between the parent and child body, then this
      # means that they are fixed to each other.
      name = f'${parent_body.name}.{body.name}'
      parent_offset = mujoco_body.pos
      rotation = self._get_rotation(mujoco_body)
      config.joints.append(
          _create_fixed_joint(
              name,
              parent_body.name,
              body.name,
              parent_offset=parent_offset,
              rotation=rotation,
              reference_rotation=reference_rotation))

    for child_mujoco_body in mujoco_body.body:
      self._add_body(child_mujoco_body, body)

  def _add_joints(self, mujoco_body: MjcfElement, parent_body: config_pb2.Body,
                  geom_colliders: List[GeomCollider],
                  geom_bodies: List[config_pb2.Body]):
    """Adds the joints of the body to the config."""
    reference_rotation = self._get_rotation(mujoco_body)
    local_mujoco_body_pos = self._get_position(mujoco_body)
    for joint in mujoco_body.joint:
      if joint.type == 'free':
        continue
      if joint.type != 'hinge':
        if self._ignore_unsupported_joints:
          return
        raise ValueError(
            f'Unsupported joint type {joint.type} for {joint.name}')
      min_i = 0
      local_joint_pos = self._get_position(joint)
      if self._add_joint_to_nearest_body:
        # When a body has multiple geometries, all except the first one to are
        # moved to auxiliary child bodies (above), therefore we need to find the
        # one that the joint is actually connected with. For now, we use the
        # distance as a proxy.
        min_d = None
        for i, geom_collider in enumerate(geom_colliders):
          d = np.linalg.norm(local_joint_pos -
                             _np_vec(geom_collider.collider.position))
          if min_d is None or min_d > d:
            min_i, min_d = i, d
            logging.info('Distance to %s is %g.', geom_bodies[min_i].name,
                         min_d)
      attachment_body = geom_bodies[min_i]
      logging.info('Joint %s connects %s to %s.', joint.name, parent_body.name,
                   attachment_body.name)
      # The joint position is relative to the Mujoco body. To find the offset
      # with respect to the parent, we need to add the joint and Mujoco body
      # position (i.e. the relative position with respect to the reference point
      # of the parent) and then change the frame of reference to that of the COM
      # of the parent.
      parent_offset = local_mujoco_body_pos + local_joint_pos
      # The attachment geometry will preserve the locality, so it is sufficient
      # to change the frame of reference to its COM.
      child_offset = local_joint_pos
      stiffness = (
          joint.stiffness
          if joint.stiffness is not None else self._default_stiffness)
      self._config.joints.append(
          _create_joint(
              joint.name,
              parent_body.name,
              attachment_body.name,
              parent_offset,
              child_offset,
              axis=joint.axis,
              stiffness=stiffness,
              joint_range=joint.range,
              reference_rotation=reference_rotation))

  def _add_actuators(self):
    """Adds the actuators to the config."""
    for motor in self._mjcf_model.actuator.motor:
      joint_name = motor.joint.name
      self._config.actuators.append(
          config_pb2.Actuator(
              name=joint_name,
              joint=joint_name,
              strength=motor.gear.item(),
              angle=config_pb2.Actuator.Angle()))

  def _add_collision_pairs(self):
    """Adds body pairs that can collide with each other to the config."""
    num_bodies = len(self._config.bodies)
    for i in range(0, num_bodies - 1):
      body1 = self._config.bodies[i]
      col1 = self._collisions[body1.name]
      for j in range(i, num_bodies):
        body2 = self._config.bodies[j]
        col2 = self._collisions[body2.name]
        # See http://www.mujoco.org/book/computation.html#Collision for more
        # information about collision detection in Mujoco.
        if col1.parent_name == col2.parent_name:
          # Two bodies with the same parent cannot collide.
          continue
        if ((col1.parent_name == body2.name and col2.parent_name) or
            (col2.parent_name == body1.name and col1.parent_name)):
          # Parent and child bodies cannot collide unless parent is a geometry
          # of the worldbody.
          continue
        if ((col1.contype & col2.conaffinity) or
            (col2.contype & col1.conaffinity)):
          # Bodies should be compatible for collision.
          ci = self._config.collide_include.add()
          ci.first = body1.name
          ci.second = body2.name
