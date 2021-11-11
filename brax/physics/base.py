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

"""Core brax structs and some conversion and slicing functions."""

from typing import Tuple

from brax import jumpy as jp
from brax import math
from brax.physics import config_pb2
from flax import struct


@struct.dataclass
class Q(object):
  """Coordinates: position and rotation.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
  """
  pos: jp.ndarray
  rot: jp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return QP(self.pos, self.rot, o.vel, o.ang)
    elif isinstance(o, Q):
      return Q(self.pos + o.pos, self.rot + o.rot)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, o.vel, o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")


@struct.dataclass
class P(object):
  """Time derivatives: velocity and angular velocity.

  Attributes:
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  vel: jp.ndarray
  ang: jp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return P(self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(o.pos, o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(o.pos, o.rot, self.vel + o.vel, self.ang + o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")

  def __mul__(self, o):
    return P(self.vel * o, self.ang * o)


@struct.dataclass
class QP(object):
  """A coordinate and time derivative frame for a brax body.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  pos: jp.ndarray
  rot: jp.ndarray
  vel: jp.ndarray
  ang: jp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return QP(self.pos, self.rot, self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel + o.vel,
                self.ang + o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")

  def __mul__(self, o):
    return QP(self.pos * o, self.rot * o, self.vel * o, self.ang * o)

  @classmethod
  def zero(cls, shape=()):
    return cls(
        pos=jp.zeros(shape + (3,)),
        rot=jp.tile(jp.array([1., 0., 0., 0]), reps=shape + (1,)),
        vel=jp.zeros(shape + (3,)),
        ang=jp.zeros(shape + (3,)))

  def to_world(self, rpos: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns world information about a point relative to a part.

    Args:
      rpos: Point relative to center of mass of part.

    Returns:
      A 2-tuple containing:
        * World-space coordinates of rpos
        * World-space velocity of rpos
    """
    rpos_off = math.rotate(rpos, self.rot)
    rvel = jp.cross(self.ang, rpos_off)
    return (self.pos + rpos_off, self.vel + rvel)

  def world_velocity(self, pos: jp.ndarray) -> jp.ndarray:
    """Returns the velocity of the point on a rigidbody in world space.

    Args:
      pos: World space position which to use for velocity calculation.
    """
    return self.vel + jp.cross(self.ang, pos - self.pos)


@struct.dataclass
class Info(object):
  """Auxilliary data calculated during the dynamics of each physics step.

  Attributes:
    contact: External contact forces applied at a step
    joint: Joint constraint forces applied at a step
    actuator: Actuator forces applied at a step
  """
  contact: P
  joint: P
  actuator: P


def validate_config(config: config_pb2.Config) -> config_pb2.Config:
  """Validate and normalize config settings for use in systems."""
  if config.dt <= 0:
    raise RuntimeError("config.dt must be positive")

  if config.substeps == 0:
    config.substeps = 1

  def find_dupes(objs):
    names = set()
    for obj in objs:
      if obj.name in names:
        raise RuntimeError(f"duplicate name in config: {obj.name}")
      names.add(obj.name)

  find_dupes(config.bodies)
  find_dupes(config.joints)
  find_dupes(config.actuators)

  # TODO: more config validation

  # reify all frozen dimensions in the system
  allvec = config_pb2.Vector3(x=1.0, y=1.0, z=1.0)
  frozen = config.frozen
  if frozen.all:
    frozen.position.CopyFrom(allvec)
    frozen.rotation.CopyFrom(allvec)
  if all([
      frozen.position.x, frozen.position.y, frozen.position.z,
      frozen.rotation.x, frozen.rotation.y, frozen.rotation.z
  ]):
    config.frozen.all = True
  for b in config.bodies:
    inertia = b.inertia
    if inertia.x == 0 and inertia.y == 0 and inertia.z == 0:
      b.inertia.x, b.inertia.y, b.inertia.z = 1, 1, 1

    b.frozen.position.x = b.frozen.position.x or frozen.position.x
    b.frozen.position.y = b.frozen.position.y or frozen.position.y
    b.frozen.position.z = b.frozen.position.z or frozen.position.z
    b.frozen.rotation.x = b.frozen.rotation.x or frozen.rotation.x
    b.frozen.rotation.y = b.frozen.rotation.y or frozen.rotation.y
    b.frozen.rotation.z = b.frozen.rotation.z or frozen.rotation.z
    if b.frozen.all:
      b.frozen.position.CopyFrom(allvec)
      b.frozen.rotation.CopyFrom(allvec)
    if all([
        b.frozen.position.x, b.frozen.position.y, b.frozen.position.z,
        b.frozen.rotation.x, b.frozen.rotation.y, b.frozen.rotation.z
    ]):
      b.frozen.all = True
  frozen.all = all(b.frozen.all for b in config.bodies)

  return config


def vec_to_arr(vec: config_pb2.Vector3) -> jp.ndarray:
  return jp.array([vec.x, vec.y, vec.z])
