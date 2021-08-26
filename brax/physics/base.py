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

from flax import struct
import jax
import jax.numpy as jnp
from brax.physics import config_pb2


@struct.dataclass
class Q(object):
  """Coordinates: position and rotation.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
  """
  pos: jnp.ndarray
  rot: jnp.ndarray

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
  vel: jnp.ndarray
  ang: jnp.ndarray

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
  pos: jnp.ndarray
  rot: jnp.ndarray
  vel: jnp.ndarray
  ang: jnp.ndarray

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
  def zero(cls):
    return cls(
        pos=jnp.zeros(3),
        rot=jnp.array([1., 0., 0., 0]),
        vel=jnp.zeros(3),
        ang=jnp.zeros(3))


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


def vec_to_np(v):
  return jnp.array([v.x, v.y, v.z])


def quat_to_np(q):
  return jnp.array([q.w, q.x, q.y, q.z])


def euler_to_quat(v):
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = jnp.cos(jnp.array([v.x, v.y, v.z]) * jnp.pi / 360)
  s1, s2, s3 = jnp.sin(jnp.array([v.x, v.y, v.z]) * jnp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return jnp.array([w, x, y, z])


def take(objects, i: jnp.ndarray, axis=0):
  """Returns objects sliced by i."""
  flat_data, py_tree_def = jax.tree_flatten(objects)
  sliced_data = [jnp.take(k, i, axis=axis, mode="clip") for k in flat_data]
  return jax.tree_unflatten(py_tree_def, sliced_data)


def validate_config(config: config_pb2.Config) -> config_pb2.Config:
  """Validate and normalize config settings for use in systems."""
  if config.dt <= 0:
    raise RuntimeError("config.dt must be positive")

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
