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

# pylint:disable=g-multiple-import, g-importing-member
"""Base brax primitives and simple manipulations of them."""

import copy
import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from brax import math
from flax import struct
import jax
from jax import numpy as jp
from jax import vmap
from jax.tree_util import tree_map
import mujoco
from mujoco import mjx
import numpy as np

# f: free, 1: 1-dof, 2: 2-dof, 3: 3-dof
Q_WIDTHS = {'f': 7, '1': 1, '2': 2, '3': 3}
QD_WIDTHS = {'f': 6, '1': 1, '2': 2, '3': 3}


@struct.dataclass
class Base:
  """Base functionality extending all brax types.

  These methods allow for brax types to be operated like arrays/matrices.
  """

  def __add__(self, o: Any) -> Any:
    return tree_map(lambda x, y: x + y, self, o)

  def __sub__(self, o: Any) -> Any:
    return tree_map(lambda x, y: x - y, self, o)

  def __mul__(self, o: Any) -> Any:
    return tree_map(lambda x: x * o, self)

  def __neg__(self) -> Any:
    return tree_map(lambda x: -x, self)

  def __truediv__(self, o: Any) -> Any:
    return tree_map(lambda x: x / o, self)

  def reshape(self, shape: Sequence[int]) -> Any:
    return tree_map(lambda x: x.reshape(shape), self)

  def select(self, o: Any, cond: jax.Array) -> Any:
    return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

  def slice(self, beg: int, end: int) -> Any:
    return tree_map(lambda x: x[beg:end], self)

  def take(self, i, axis=0) -> Any:
    return tree_map(lambda x: jp.take(x, i, axis=axis, mode='wrap'), self)

  def concatenate(self, *others: Any, axis: int = 0) -> Any:
    return tree_map(lambda *x: jp.concatenate(x, axis=axis), self, *others)

  def index_set(
      self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
  ) -> Any:
    return tree_map(lambda x, y: x.at[idx].set(y), self, o)

  def index_sum(
      self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
  ) -> Any:
    return tree_map(lambda x, y: x.at[idx].add(y), self, o)

  def vmap(self, in_axes=0, out_axes=0):
    """Returns an object that vmaps each follow-on instance method call."""

    # TODO: i think this is kinda handy, but maybe too clever?

    outer_self = self

    class VmapField:
      """Returns instance method calls as vmapped."""

      def __init__(self, in_axes, out_axes):
        self.in_axes = [in_axes]
        self.out_axes = [out_axes]

      def vmap(self, in_axes=0, out_axes=0):
        self.in_axes.append(in_axes)
        self.out_axes.append(out_axes)
        return self

      def __getattr__(self, attr):
        fun = getattr(outer_self.__class__, attr)
        # load the stack from the bottom up
        vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
        for in_axes, out_axes in vmap_order:
          fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
        fun = functools.partial(fun, outer_self)
        return fun

    return VmapField(in_axes, out_axes)

  def tree_replace(
      self, params: Dict[str, Optional[jax.typing.ArrayLike]]
  ) -> 'Base':
    """Creates a new object with parameters set.

    Args:
      params: a dictionary of key value pairs to replace

    Returns:
      data clas with new values

    Example:
      If a system has 3 links, the following code replaces the mass
      of each link in the System:
      >>> sys = sys.tree_replace(
      >>>     {'link.inertia.mass', jp.array([1.0, 1.2, 1.3])})
    """
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split('.'), v)
    return new

  @property
  def T(self):  # pylint:disable=invalid-name
    return tree_map(lambda x: x.T, self)


def _tree_replace(
    base: Base,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Base:
  """Sets attributes in a struct.dataclass with values."""
  if not attr:
    return base

  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    lst = copy.deepcopy(getattr(base, attr[0]))

    for i, g in enumerate(lst):
      if not hasattr(g, attr[1]):
        continue
      v = val if not hasattr(val, '__iter__') else val[i]
      lst[i] = _tree_replace(g, attr[1:], v)

    return base.replace(**{attr[0]: lst})

  if len(attr) == 1:
    return base.replace(**{attr[0]: val})

  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )


@struct.dataclass
class Transform(Base):
  """Transforms the position and rotation of a coordinate frame.

  Attributes:
    pos: (3,) position transform of the coordinate frame
    rot: (4,) quaternion rotation the coordinate frame
  """

  pos: jax.Array
  rot: jax.Array

  def do(self, o):
    """Apply the transform."""
    return _transform_do(o, self)

  def inv_do(self, o):
    """Apply the inverse of the transform."""
    return _transform_inv_do(o, self)

  def to_local(self, t: 'Transform') -> 'Transform':
    """Move transform into basis of t."""
    pos = math.rotate(self.pos - t.pos, math.quat_inv(t.rot))
    rot = math.quat_mul(math.quat_inv(t.rot), self.rot)
    return Transform(pos=pos, rot=rot)

  @classmethod
  def create(
      cls, pos: Optional[jax.Array] = None, rot: Optional[jax.Array] = None
  ) -> 'Transform':
    """Creates a transform with either pos, rot, or both."""
    if pos is None and rot is None:
      raise ValueError('must specify either pos or rot')
    elif pos is None and rot is not None:
      pos = jp.zeros(rot.shape[:-1] + (3,))
    elif rot is None and pos is not None:
      rot = jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), pos.shape[:-1] + (1,))
    return Transform(pos=pos, rot=rot)

  @classmethod
  def zero(cls, shape=()) -> 'Transform':
    """Returns a zero transform with a batch shape."""
    pos = jp.zeros(shape + (3,))
    rot = jp.tile(jp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
    return Transform(pos, rot)


@struct.dataclass
class Motion(Base):
  """Spatial motion vector describing linear and angular velocity.

  More on spatial vectors: http://royfeatherstone.org/spatial/v2/index.html

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jax.Array
  vel: jax.Array

  def cross(self, other):
    return _motion_cross(other, self)

  def dot(self, m: Union['Motion', 'Force']) -> jax.Array:
    return jp.dot(self.vel, m.vel) + jp.dot(self.ang, m.ang)

  def matrix(self) -> jax.Array:
    return jp.concatenate([self.ang, self.vel], axis=-1)

  @classmethod
  def create(
      cls, ang: Optional[jax.Array] = None, vel: Optional[jax.Array] = None
  ) -> 'Motion':
    if ang is None and vel is None:
      raise ValueError('must specify either ang or vel')
    ang = jp.zeros_like(vel) if ang is None else ang
    vel = jp.zeros_like(ang) if vel is None else vel

    return Motion(ang=ang, vel=vel)

  @classmethod
  def zero(cls, shape=()) -> 'Motion':
    ang = jp.zeros(shape + (3,))
    vel = jp.zeros(shape + (3,))
    return Motion(ang, vel)


@struct.dataclass
class Force(Base):
  """Spatial force vector describing linear and angular (torque) force.

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jax.Array
  vel: jax.Array

  @classmethod
  def create(
      cls, ang: Optional[jax.Array] = None, vel: Optional[jax.Array] = None
  ) -> 'Force':
    if ang is None and vel is None:
      raise ValueError('must specify either ang or vel')
    ang = jp.zeros_like(vel) if ang is None else ang
    vel = jp.zeros_like(ang) if vel is None else vel

    return Force(ang=ang, vel=vel)


@struct.dataclass
class Inertia(Base):
  """Angular inertia, mass, and center of mass location.

  Attributes:
    transform: transform for the inertial frame relative to the link frame
      (i.e. center of mass position and orientation)
    i: (3, 3) inertia matrix about a point P
    mass: scalar mass
  """

  transform: Transform
  i: jax.Array
  mass: jax.Array

  def mul(self, m: Motion) -> 'Force':
    """Multiplies inertia with motion yielding a force."""
    ang = jp.dot(self.i, m.ang) + jp.cross(self.transform.pos, m.vel)
    vel = self.mass * m.vel - jp.cross(self.transform.pos, m.ang)

    return Force(ang=ang, vel=vel)


@struct.dataclass
class Link(Base):
  """A rigid segment of an articulated body.

  Links are connected to each other by joints.  By moving (rotating or
  translating) the joints, the entire system can be articulated.

  Attributes:
    transform: transform for the link frame relative to the parent frame
    joint: transform for the joint frame relative to the link frame
    inertia: mass, center of mass location, and inertia of this link
    invweight: mean inverse inertia at init_q
    constraint_stiffness: (num_link,) constraint spring for joint.
    constraint_vel_damping: (num_link,) linear damping for constraint spring.
    constraint_limit_stiffness: (num_link,) constraint for angle limits
    constraint_ang_damping: (num_link,) angular damping for constraint spring.
  """

  transform: Transform
  joint: Transform
  inertia: Inertia
  invweight: jax.Array
  # only used by `brax.physics.spring`:
  constraint_stiffness: jax.Array
  constraint_vel_damping: jax.Array
  constraint_limit_stiffness: jax.Array
  # only used by `brax.physics.spring` and `brax.physics.positional`:
  constraint_ang_damping: jax.Array


@struct.dataclass
class DoF(Base):
  """A degree of freedom in the system.

  Attributes:
    motion: spatial motion (linear or angular) of this DoF
    armature: models the inertia of a rotor (moving part of a motor)
    stiffness: restorative force back to zero position
    damping: restorative force back to zero velocity
    limit: tuple of min, max angle limits
    invweight: diagonal inverse inertia at init_qpos
    solver_params: (7,) limit constraint solver parameters
  """

  motion: Motion
  armature: jax.Array
  stiffness: jax.Array
  damping: jax.Array
  limit: Tuple[jax.Array, jax.Array]
  # only used by `brax.physics.generalized`:
  invweight: jax.Array
  solver_params: jax.Array


class Contact(mjx.Contact, Base):
  """Contact between two geometries.

  Attributes:
    link_idx: Tuple of link indices participating in contact.
    elasticity: bounce/restitution encountered when hitting another geometry
  """

  link_idx: jax.Array
  elasticity: jax.Array


@struct.dataclass
class Actuator(Base):
  """Actuator, transforms an input signal into a force (motor or thruster).

  Attributes:
    q_id: (num_actuators,) q index associated with an actuator
    qd_id: (num_actuators,) qd index associated with an actuator
    ctrl_range: (num_actuators, 2) actuator control range
    force_range: (num_actuators, 2) actuator force range
    gain: (num_actuators,) scaling factor for each actuator control input
    gear: (num_actuators,) scaling factor for each actuator force output
    bias_q: (num_actuators,) bias applied by q (e.g. position actuators)
    bias_qd: (num_actuators,) bias applied by qd (e.g. velocity actuators)
  """

  q_id: jax.Array
  qd_id: jax.Array
  ctrl_range: jax.Array
  force_range: jax.Array
  gain: jax.Array
  gear: jax.Array
  bias_q: jax.Array
  bias_qd: jax.Array


@struct.dataclass
class State:
  """Dynamic state that changes after every pipeline step.

  Attributes:
    q: (q_size,) joint position vector
    qd: (qd_size,) joint velocity vector
    x: (num_links,) link position in world frame
    xd: (num_links,) link velocity in world frame
    contact: calculated contacts
  """

  q: jax.Array
  qd: jax.Array
  x: Transform
  xd: Motion
  contact: Optional[Contact]


class System(mjx.Model):
  r"""Describes a physical environment: its links, joints and geometries.

  Attributes:
    gravity: (3,) linear universal force applied during forward dynamics
    viscosity: (1,) viscosity of the medium applied to all links
    density: (1,) density of the medium applied to all links
    link: (num_link,) the links in the system
    dof: (qd_size,) every degree of freedom for the system
    actuator: actuators that can be applied to links
    init_q: (q_size,) initial q position for the system
    elasticity: bounce/restitution encountered when hitting another geometry
    vel_damping: (1,) linear vel damping applied to each body.
    ang_damping: (1,) angular vel damping applied to each body.
    baumgarte_erp: how aggressively interpenetrating bodies should push away\
                from one another
    spring_mass_scale: a float that scales mass as `mass^(1 - x)`
    spring_inertia_scale: a float that scales inertia diag as `inertia^(1 - x)`
    joint_scale_ang: scale for position-based joint rotation update
    joint_scale_pos: scale for position-based joint position update
    collide_scale: fraction of position based collide update to apply
    enable_fluid: (1,) enables or disables fluid forces based on the
      default viscosity and density parameters provided in the XML
    link_names: (num_link,) link names
    link_types: (num_link,) string specifying the joint type of each link
                valid types are:
                * 'f': free, full 6 dof (position + rotation), no parent link
                * '1': revolute,  1 dof, like a hinge
                * '2': universal, 2 dof, like a drive shaft joint
                * '3': spherical, 3 dof, like a ball joint
    link_parents: (num_link,) int list specifying the index of each link's
                  parent link, or -1 if the link has no parent
    matrix_inv_iterations: maximum number of iterations of the matrix inverse
    solver_iterations: maximum number of iterations of the constraint solver
    solver_maxls: maximum number of line searches of the constraint solver
    mj_model: mujoco.MjModel that was used to build this brax System
  """

  gravity: jax.Array
  viscosity: Union[float, jax.Array]
  density: Union[float, jax.Array]
  link: Link
  dof: DoF
  actuator: Actuator
  init_q: jax.Array
  # only used in `brax.physics.spring` and `brax.physics.positional`:
  elasticity: jax.Array
  vel_damping: Union[float, jax.Array]
  ang_damping: Union[float, jax.Array]
  baumgarte_erp: Union[float, jax.Array]
  spring_mass_scale: Union[float, jax.Array]
  spring_inertia_scale: Union[float, jax.Array]
  # only used in `brax.physics.positional`:
  joint_scale_ang: Union[float, jax.Array]
  joint_scale_pos: Union[float, jax.Array]
  collide_scale: Union[float, jax.Array]
  # non-pytree nodes
  enable_fluid: bool = struct.field(pytree_node=False)
  link_names: List[str] = struct.field(pytree_node=False)
  link_types: str = struct.field(pytree_node=False)
  link_parents: Tuple[int, ...] = struct.field(pytree_node=False)
  # only used in `brax.physics.generalized`:
  matrix_inv_iterations: int = struct.field(pytree_node=False)
  solver_iterations: int = struct.field(pytree_node=False)
  solver_maxls: int = struct.field(pytree_node=False)
  mj_model: mujoco.MjModel = struct.field(pytree_node=False, default=None)

  def num_links(self) -> int:
    """Returns the number of links in the system."""
    return len(self.link_types)

  def dof_link(self, depth=False) -> jax.Array:
    """Returns the link index corresponding to each system dof."""
    link_idxs = []
    for i, link_type in enumerate(self.link_types):
      link_idxs.extend([i] * QD_WIDTHS[link_type])
    if depth:
      depth_fn = lambda i, p=self.link_parents: p[i] + 1 and 1 + depth_fn(p[i])
      depth_count = {}
      link_idx_depth = []
      for i in range(self.num_links()):
        depth = depth_fn(i)
        depth_idx = depth_count.get(depth, 0)
        depth_count[depth] = depth_idx + 1
        link_idx_depth.append(depth_idx)
      link_idxs = [link_idx_depth[i] for i in link_idxs]

    return jp.array(link_idxs)

  def dof_ranges(self) -> List[List[int]]:
    """Returns the dof ranges corresponding to each link."""
    beg, ranges = 0, []
    for t in self.link_types:
      ranges.append(list(range(beg, beg + QD_WIDTHS[t])))
      beg += QD_WIDTHS[t]
    return ranges

  def q_idx(self, link_type: str) -> jax.Array:
    """Returns the q indices corresponding to a link type."""
    idx, idxs = 0, []
    for typ in self.link_types:
      if typ in link_type:
        idxs.extend(range(idx, idx + Q_WIDTHS[typ]))
      idx += Q_WIDTHS[typ]
    return jp.array(idxs)

  def qd_idx(self, link_type: str) -> jax.Array:
    """Returns the qd indices corresponding to a link type."""
    idx, idxs = 0, []
    for typ in self.link_types:
      if typ in link_type:
        idxs.extend(range(idx, idx + QD_WIDTHS[typ]))
      idx += QD_WIDTHS[typ]
    return jp.array(idxs)

  def q_size(self) -> int:
    """Returns the size of the q vector (joint position) for this system."""
    return self.nq

  def qd_size(self) -> int:
    """Returns the size of the qd vector (joint velocity) for this system."""
    return self.nv

  def act_size(self) -> int:
    """Returns the act dimension for the system."""
    return self.nu


# below are some operation dispatch derivations


@functools.singledispatch
def _transform_do(other, self: Transform):
  del other, self
  return NotImplemented


@functools.singledispatch
def _transform_inv_do(other, self: Transform):
  del other, self
  return NotImplemented


@_transform_do.register(Transform)
def _(t: Transform, self: Transform) -> Transform:
  pos = self.pos + math.rotate(t.pos, self.rot)
  rot = math.quat_mul(self.rot, t.rot)
  return Transform(pos, rot)


@_transform_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
  rot_t = math.quat_inv(self.rot)
  ang = math.rotate(m.ang, rot_t)
  vel = math.rotate(m.vel - jp.cross(self.pos, m.ang), rot_t)
  return Motion(ang, vel)


@_transform_inv_do.register(Motion)
def _(m: Motion, self: Transform) -> Motion:
  rot_t = self.rot
  ang = math.rotate(m.ang, rot_t)
  vel = math.rotate(m.vel, rot_t) + jp.cross(self.pos, ang)
  return Motion(ang, vel)


@_transform_do.register(Force)
def _(f: Force, self: Transform) -> Force:
  vel = math.rotate(f.vel, self.rot)
  ang = math.rotate(f.ang, self.rot) + jp.cross(self.pos, vel)
  return Force(ang, vel)


@_transform_do.register(Inertia)
def _(it: Inertia, self: Transform) -> Inertia:
  h = jp.cross(self.pos, -jp.eye(3))
  rot = math.quat_to_3x3(self.rot)
  i = rot @ it.i @ rot.T + h @ h.T * it.mass
  transform = Transform(pos=self.pos * it.mass, rot=self.rot)
  return Inertia(transform=transform, i=i, mass=it.mass)


@functools.singledispatch
def _motion_cross(other, self: Motion):
  del other, self
  return NotImplemented


@_motion_cross.register(Motion)
def _(m: Motion, self: Motion) -> Motion:
  vel = jp.cross(self.ang, m.vel) + jp.cross(self.vel, m.ang)
  ang = jp.cross(self.ang, m.ang)
  return Motion(ang, vel)


@_motion_cross.register(Force)
def _(f: Force, self: Motion) -> Force:
  vel = jp.cross(self.ang, f.vel)
  ang = jp.cross(self.ang, f.ang) + jp.cross(self.vel, f.vel)
  return Force(ang, vel)
