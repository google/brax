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

"""Base brax primitives and simple manipulations of them."""
import functools
from typing import Any, List, Optional, Sequence, Tuple, Union

from brax.v2 import math
from flax import struct
from jax import numpy as jp
from jax import vmap
from jax.tree_util import tree_map

# f: free, 1: 1-dof, 2: 2-dof, 3: 3-dof
Q_WIDTHS = {'f': 7, '1': 1, '2': 2, '3': 3}
QD_WIDTHS = {'f': 6, '1': 1, '2': 2, '3': 3}


class _Base:
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

  def select(self, o: Any, cond: jp.ndarray) -> Any:
    return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

  def slice(self, beg: int, end: int) -> Any:
    return tree_map(lambda x: x[beg:end], self)

  def take(self, i, axis=0) -> Any:
    return tree_map(lambda x: jp.take(x, i, axis=axis, mode='wrap'), self)

  def concatenate(self, *others: Any, axis: int = 0) -> Any:
    return tree_map(lambda *x: jp.concatenate(x, axis=axis), self, *others)

  def index_set(
      self, idx: Union[jp.ndarray, Sequence[jp.ndarray]], o: Any
  ) -> Any:
    return tree_map(lambda x, y: x.at[idx].set(y), self, o)

  def index_sum(
      self, idx: Union[jp.ndarray, Sequence[jp.ndarray]], o: Any
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

  @property
  def T(self):  # pylint:disable=invalid-name
    return tree_map(lambda x: x.T, self)


@struct.dataclass
class Transform(_Base):
  """Transforms the position and rotation of a coordinate frame.

  Attributes:
    pos: (3,) position transform of the coordinate frame
    rot: (4,) quaternion rotation the coordinate frame
  """

  pos: jp.ndarray
  rot: jp.ndarray

  def do(self, o):
    """Apply the transform."""
    return _transform_do(o, self)

  def to_local(self, t: 'Transform') -> 'Transform':
    """Move transform into basis of t."""
    pos = math.rotate(self.pos - t.pos, math.quat_inv(t.rot))
    rot = math.quat_mul(math.quat_inv(t.rot), self.rot)
    return Transform(pos=pos, rot=rot)

  def inv(self):
    """Invert the transform."""
    return Transform(pos=-1.0 * self.pos, rot=math.quat_inv(self.rot))

  @classmethod
  def create(
      cls, pos: Optional[jp.ndarray] = None, rot: Optional[jp.ndarray] = None
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
class Motion(_Base):
  """Spatial motion vector describing linear and angular velocity.

  More on spatial vectors: http://royfeatherstone.org/spatial/v2/index.html

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jp.ndarray
  vel: jp.ndarray

  def cross(self, other):
    return _motion_cross(other, self)

  def dot(self, m: Union['Motion', 'Force']) -> jp.ndarray:
    return jp.dot(self.vel, m.vel) + jp.dot(self.ang, m.ang)

  def matrix(self) -> jp.ndarray:
    return jp.concatenate([self.ang, self.vel], axis=-1)

  @classmethod
  def create(
      cls, ang: Optional[jp.ndarray] = None, vel: Optional[jp.ndarray] = None
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
class Force(_Base):
  """Spatial force vector describing linear and angular (torque) force.

  Attributes:
    ang: (3,) angular velocity about a normal
    vel: (3,) linear velocity in the direction of the normal
  """

  ang: jp.ndarray
  vel: jp.ndarray


@struct.dataclass
class Inertia(_Base):
  """Angular inertia, mass, and center of mass location.

  Attributes:
    transform: transform from inertia frame to link frame, corresponding to
      center of mass position and orientation
    i: (3, 3) inertia matrix about a point P
    mass: scalar mass
  """

  transform: Transform
  i: jp.ndarray
  mass: jp.ndarray

  def mul(self, m: Motion) -> 'Force':
    """Multiplies inertia with motion yielding a force."""
    ang = jp.dot(self.i, m.ang) + jp.cross(self.transform.pos, m.vel)
    vel = self.mass * m.vel - jp.cross(self.transform.pos, m.ang)

    return Force(ang=ang, vel=vel)


@struct.dataclass
class Link(_Base):
  """A rigid segment of an articulated body.

  Links are connected to eachother by joints.  By moving (rotating or
  translating) the joints, the entire system can be articulated.

  Attributes:
    transform: transform from link frame to its parent
    joint: location of joint in link frame
    inertia: mass, center of mass location, and inertia of this link
    invweight: mean inverse inertia at init_q
    constraint_stiffness: (num_link,) constraint spring for joint.
    constraint_damping: (num_link,) damping for constraint spring.
    constraint_limit_stiffness: (num_link,) constraint for angle limits
    constraint_ang_damping: (num_link,) angular damping for constraint spring.
  """

  transform: Transform
  joint: Transform
  inertia: Inertia
  invweight: jp.ndarray
  # only used by `brax.physics.spring`:
  constraint_stiffness: jp.ndarray
  constraint_damping: jp.ndarray
  constraint_limit_stiffness: jp.ndarray
  # only used by `brax.physics.spring` and `brax.physics.pbd`:
  constraint_ang_damping: jp.ndarray


@struct.dataclass
class DoF(_Base):
  """A degree of freedom in the system.

  Attributes:
    motion: spatial motion (linear or angular) of this DoF
    armature: models the inertia of a rotor (moving part of a motor)
    stiffness: restorative force back to zero position
    damping: restorative force back to zero velocity
    limit: tuple of min, max angle limits
    invweight: diagonal inverse inertia at init_qpos
  """

  motion: Motion
  armature: jp.ndarray
  stiffness: jp.ndarray
  damping: jp.ndarray
  limit: Tuple[jp.ndarray, jp.ndarray]
  # only used by `brax.physics.generalized`:
  invweight: jp.ndarray


@struct.dataclass
class Geometry(_Base):
  """A surface or spatial volume with a shape and material properties.

  Attributes:
    link_idx: Link index to which this Geometry is attached
    transform: transform from this geometry's coordinate space to its parent
      link, or to world space in the case of unparented geometry
    friction: resistance encountered when sliding against another geometry
    elasticity: bounce/restitution encountered when hitting another geometry
  """

  link_idx: Optional[jp.ndarray]
  transform: Transform
  friction: jp.ndarray
  elasticity: jp.ndarray


@struct.dataclass
class Sphere(Geometry):
  """A sphere.

  Attributes:
    radius: radius of the sphere
  """

  radius: jp.ndarray


@struct.dataclass
class Capsule(Geometry):
  """A capsule.

  Attributes:
    radius: radius of the capsule end
    length: distance between the two capsule end centroids
  """

  radius: jp.ndarray
  length: jp.ndarray


@struct.dataclass
class Box(Geometry):
  """A box.

  Attributes:
    halfsize: (3,) half sizes for each box side
  """

  halfsize: jp.ndarray


@struct.dataclass
class Plane(Geometry):
  """An infinite plane whose normal points at +z in its coordinate space."""


@struct.dataclass
class Mesh(Geometry):
  """A mesh loaded from an OBJ or STL file.

  The mesh is expected to be in the counter-clockwise winding order.

  Attributes:
    vert: (num_verts, 3) spatial coordinates associated with each vertex
    face: (num_faces, num_face_vertices) vertices associated with each face
  """

  vert: jp.ndarray
  face: jp.ndarray


@struct.dataclass
class Convex(Mesh):
  """A convex mesh geometry.

  Attributes:
    unique_edge: (num_unique, 2) vert index associated with each unique edge
  """

  unique_edge: jp.ndarray


@struct.dataclass
class Contact(_Base):
  """Contact between two geometries.

  Attributes:
    pos: contact position, or average of the two closest points, in world frame
    normal: contact normal on the surface of geometry b
    penetration: penetration distance between two geometries. positive means the
      two geometries are interpenetrating, negative means they are not
    friction: resistance encountered when sliding against another geometry
    elasticity: bounce/restitution encountered when hitting another geometry
    link_idx: Tuple of link indices participating in contact.  The second part
      of the tuple can be None if the second geometry is static.
  """

  pos: jp.ndarray
  normal: jp.ndarray
  penetration: jp.ndarray
  friction: jp.ndarray
  # only used by `brax.physics.spring` and `brax.physics.pbd`:
  elasticity: jp.ndarray

  link_idx: Tuple[jp.ndarray, Optional[jp.ndarray]]


@struct.dataclass
class Actuator(_Base):
  """Actuator, transforms an input signal into a force (motor or thruster).

  Attributes:
    ctrl_range: (num_actuators,) control range for each actuator
    gear: (num_actuators,) a list of floats used as a scaling factor for each
      actuator torque output
  """

  ctrl_range: jp.ndarray
  gear: jp.ndarray


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

  q: jp.ndarray
  qd: jp.ndarray
  x: jp.ndarray
  xd: jp.ndarray
  contact: Optional[Contact]


@struct.dataclass
class System:
  r"""Describes a physical environment: its links, joints and geometries.

  Attributes:
    dt: timestep used for the simulation
    gravity: (3,) linear universal force applied during forward dynamics
    link: (num_link,) the links in the system
    dof: (qd_size,) every degree of freedom for the system
    geoms: (num_geoms,) every geom in the system
    contacts: a list of all geometry pairs to test for this system
    actuator: actuators that can be applied to links
    init_q: (q_size,) initial q position for the system
    vel_damping: (1,) linear vel damping applied to each body.
    ang_damping: (1,) angular vel damping applied to each body.
    baumgarte_erp: how aggressively interpenetrating bodies should push away\
                from one another
    link_names: (num_link,) link names
    link_types: (num_link,) string specifying the joint type of each link
                valid types are:
                * 'f': free, full 6 dof (position + rotation), no parent link
                * '1': revolute,  1 dof, like a hinge
                * '2': universal, 2 dof, like a drive shaft joint
                * '3': spherical, 3 dof, like a ball joint
    link_parents: (num_link,) int list specifying the index of each link's
                  parent link, or -1 if the link has no parent
    actuator_types: (num_actuators,) string specifying the actuator types:
                * 't': torque
                * 'p': position
    actuator_link_id: (num_actuators,) the link id associated with each actuator
    actuator_qid: (num_actuators,) the q index associated with each actuator
    actuator_qdid: (num_actuators,) the qd index associated with each actuator
    solver_iterations: maximum number of iterations of the constraint solver
  """

  dt: jp.ndarray
  gravity: jp.ndarray
  link: Link
  dof: DoF
  geoms: List[Geometry]
  contacts: List[Tuple[Geometry, Geometry]]
  actuator: Actuator
  init_q: jp.ndarray
  # only used in `brax.physics.spring` and `brax.physics.pbd`:
  vel_damping: jp.float32
  ang_damping: jp.float32
  baumgarte_erp: jp.float32

  link_names: List[str] = struct.field(pytree_node=False)
  link_types: str = struct.field(pytree_node=False)
  link_parents: Tuple[int, ...] = struct.field(pytree_node=False)
  actuator_types: str = struct.field(pytree_node=False)
  actuator_link_id: List[int] = struct.field(pytree_node=False)
  actuator_qid: List[int] = struct.field(pytree_node=False)
  actuator_qdid: List[int] = struct.field(pytree_node=False)
  # only used in `brax.physics.generalized`:
  solver_iterations: int = struct.field(pytree_node=False)

  def num_links(self) -> int:
    """Returns the number of links in the system."""
    return len(self.link_types)

  def dof_link(self, depth=False) -> jp.ndarray:
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

  def q_idx(self, link_type: str) -> jp.ndarray:
    """Returns the q indices corresponding to a link type."""
    idx, idxs = 0, []
    for typ in self.link_types:
      if typ in link_type:
        idxs.extend(range(idx, idx + Q_WIDTHS[typ]))
      idx += Q_WIDTHS[typ]
    return jp.array(idxs)

  def qd_idx(self, link_type: str) -> jp.ndarray:
    """Returns the qd indices corresponding to a link type."""
    idx, idxs = 0, []
    for typ in self.link_types:
      if typ in link_type:
        idxs.extend(range(idx, idx + QD_WIDTHS[typ]))
      idx += QD_WIDTHS[typ]
    return jp.array(idxs)

  def q_size(self) -> int:
    """Returns the size of the q vector (joint position) for this sytem."""
    return sum([Q_WIDTHS[t] for t in self.link_types])

  def qd_size(self) -> int:
    """Returns the size of the qd vector (joint velocity) for this sytem."""
    return sum([QD_WIDTHS[t] for t in self.link_types])

  def act_size(self) -> int:
    """Returns the act dimension for the system."""
    return sum([QD_WIDTHS[self.link_types[i]] for i in self.actuator_link_id])


# below are some operation dispatch derivations


@functools.singledispatch
def _transform_do(other, self: Transform):
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


@_transform_do.register(Force)
def _(f: Force, self: Transform) -> Force:
  vel = math.rotate(f.vel, self.rot)
  ang = math.rotate(f.ang, self.rot) + jp.cross(self.pos, f.vel)

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
