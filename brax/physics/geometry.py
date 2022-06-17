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
"""Geometry functions and classes to be used for collision detection."""

import itertools
from typing import List, Mapping, Tuple

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics.base import QP, vec_to_arr

# Coordinates of the 8 corners of a box.
_BOX_CORNERS = jp.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))))

# pyformat: disable
# The faces of a triangulated box, i.e. the indices in _BOX_CORNERS of the
# vertices of the 12 triangles (two triangles for each side of the box).
_BOX_FACES = [
    0, 4, 1, 4, 5, 1,  # front
    0, 2, 4, 2, 6, 4,  # bottom
    6, 5, 4, 6, 7, 5,  # right
    2, 3, 6, 3, 7, 6,  # back
    1, 5, 3, 5, 7, 3,  # top
    0, 1, 2, 1, 3, 2,  # left
]

# Normals of the triangulated box faces above.
_BOX_FACE_NORMALS = jp.array([
    [0, -1., 0],  # front
    [0, -1., 0],
    [0, 0, -1.],  # bottom
    [0, 0, -1.],
    [+1., 0, 0],  # right
    [+1., 0, 0],
    [0, +1., 0],  # back
    [0, +1., 0],
    [0, 0, +1.],  # top
    [0, 0, +1.],
    [-1., 0, 0],  # left
    [-1., 0, 0],
])
# pyformat: enable


@pytree.register
class Collidable:
  """Part of a body (with geometry and mass/inertia) that can collide.

  Collidables can repeat for a geometry. e.g. a body with a box collider has 8
  corner collidables.
  """

  def __init__(self, collidables: List[config_pb2.Body], body: bodies.Body):
    self.body = jp.take(body, [body.index[c.name] for c in collidables])
    self.pos = jp.array(
        [vec_to_arr(c.colliders[0].position) for c in collidables])
    self.friction = jp.array(
        [c.colliders[0].material.friction for c in collidables])
    self.elasticity = jp.array(
        [c.colliders[0].material.elasticity for c in collidables])

  def position(self, qp: QP) -> jp.ndarray:
    """Returns the collidable's position in world space."""
    pos = jp.take(qp.pos, self.body.idx)
    rot = jp.take(qp.rot, self.body.idx)
    return pos + jp.vmap(math.rotate)(self.pos, rot)


@pytree.register
class Contact:
  """Stores information about contacts between two collidables."""

  def __init__(self, pos: jp.ndarray, vel: jp.ndarray, normal: jp.ndarray,
               penetration: jp.ndarray):
    """Creates a Contact.

    Args:
      pos: contact position in world space
      vel: contact velocity in world space
      normal: normal vector of surface providing contact
      penetration: distance the two collidables are penetrating one-another
    """
    self.pos = pos
    self.vel = vel
    self.normal = normal
    self.penetration = penetration


@pytree.register
class BoxCorner(Collidable):
  """A box corner."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    super().__init__([boxes[i // 8] for i in range(len(boxes) * 8)], body)
    corners = []
    for b in boxes:
      col = b.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      box = _BOX_CORNERS * vec_to_arr(col.box.halfsize)
      box = jp.vmap(math.rotate, include=(True, False))(box, rot)
      box = box + vec_to_arr(col.position)
      corners.extend(box)
    self.corner = jp.array(corners)


@pytree.register
class BaseMesh(Collidable):
  """Base class for mesh colliders."""

  def __init__(self, collidables: List[config_pb2.Body], body: bodies.Body,
               vertices: jp.ndarray, faces: jp.ndarray,
               face_normals: jp.ndarray):
    super().__init__(collidables, body)
    self.vertices = vertices
    self.faces = faces
    self.face_normals = face_normals


@pytree.register
class TriangulatedBox(BaseMesh):
  """A box converted into a triangular mesh."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    vertices = []
    faces = []
    face_normals = []
    for b in boxes:
      col = b.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      vertex = _BOX_CORNERS * vec_to_arr(col.box.halfsize)
      vertex = jp.vmap(math.rotate, include=(True, False))(vertex, rot)
      vertex = vertex + vec_to_arr(col.position)
      vertices.extend(vertex)

      # Each face consists of two triangles.
      face = jp.reshape(jp.take(vertex, _BOX_FACES), (-1, 3, 3))
      faces.extend(face)

      # Apply rotation to face normals.
      face_normal = jp.vmap(
          math.rotate, include=(True, False))(_BOX_FACE_NORMALS, rot)
      face_normals.extend(face_normal)

    # Each triangle is a collidable.
    super().__init__([boxes[i // 12] for i in range(len(boxes) * 12)], body,
                     jp.array(vertices), jp.array(faces),
                     jp.array(face_normals))


@pytree.register
class Plane(Collidable):
  """An infinite plane with normal pointing in the +z direction."""


@pytree.register
class Capsule(Collidable):
  """A capsule with an ends pointing in the +z, -z directions."""

  def __init__(self, capsules: List[config_pb2.Body], body: bodies.Body):
    super().__init__(capsules, body)
    ends = []
    radii = []
    for c in capsules:
      col = c.colliders[0]
      axis = math.rotate(
          jp.array([0., 0., 1.]), math.euler_to_quat(vec_to_arr(col.rotation)))
      segment_length = col.capsule.length / 2. - col.capsule.radius
      ends.append(axis * segment_length)
      radii.append(col.capsule.radius)
    self.end = jp.array(ends)
    self.radius = jp.array(radii)


@pytree.register
class CapsuleEnd(Collidable):
  """A capsule with variable ends either in the +z or -z directions."""

  def __init__(self, capsules: List[config_pb2.Body], body: bodies.Body):
    var_caps = [[c] if c.colliders[0].capsule.end else [c, c] for c in capsules]
    super().__init__(sum(var_caps, []), body)
    ends = []
    radii = []
    for c in capsules:
      col = c.colliders[0]
      axis = math.rotate(
          jp.array([0., 0., 1.]), math.euler_to_quat(vec_to_arr(col.rotation)))
      segment_length = col.capsule.length / 2. - col.capsule.radius
      for end in [col.capsule.end] if col.capsule.end else [-1, 1]:
        ends.append(vec_to_arr(col.position) + end * axis * segment_length)
        radii.append(col.capsule.radius)
    self.end = jp.array(ends)
    self.radius = jp.array(radii)


@pytree.register
class HeightMap(Collidable):
  """A height map with heights in a grid layout."""

  def __init__(self, heightmaps: List[config_pb2.Body], body: bodies.Body):
    super().__init__(heightmaps, body)
    heights = []
    cell_sizes = []
    for h in heightmaps:
      col = h.colliders[0]
      mesh_size = int(jp.sqrt(len(col.heightMap.data)))
      if len(col.heightMap.data) != mesh_size**2:
        raise ValueError('height map data length should be a perfect square.')
      height = jp.array(col.heightMap.data).reshape((mesh_size, mesh_size))
      heights.append(height)
      cell_sizes.append(col.heightMap.size / (mesh_size - 1))
    self.height = jp.array(heights)
    self.cell_size = jp.array(cell_sizes)


@pytree.register
class Mesh(BaseMesh):
  """A triangular mesh with vertex or face collidables."""

  def __init__(self,
               meshes: List[config_pb2.Body],
               body: bodies.Body,
               mesh_geoms: Mapping[str, config_pb2.MeshGeometry],
               use_points: bool = False):
    """Initializes a triangular mesh collider.

    Args:
      meshes: Mesh colliders of the body in the config.
      body: The body that the mesh colliders belong to.
      mesh_geoms: The dictionary of the mesh geometries keyed by their names.
      use_points: Whether to use the points or the faces of the mesh as the
        collidables.
    """
    geoms = [mesh_geoms[m.colliders[0].mesh.name] for m in meshes]

    vertices = []
    faces = []
    face_normals = []
    for m, g in zip(meshes, geoms):
      col = m.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      scale = col.mesh.scale if col.mesh.scale else 1

      # Apply scaling and body transformations to the vertices.
      vertex = jp.array(
          [[v.x * scale, v.y * scale, v.z * scale] for v in g.vertices])
      vertex = jp.vmap(math.rotate, include=(True, False))(vertex, rot)
      vertex = vertex + vec_to_arr(col.position)
      vertices.extend(vertex)

      # Each face is a triangle.
      face = jp.reshape(jp.take(vertex, g.faces), (-1, 3, 3))
      faces.extend(face)

      # Apply rotation to face normals.
      face_normal = jp.array([vec_to_arr(n) for n in g.face_normals])
      face_normal = jp.vmap(
          math.rotate, include=(True, False))(face_normal, rot)
      face_normals.extend(face_normal)

    collidables = [[m] * len(g.vertices if use_points else g.faces)
                   for m, g in zip(meshes, geoms)]
    super().__init__(
        sum(collidables, []), body, jp.array(vertices), jp.array(faces),
        jp.array(face_normals))


@pytree.register
class PointMesh(Mesh):
  """A triangular mesh with vertex collidables."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs, use_points=True)


def closest_segment_point(a: math.Vector3, b: math.Vector3,
                          pt: math.Vector3) -> math.Vector3:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0., 1.) * ab


def closest_segment_point_and_dist(
    a: math.Vector3, b: math.Vector3,
    pt: math.Vector3) -> Tuple[math.Vector3, jp.ndarray]:
  """Returns the closest point and distance^2 on the a-b line segment to pt."""
  p = closest_segment_point(a, b, pt)
  v = pt - p
  return p, jp.dot(v, v)


def closest_segment_point_plane(a: math.Vector3, b: math.Vector3,
                                p0: math.Vector3,
                                plane_normal: math.Vector3) -> math.Vector3:
  """Gets the closest point between a line segment and a plane."""
  # If a line segment is parametrized as S(t) = a + t * (b - a), we can
  # plug it into the plane equation dot(n, S(t)) - d = 0, then solve for t to
  # get the line-plane intersection. We then clip t to be in [0, 1] to be on
  # the line segment.
  n = plane_normal
  d = jp.dot(p0, n)  # shortest distance from origin to plane
  denom = jp.dot(n, b - a)
  t = (d - jp.dot(n, a)) / (denom + 1e-6)
  t = jp.clip(t, 0, 1)
  segment_point = a + t * (b - a)
  return segment_point


def closest_segment_to_segment_points(
    a0: math.Vector3, a1: math.Vector3, b0: math.Vector3,
    b1: math.Vector3) -> Tuple[math.Vector3, math.Vector3]:
  """Returns closest points on two line segments."""
  # Gets the closest segment points by first finding the closest points
  # between two lines. Points are then clipped to be on the line segments
  # and edge cases with clipping are handled.
  dir_a = a1 - a0
  len_a = jp.safe_norm(dir_a)
  dir_a = dir_a / (len_a + 1e-6)
  half_len_a = len_a / 2

  dir_b = b1 - b0
  len_b = jp.safe_norm(dir_b)
  dir_b = dir_b / (len_b + 1e-6)
  half_len_b = len_b / 2

  # Segment mid-points.
  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Translation between two segment mid-points.
  trans = a_mid - b_mid

  # Parametrize points on each line as follows:
  #  point_on_a = a_mid + t_a * dir_a
  #  point_on_b = b_mid + t_b * dir_b
  # and analytically minimize the distance between the two points.
  dira_dot_dirb = dir_a.dot(dir_b)
  dira_dot_trans = dir_a.dot(trans)
  dirb_dot_trans = dir_b.dot(trans)
  denom = 1 - dira_dot_dirb * dira_dot_dirb

  t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
  t_b = dirb_dot_trans + t_a * dira_dot_dirb
  t_a = jp.clip(t_a, -half_len_a, half_len_a)
  t_b = jp.clip(t_b, -half_len_b, half_len_b)

  best_a = a_mid + dir_a * t_a
  best_b = b_mid + dir_b * t_b

  # Resolve edge cases where both closest points are clipped to the segment
  # endpoints by recalculating the closest segment points for the current
  # clipped points, and then picking the pair of points with smallest
  # distance. An example of this edge case is when lines intersect but line
  # segments don't.
  new_a, d1 = closest_segment_point_and_dist(a0, a1, best_b)
  new_b, d2 = closest_segment_point_and_dist(b0, b1, best_a)
  best_a = jp.where(d1 < d2, new_a, best_a)
  best_b = jp.where(d1 < d2, best_b, new_b)

  return best_a, best_b


def closest_triangle_point(p0: math.Vector3, p1: math.Vector3, p2: math.Vector3,
                           pt: math.Vector3) -> math.Vector3:
  """Gets the closest point on a triangle to another point in space."""
  # Parametrize the triangle s.t. a point inside the triangle is
  # Q = p0 + u * e0 + v * e1, when 0 <= u <= 1, 0 <= v <= 1, and
  # 0 <= u + v <= 1. Let e0 = (p1 - p0) and e1 = (p2 - p0).
  # We analytically minimize the distance between the point pt and Q.
  e0 = p1 - p0
  e1 = p2 - p0
  a = e0.dot(e0)
  b = e0.dot(e1)
  c = e1.dot(e1)
  d = pt - p0
  # The determinant is 0 only if the angle between e1 and e0 is 0
  # (i.e. the triangle has overlapping lines).
  det = (a * c - b * b)
  u = (c * e0.dot(d) - b * e1.dot(d)) / det
  v = (-b * e0.dot(d) + a * e1.dot(d)) / det
  inside = (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1) & (u + v <= 1)
  closest_p = p0 + u * e0 + v * e1
  d0 = (closest_p - pt).dot(closest_p - pt)

  # If the closest point is outside the triangle, it must be on an edge, so we
  # check each triangle edge for a closest point to the point pt.
  closest_p1, d1 = closest_segment_point_and_dist(p0, p1, pt)
  closest_p = jp.where((d0 < d1) & inside, closest_p, closest_p1)
  min_d = jp.where((d0 < d1) & inside, d0, d1)

  closest_p2, d2 = closest_segment_point_and_dist(p1, p2, pt)
  closest_p = jp.where(d2 < min_d, closest_p2, closest_p)
  min_d = jp.minimum(min_d, d2)

  closest_p3, d3 = closest_segment_point_and_dist(p2, p0, pt)
  closest_p = jp.where(d3 < min_d, closest_p3, closest_p)
  min_d = jp.minimum(min_d, d3)

  return closest_p


def closest_segment_triangle_points(
    a: math.Vector3, b: math.Vector3, p0: math.Vector3, p1: math.Vector3,
    p2: math.Vector3,
    triangle_normal: math.Vector3) -> Tuple[math.Vector3, math.Vector3]:
  """Gets the closest points between a line segment and triangle."""
  # The closest triangle point is either on the edge or within the triangle.
  # First check triangle edges for the closest point.
  seg_pt1, tri_pt1 = closest_segment_to_segment_points(a, b, p0, p1)
  d1 = (seg_pt1 - tri_pt1).dot(seg_pt1 - tri_pt1)
  seg_pt2, tri_pt2 = closest_segment_to_segment_points(a, b, p1, p2)
  d2 = (seg_pt2 - tri_pt2).dot(seg_pt2 - tri_pt2)
  seg_pt3, tri_pt3 = closest_segment_to_segment_points(a, b, p0, p2)
  d3 = (seg_pt3 - tri_pt3).dot(seg_pt3 - tri_pt3)

  # Next, handle the case where the closest triangle point is inside the
  # triangle. Either the line segment intersects the triangle or a segment
  # endpoint is closest to a point inside the triangle.
  # If the line overlaps the triangle and is parallel to the triangle plane,
  # the chosen triangle point is arbitrary.
  seg_pt4 = closest_segment_point_plane(a, b, p0, triangle_normal)
  tri_pt4 = closest_triangle_point(p0, p1, p2, seg_pt4)
  d4 = (seg_pt4 - tri_pt4).dot(seg_pt4 - tri_pt4)

  # Get the point with minimum distance from the line segment point to the
  # triangle point.
  distance = jp.array([[d1, d2, d3, d4]])
  min_dist = jp.amin(distance)
  mask = (distance == min_dist).T
  seg_pt = jp.array([seg_pt1, seg_pt2, seg_pt3, seg_pt4]) * mask
  tri_pt = jp.array([tri_pt1, tri_pt2, tri_pt3, tri_pt4]) * mask
  seg_pt = jp.sum(seg_pt, axis=0) / jp.sum(mask)
  tri_pt = jp.sum(tri_pt, axis=0) / jp.sum(mask)
  return seg_pt, tri_pt
