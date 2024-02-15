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

# pylint:disable=g-multiple-import
"""Geometry functions and classes to be used for collision detection."""

import itertools
from typing import List, Mapping, Tuple, Union

from brax.v1 import jumpy as jp
from brax.v1 import math
from brax.v1 import pytree
from brax.v1.physics import bodies
from brax.v1.physics import config_pb2
from brax.v1.physics.base import QP, vec_to_arr

# Coordinates of the 8 corners of a box.
_BOX_CORNERS = jp.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))))

# pyformat: disable
# The faces of a triangulated box, i.e. the indices in _BOX_CORNERS of the
# vertices of the 12 triangles (two triangles for each side of the box).
_TRIANGULATED_BOX_FACES = [
    0, 4, 1, 4, 1, 5,  # front
    0, 4, 2, 2, 4, 6,  # bottom
    6, 4, 5, 6, 5, 7,  # right
    2, 6, 3, 3, 6, 7,  # back
    1, 3, 5, 5, 3, 7,  # top
    0, 2, 1, 1, 2, 3,  # left
]
# Normals of the triangulated box faces above.
_TRIANGULATED_BOX_FACE_NORMALS = [
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
]
# Polygon box faces using a clockwise winding order convention.
_BOX_FACES = [
    0, 1, 5, 4,  # front
    0, 4, 6, 2,  # bottom
    6, 4, 5, 7,  # right
    2, 6, 7, 3,  # back
    1, 3, 7, 5,  # top
    0, 2, 3, 1,  # left
]
# Normals of the polygon box faces above.
_BOX_FACE_NORMALS = [
    [0, -1., 0],  # front
    [0, 0, -1.],  # bottom
    [+1., 0, 0],  # right
    [0, +1., 0],  # back
    [0, 0, +1.],  # top
    [-1., 0, 0],  # left
]
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

  def position(self, qp: QP) -> math.Vector3:
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
class Box(Collidable):
  """A box with corner collidables."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    super().__init__(boxes, body)
    corners = []
    for b in boxes:
      col = b.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      box = _BOX_CORNERS * vec_to_arr(col.box.halfsize)
      box = jp.vmap(math.rotate, include=(True, False))(box, rot)
      box = box + vec_to_arr(col.position)
      corners.append(box)
    self.corner = jp.array(corners)


@pytree.register
class BaseMesh(Collidable):
  """Base class for mesh colliders."""

  def __init__(self, collidables: List[config_pb2.Body], body: bodies.Body,
               vertices: jp.ndarray, faces: jp.ndarray,
               face_normals: jp.ndarray):
    super().__init__(collidables, body)
    new_faces = []
    for face_normal, face in zip(face_normals, faces):
      # Fix the winding order of the faces if necessary by checking the first
      # two edges of each face for a clockwise winding order.
      winding_order = jp.vmap(jp.dot)(jp.cross(
          face[:, 0] - face[:, -1], face[:, 0] - face[:, 1]), face_normal) >= 0
      face = jp.vmap(lambda x, y: jp.where(x, y, y[::-1]))(winding_order, face)
      new_faces.append(face)
    self.faces = jp.array(new_faces)
    self.face_normals = face_normals
    self.vertices = vertices


@pytree.register
class BoxMesh(BaseMesh):
  """A box converted into a mesh."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body,
               box_faces: List[int], box_face_normals: List[List[float]]):
    vertices = []
    faces = []
    face_normals = []
    num_face_points = len(box_faces) // len(box_face_normals)
    box_face_normals = jp.array(box_face_normals)
    for b in boxes:
      col = b.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      vertex = _BOX_CORNERS * vec_to_arr(col.box.halfsize)
      vertex = jp.vmap(math.rotate, include=(True, False))(vertex, rot)
      vertex = vertex + vec_to_arr(col.position)
      vertices.append(vertex)

      # Apply rotation to face normals.
      face_normal = jp.vmap(
          math.rotate, include=(True, False))(box_face_normals, rot)
      face_normals.append(face_normal)

      # Each face consists of a `num_face_points` dimensional polygon.
      face = jp.reshape(jp.take(vertex, box_faces), (-1, num_face_points, 3))
      faces.append(face)

    super().__init__(boxes, body, jp.array(vertices), jp.array(faces),
                     jp.array(face_normals))


@pytree.register
class TriangulatedBox(BoxMesh):
  """A box converted into a triangular mesh."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    super().__init__(boxes, body, _TRIANGULATED_BOX_FACES,
                     _TRIANGULATED_BOX_FACE_NORMALS)


@pytree.register
class HullBox(BoxMesh):
  """A box converted into a polygon mesh."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    super().__init__(boxes, body, _BOX_FACES, _BOX_FACE_NORMALS)


@pytree.register
class Plane(Collidable):
  """An infinite plane with normal pointing in the +z direction."""


@pytree.register
class ClippedPlane(Collidable):
  """A clipped plane with the normal pointing in a user specified direction."""

  def  __init__(self, planes: List[config_pb2.Body], body: bodies.Body):
    super().__init__(planes, body)
    normals, xdir, ydir = [], [], []
    halfsize_x, halfsize_y, pos = [], [], []
    for p in planes:
      col = p.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      normal = math.rotate(jp.array([0., 0., 1.]), rot)
      x = math.rotate(jp.array([1., 0., 0.]), rot)
      y = math.rotate(jp.array([0., 1., 0.]), rot)
      normals.append(normal)
      xdir.append(x)
      ydir.append(y)
      halfsize_x.append(col.clipped_plane.halfsize_x)
      halfsize_y.append(col.clipped_plane.halfsize_y)
      pos.append(vec_to_arr(col.position))

    self.normal = jp.array(normals)
    self.x, self.y = jp.array(xdir), jp.array(ydir)
    self.halfsize_x = jp.array(halfsize_x)
    self.halfsize_y = jp.array(halfsize_y)
    self.pos = jp.array(pos)


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
      segment_length = col.capsule.length * 0.5 - col.capsule.radius
      ends.append(axis * segment_length)
      radii.append(col.capsule.radius)
    self.end = jp.array(ends)
    self.radius = jp.array(radii)


@pytree.register
class CapsuleEnd(Collidable):
  """A capsule with variable ends either in the +z or -z directions."""

  def __init__(self, capsules: List[config_pb2.Body], body: bodies.Body):
    super().__init__(capsules, body)
    ends = []
    radii = []
    for c in capsules:
      col = c.colliders[0]
      axis = math.rotate(
          jp.array([0., 0., 1.]), math.euler_to_quat(vec_to_arr(col.rotation)))
      segment_length = col.capsule.length * 0.5 - col.capsule.radius
      caps = []
      for end in [col.capsule.end] if col.capsule.end else [-1, 1]:
        caps.append(vec_to_arr(col.position) + end * axis * segment_length)
      ends.append(caps)
      radii.append(col.capsule.radius)
    # if there's a mix of 1 and 2 end capsules, pad the 1-end capsules
    # with a dummy cap that has 0 radius with the same end-location.
    # this facilitates vectorizing over the cap dimension
    if len(set([len(e) for e in ends])) != 1:
      for e in ends:
        if len(e) == 1:
          e.append(e[0])

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
      mesh_size = int(jp.sqrt(len(col.heightMap.data)))  # pytype: disable=wrong-arg-types  # jax-ndarray
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

  def __init__(self, meshes: List[config_pb2.Body], body: bodies.Body,
               mesh_geoms: Mapping[str, config_pb2.MeshGeometry]):
    """Initializes a triangular mesh collider.

    Args:
      meshes: Mesh colliders of the body in the config.
      body: The body that the mesh colliders belong to.
      mesh_geoms: The dictionary of the mesh geometries keyed by their names.
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
      vertices.append(vertex)

      # Each face is a triangle.
      face = jp.reshape(jp.take(vertex, g.faces), (-1, 3, 3))
      faces.append(face)

      # Apply rotation to face normals.
      face_normal = jp.array([vec_to_arr(n) for n in g.face_normals])
      face_normal = jp.vmap(
          math.rotate, include=(True, False))(face_normal, rot)
      face_normals.append(face_normal)

    super().__init__(meshes, body, jp.array(vertices), jp.array(faces),
                     jp.array(face_normals))


@pytree.register
class PointMesh(Mesh):
  """A triangular mesh with vertex collidables."""


def closest_segment_point(a: math.Vector3, b: math.Vector3,
                          pt: math.Vector3) -> math.Vector3:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0., 1.) * ab  # pytype: disable=wrong-arg-types  # jax-ndarray


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
  t = jp.clip(t, 0, 1)  # pytype: disable=wrong-arg-types  # jax-ndarray
  segment_point = a + t * (b - a)
  return segment_point


def _closest_segment_to_segment_points(
    a0: math.Vector3, a1: math.Vector3, b0: math.Vector3,
    b1: math.Vector3) -> Tuple[math.Vector3, math.Vector3, float, float]:
  """Returns closest points on two line segments, and the barycentric vars."""
  # Gets the closest segment points by first finding the closest points
  # between two lines. Points are then clipped to be on the line segments
  # and edge cases with clipping are handled.
  dir_a = a1 - a0
  len_a = jp.safe_norm(dir_a)
  len_a += 1e-6 * (len_a == 0.)
  dir_a = dir_a / len_a
  half_len_a = len_a * 0.5

  dir_b = b1 - b0
  len_b = jp.safe_norm(dir_b)
  len_b += 1e-6 * (len_b == 0.)
  dir_b = dir_b / len_b
  half_len_b = len_b * 0.5

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

  orig_t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
  orig_t_b = dirb_dot_trans + orig_t_a * dira_dot_dirb
  t_a = jp.clip(orig_t_a, -half_len_a, half_len_a)
  t_b = jp.clip(orig_t_b, -half_len_b, half_len_b)

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

  # Transform the barycentric coordinate into [0, 1].
  t_a = (orig_t_a + half_len_a) / len_a
  t_b = (orig_t_b + half_len_b) / len_b

  return best_a, best_b, t_a, t_b


def closest_segment_to_segment_points(
    a0: math.Vector3, a1: math.Vector3, b0: math.Vector3,
    b1: math.Vector3) -> Tuple[math.Vector3, math.Vector3]:
  """Returns closest points on two line segments."""
  best_a, best_b, *_ = _closest_segment_to_segment_points(a0, a1, b0, b1)
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


def _project_pt_onto_plane(pt: math.Vector3, plane_pt: math.Vector3,
                           plane_normal: math.Vector3) -> math.Vector3:
  """Projects a point onto a plane."""
  dist = (pt - plane_pt).dot(plane_normal)
  return pt - dist * plane_normal


def _project_poly_onto_plane(poly: jp.ndarray, plane_pt: math.Vector3,
                             plane_normal: math.Vector3) -> jp.ndarray:
  """Projects a polygon onto a plane."""
  return jp.vmap(
      _project_pt_onto_plane, include=[True, False,
                                       False])(poly, plane_pt,
                                               math.normalize(plane_normal))


def _project_poly_onto_poly_plane(poly1: math.Vector3, norm1: math.Vector3,
                                  poly2: math.Vector3,
                                  norm2: math.Vector3) -> math.Vector3:
  """"Projects poly1 onto the poly2 plane along norm1."""
  d = poly2[0].dot(norm2)
  denom = norm1.dot(norm2)
  t = (d - poly1.dot(norm2)) / (denom + 1e-6 * (denom == 0.))
  new_poly = poly1 + t.reshape(-1, 1) * norm1
  return new_poly


def point_in_front_of_plane(plane_pt: math.Vector3, plane_normal: math.Vector3,
                            pt: math.Vector3) -> bool:
  return (pt - plane_pt).dot(plane_normal) > 1e-6


def get_orthogonals(a):
  """Given a normal vector `a`, returns orthogonal vectors `b` and `c`."""
  # TODO reconcile `get_orthogonals` with the impl in `math.py`.
  a_abs = jp.abs(a)
  b = jp.ones_like(a)
  idx = jp.argmax(a_abs)
  denom = a[idx] + 1e-6 * (a[idx] == 0.)
  b = jp.index_update(b, idx, -(a.sum(axis=0) - a[idx]) / denom)
  c = jp.cross(a, b)
  return b, c


def _clip_edge_to_planes(edge_p0, edge_p1, plane_pts, plane_normals):
  """Clips an edge to planes."""
  # We return two clipped points, and a mask to include the new edge or not.
  p0, p1 = edge_p0, edge_p1
  p0_in_front = jp.vmap(jp.dot)(p0 - plane_pts, plane_normals) > 1e-6
  p1_in_front = jp.vmap(jp.dot)(p1 - plane_pts, plane_normals) > 1e-6

  # Get candidate clipped points along line segment (p0, p1) by clipping against
  # all clipping planes.
  candidate_clipped_ps = jp.vmap(
      closest_segment_point_plane,
      include=[False, False, True, True])(p0, p1, plane_pts, plane_normals)

  def clip_edge_point(p0, p1, p0_in_front, clipped_ps):

    @jp.vmap
    def choose_edge_point(in_front, clipped_p):
      return jp.where(in_front, clipped_p, p0)

    # Pick the clipped point if p0 is in front of the clipping plane. Otherwise
    # keep p0 as the edge point.
    new_edge_ps = choose_edge_point(p0_in_front, clipped_ps)

    # Pick the clipped point that is most along the edge direction.
    # This degenerates to picking the original point p0 if p0 is *not* in front
    # of any clipping planes.
    dists = jp.dot(new_edge_ps - p0, p1 - p0)
    new_edge_p = new_edge_ps[jp.argmax(dists)]
    return new_edge_p

  # Clip each edge point.
  new_p0 = clip_edge_point(p0, p1, p0_in_front, candidate_clipped_ps)
  new_p1 = clip_edge_point(p1, p0, p1_in_front, candidate_clipped_ps)
  clipped_pts = jp.array([new_p0, new_p1])

  # Keep the original points if both points are in front of any of the clipping
  # planes, rather than creating a new clipped edge. If the entire subject edge
  # is in front of any clipping plane, we need to grab an edge from the clipping
  # polygon instead.
  both_in_front = p0_in_front & p1_in_front
  mask = ~jp.any(both_in_front)
  new_ps = jp.where(mask, clipped_pts, jp.array([p0, p1]))
  # Mask out crossing clipped edge points.
  mask = jp.where((p0 - p1).dot(new_ps[0] - new_ps[1]) < 0, False, mask)  # pytype: disable=wrong-arg-types  # jax-ndarray
  return new_ps, jp.array([mask, mask])


def clip(clipping_poly: math.Vector3, subject_poly: math.Vector3,
         clipping_normal: math.Vector3,
         subject_normal: math.Vector3) -> Tuple[math.Vector3, jp.ndarray]:
  """Clips a subject polygon against a clipping polygon.

  A parallelized clipping algorithm for convex polygons. The result is a set of
  vertices on the clipped subject polygon in the subject polygon plane.

  Args:
    clipping_poly: The polygon that we use to clip the subject polygon against.
    subject_poly: The polygon that gets clipped.
    clipping_normal: Normal of the clipping polygon.
    subject_normal: Normal of the subject polygon.

  Returns:
    clipped_pts: The clipped polygon.
    mask: True if a point is in the clipping polygon.
  """
  # Get clipping edge points, edge planes, and edge normals.
  clipping_p0 = jp.roll(clipping_poly, 1, axis=0)
  clipping_plane_pts = clipping_p0
  clipping_p1 = clipping_poly
  clipping_plane_normals = jp.vmap(
      jp.cross, include=[False, True])(clipping_normal,
                                       clipping_p1 - clipping_p0)

  # Get subject edge points, edge planes, and edge normals.
  subject_edge_p0 = jp.roll(subject_poly, 1, axis=0)
  subject_plane_pts = subject_edge_p0
  subject_edge_p1 = subject_poly
  subject_plane_normals = jp.vmap(
      jp.cross, include=[False, True])(subject_normal,
                                       subject_edge_p1 - subject_edge_p0)

  # Clip all edges of the subject poly against clipping planes.
  clipped_edges0, masks0 = jp.vmap(
      _clip_edge_to_planes,
      include=[True, True, False,
               False])(subject_edge_p0, subject_edge_p1, clipping_plane_pts,
                       clipping_plane_normals)

  # Project the clipping poly onto the subject plane.
  clipping_p0_s = _project_poly_onto_poly_plane(clipping_p0, clipping_normal,
                                                subject_poly, subject_normal)
  clipping_p1_s = _project_poly_onto_poly_plane(clipping_p1, clipping_normal,
                                                subject_poly, subject_normal)
  # Clip all edges of the clipping poly against subject planes.
  clipped_edges1, masks1 = jp.vmap(
      _clip_edge_to_planes,
      include=[True, True, False,
               False])(clipping_p0_s, clipping_p1_s, subject_plane_pts,
                       subject_plane_normals)

  # Merge the points and reshape.
  clipped_edges = jp.concatenate([clipped_edges0, clipped_edges1])
  masks = jp.concatenate([masks0, masks1])
  clipped_points = clipped_edges.reshape((-1, 3))
  mask = masks.reshape(-1)

  return clipped_points, mask


def _create_sat_edge_contact(a0: math.Vector3, a1: math.Vector3,
                             b0: math.Vector3, b1: math.Vector3,
                             maybe_edge_contact: bool, sep_axis: math.Vector3,
                             sep_dist: float) -> Contact:
  """Creates an edge contact from a separating axis."""
  # We get the closest segment-segment points between the two edges that are
  # creating the separating axis.
  an, bn, ta, tb = _closest_segment_to_segment_points(a0, a1, b0, b1)
  valid_edge_contact = (
      maybe_edge_contact & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1))
  pos = jp.where(valid_edge_contact, bn + (an - bn) * 0.5, jp.zeros_like(a0))  # pytype: disable=wrong-arg-types  # jax-ndarray
  normal = -sep_axis
  penetration = jp.where(valid_edge_contact, -sep_dist, -jp.ones_like(sep_dist))  # pytype: disable=wrong-arg-types  # jax-ndarray
  contact = Contact(
      pos=pos, vel=jp.zeros_like(pos), normal=normal, penetration=penetration)
  # Create 4 contact points, even though we only use the first one. The first
  # contact point is from the edge contact, and the other three are padding
  # since SAT creates up to four contact points.
  mask = jp.array([valid_edge_contact, False, False, False])
  contact = jp.tree_map(lambda x: jp.stack([x] * 4, axis=0), contact)
  contact.penetration = jp.where(mask, penetration, -jp.ones_like(penetration))
  return contact


def _create_sat_contact_manifold(
    clipping_poly: jp.ndarray,
    subject_poly: jp.ndarray,
    clipping_norm: math.Vector3,
    subject_norm: math.Vector3,
    sep_axis_sign: Union[int, jp.ndarray],
) -> Contact:
  """Creates a contact manifold between two convex polygons.

  The polygon faces are expected to have a clockwise winding order so that
  clipping plane normals point away from the polygon center.

  Args:
    clipping_poly: The reference polygon to clip the contact against.
    subject_poly: The subject polygon to clip contacts onto.
    clipping_norm: The clipping polygon normal.
    subject_norm: The subject polygon normal.
    sep_axis_sign: The sign of the separating axis wrt the clipping polygon
      normal.

  Returns:
    contact: Contact object.
  """
  # Clip the subject (incident) face onto the clipping (reference) face.
  # The incident points are clipped points on the subject polygon.
  poly_incident, mask = clip(clipping_poly, subject_poly, clipping_norm,
                             subject_norm)
  # The reference points are clipped points on the clipping polygon.
  poly_ref = _project_poly_onto_plane(poly_incident, clipping_poly[0],
                                      clipping_norm)
  behind_clipping_plane = point_in_front_of_plane(clipping_poly[0],
                                                  -clipping_norm, poly_incident)
  mask = mask & behind_clipping_plane

  # TODO: consider changing this to maximize the manifold area.
  ortho_1, ortho_2 = get_orthogonals(clipping_norm)
  dist_mask = jp.where(mask, 0.0, -1e6)  # pytype: disable=wrong-arg-types  # jax-ndarray
  best_x = jp.argmax(poly_ref.dot(ortho_1) + dist_mask)
  best_nx = jp.argmax(poly_ref.dot(-ortho_1) + dist_mask)
  best_y = jp.argmax(poly_ref.dot(ortho_2) + dist_mask)
  best_ny = jp.argmax(poly_ref.dot(-ortho_2) + dist_mask)
  # Choose up to four contact points.
  best = jp.array([best_x, best_nx, best_y, best_ny])
  contact_pts = jp.take(poly_ref, best)
  mask_pts = jp.take(mask, best)
  penetration_dir = jp.take(poly_incident, best) - contact_pts
  penetration = penetration_dir.dot(-clipping_norm)
  penetration = jp.where(mask_pts, penetration, -jp.ones_like(penetration))

  contact = Contact(
      pos=contact_pts,
      vel=-jp.ones_like(contact_pts),
      normal=jp.stack([sep_axis_sign * clipping_norm] * 4, 0),
      penetration=penetration)

  return contact


def sat_hull_hull(faces_a: jp.ndarray, faces_b: jp.ndarray,
                  vertices_a: jp.ndarray, vertices_b: jp.ndarray,
                  normals_a: jp.ndarray,
                  normals_b: jp.ndarray) -> Tuple[Contact, Contact]:
  """Runs the Separating Axis Test for a pair of hulls.

  Given two convex hulls, the Separating Axis Test finds a separating axis
  between all edge pairs and face pairs. Edge pairs create a single contact
  point and face pairs create a contact manifold (up to four contact points).
  We return both the edge and face contacts. Valid contacts can be checked with
  penetration > 0. Resulting edge contacts should be preferred over face
  contacts.

  Args:
    faces_a: An ndarray of hull A's polygon faces.
    faces_b: An ndarray of hull B's polygon faces.
    vertices_a: Vertices for hull A.
    vertices_b: Vertices for hull B.
    normals_a: Normal vectors for hull A's polygon faces.
    normals_b: Normal vectors for hull B's polygon faces.

  Returns:
    edge_contact: An edge contact.
    face_contact: A face contact.
  """

  @jp.vmap
  def get_edge_axes(poly1, poly2):
    # TODO: consider caching/de-duping edges at system creation time.
    points11 = poly1
    points12 = jp.roll(poly1, 1, axis=0)
    poly_edges1 = points11 - points12
    points21 = poly2
    points22 = jp.roll(poly2, 1, axis=0)
    poly_edges2 = points21 - points22

    edges1 = jp.tile(poly_edges1, reps=(poly_edges2.shape[0], 1))
    points11p = jp.tile(points11, reps=(poly_edges2.shape[0], 1))
    points12p = jp.tile(points12, reps=(poly_edges2.shape[0], 1))

    edges2 = jp.repeat(poly_edges2, repeats=poly_edges1.shape[0], axis=0)
    points21p = jp.repeat(points21, repeats=poly_edges1.shape[0], axis=0)
    points22p = jp.repeat(points22, repeats=poly_edges1.shape[0], axis=0)

    edge_edge_axes = jp.vmap(jp.cross)(edges1, edges2)
    directions = jp.vmap(jp.dot)(points11p - origin_a, edge_edge_axes)
    directions = directions.reshape(-1, 1)
    edge_edge_axes *= jp.where(directions > 0, 1, -1)  # pytype: disable=wrong-arg-types  # jax-ndarray
    return edge_edge_axes, points11p, points12p, points21p, points22p

  def get_face_support(vertices_b, plane_normals_a, plane_points_a):
    dists = jp.vmap(
        lambda n, p, s: jp.vmap(jp.dot)(n, s - p),
        include=[False, False, True])(plane_normals_a, plane_points_a,
                                      vertices_b)
    dists = jp.amin(dists, axis=0)
    best_dist_idx = jp.argmax(dists, axis=0)
    return dists[best_dist_idx], best_dist_idx

  def get_edge_support(vertices_b, plane_normals_a, plane_points_a, edge_b1,
                       edge_b2, aux_dists, mask):
    dists = jp.vmap(
        lambda n, p, s: jp.vmap(jp.dot)(n, s - p),
        include=[False, False, True])(plane_normals_a, plane_points_a,
                                      vertices_b)
    support_idx = jp.argmin(dists, axis=0)
    support_point = vertices_b[support_idx]
    support_dist = jp.vmap(jp.take)(dists.T, support_idx)

    # Mask support points if they are not on the edge that created the
    # separating axis.
    support_point_on_edge_b1 = (edge_b1 - support_point).sum(axis=1) == 0
    support_point_on_edge_b2 = (edge_b2 - support_point).sum(axis=1) == 0
    mask = mask | ~(support_point_on_edge_b1 | support_point_on_edge_b2)

    support_dist = jp.where(mask, -1e6, support_dist)  # pytype: disable=wrong-arg-types  # jax-ndarray
    best_dist_idx = jp.argmax(support_dist + aux_dists, axis=0)
    return support_dist[best_dist_idx], best_dist_idx

  # TODO: considering making the origin and edges static.
  origin_a = jp.mean(vertices_a, axis=0)

  # Get all face normal candidate sep axes, then find the face with min
  # separation.
  dist1, idx1 = get_face_support(vertices_a, normals_b, faces_b[:, 0, :])
  dist2, idx2 = get_face_support(vertices_b, normals_a, faces_a[:, 0, :])
  face_dist = jp.where(dist1 > dist2, dist1, dist2)
  face_idx = jp.where(dist1 > dist2, idx1, idx2)
  ref_face = jp.where(dist1 > dist2, faces_b[face_idx], faces_a[face_idx])
  ref_face_norm = jp.where(dist1 > dist2, normals_b[face_idx],
                           normals_a[face_idx])
  sep_axis_sign = jp.where(dist1 > dist2, 1, -1)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # Get the incident (most antiparallel) face on the other hull.
  incident_faces = jp.where(dist1 > dist2, faces_a, faces_b)
  incident_face_norms = jp.where(dist1 > dist2, normals_a, normals_b)
  dirs = jp.vmap(
      jp.dot, include=[True, False])(incident_face_norms, ref_face_norm)
  incident_face_idx = jp.argmin(dirs)
  incident_face = incident_faces[incident_face_idx]
  incident_face_norm = incident_face_norms[incident_face_idx]

  # Create potential face contact.
  face_contact = _create_sat_contact_manifold(ref_face, incident_face,
                                              ref_face_norm, incident_face_norm,
                                              sep_axis_sign)

  # Get all edge candidate separating axes by enumerating all edge pairs.
  num_faces_a, num_faces_b = faces_a.shape[0], faces_b.shape[0]
  faces_a_r = jp.tile(faces_a, reps=(num_faces_a, 1, 1))
  faces_b_r = jp.repeat(faces_b, repeats=num_faces_b, axis=0)
  edge_normals, edge_a1, edge_a2, edge_b1, edge_b2 = jp.tree_map(
      jp.concatenate, get_edge_axes(faces_a_r, faces_b_r))

  # Get rid of separating axes created by paralell edges.
  bad_edge_axes = jp.all(edge_normals == 0., axis=1)
  # Get rid of separating axes that don't create valid separating planes.
  self_dists = jp.vmap(
      lambda n, p, s: jp.vmap(jp.dot)(n, s - p),
      include=[False, False, True])(edge_normals, edge_a1, vertices_a)
  sep_plane_intersects_hull_a = jp.amax(self_dists, axis=0) > 0
  bad_edge_axes = bad_edge_axes | sep_plane_intersects_hull_a
  # Get the distance between edge-pairs to use in the get_edge_support function.
  edges_a_mid = edge_a1 + (edge_a2 - edge_a1) * 0.5
  edges_b_mid = edge_b1 + (edge_b2 - edge_b1) * 0.5
  midpoint_dists = -jp.vmap(jp.dot)(edges_a_mid - edges_b_mid,
                                    edges_a_mid - edges_b_mid)

  # Find the edge-pair with min separation.
  edge_dist, edge_idx = get_edge_support(vertices_b, edge_normals, edge_a1,
                                         edge_b1, edge_b2, midpoint_dists,
                                         bad_edge_axes)
  edge_normal = edge_normals[edge_idx]
  maybe_edge_contact = edge_dist > face_dist
  edge_normal = edge_normal / jp.safe_norm(edge_normal)

  best_dist = jp.amax(jp.array([edge_dist, face_dist]))
  has_intersection = best_dist < 0

  # Create potential edge contact.
  edge_contact = _create_sat_edge_contact(edge_a1[edge_idx], edge_a2[edge_idx],
                                          edge_b1[edge_idx], edge_b2[edge_idx],
                                          has_intersection & maybe_edge_contact,
                                          edge_normal, edge_dist)

  return edge_contact, face_contact


def rotate_point_axis(qp: QP, rotate_idxs: jp.ndarray, point: jp.ndarray,
                      axis: jp.ndarray, angle: float):
  """Convenience function for rotating qps in world space around a point.

  For a selection of body indices, this function takes those bodies and
  rotates them around some point, around an axis emanating from that point.

  Args:
    qp: (n, ...) Full state qp for system
    rotate_idxs: (m<=n,) integer body idxs to be rotated
    point: (3,) a point in space to rotate about
    axis: (3,) an axis in space to rotate around
    angle: float angle to rotate by

  Returns:
    Full state qp with bodies at rotate_idxs rotated.
  """
  qrot = jp.array(qp.rot.shape[0] * [[1., 0., 0., 0.]])
  qrot[rotate_idxs] = math.quat_rot_axis(axis, angle)
  disp = qp.pos - point
  rotated_pos = jp.vmap(math.rotate)(disp, qrot) + point
  rotated_rot = jp.vmap(math.quat_mul)(qrot, qp.rot)
  qp = qp.replace(pos=rotated_pos, rot=rotated_rot)
  return qp
