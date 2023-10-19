# Copyright 2023 The Brax Authors.
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

"""Geometry functions."""

from typing import Tuple

from brax import math
from brax.base import Contact
import jax
from jax import numpy as jp


def closest_segment_point(
    a: jax.Array, b: jax.Array, pt: jax.Array
) -> jax.Array:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0.0, 1.0) * ab


def closest_segment_point_and_dist(
    a: jax.Array, b: jax.Array, pt: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns closest point on the line segment and the distance squared."""
  closest = closest_segment_point(a, b, pt)
  dist = (pt - closest).dot(pt - closest)
  return closest, dist


def closest_line_point(a: jax.Array, b: jax.Array, pt: jax.Array) -> jax.Array:
  """Returns the closest point on the a-b line to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + t * ab


def closest_segment_to_segment_points(
    a0: jax.Array, a1: jax.Array, b0: jax.Array, b1: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns closest points on two line segments."""
  # Gets the closest segment points by first finding the closest points
  # between two lines. Points are then clipped to be on the line segments
  # and edge cases with clipping are handled.
  dir_a, len_a = math.normalize(a1 - a0)
  dir_b, len_b = math.normalize(b1 - b0)

  # Segment mid-points.
  half_len_a = len_a * 0.5
  half_len_b = len_b * 0.5
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

  return best_a, best_b


def closest_segment_point_plane(
    a: jax.Array, b: jax.Array, p0: jax.Array, plane_normal: jax.Array
) -> jax.Array:
  """Gets the closest point between a line segment and a plane."""
  # If a line segment is parametrized as S(t) = a + t * (b - a), we can
  # plug it into the plane equation dot(n, S(t)) - d = 0, then solve for t to
  # get the line-plane intersection. We then clip t to be in [0, 1] to be on
  # the line segment.
  n = plane_normal
  d = jp.sum(p0 * n)  # shortest distance from origin to plane
  denom = jp.sum(n * (b - a))
  t = (d - jp.sum(n * a)) / (denom + 1e-6 * (denom == 0.0))
  t = jp.clip(t, 0, 1)
  segment_point = a + t * (b - a)

  return segment_point


def closest_triangle_point(
    p0: jax.Array, p1: jax.Array, p2: jax.Array, pt: jax.Array
) -> jax.Array:
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
  det = a * c - b * b
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

  return closest_p


def closest_segment_triangle_points(
    a: jax.Array,
    b: jax.Array,
    p0: jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    triangle_normal: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
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


def project_pt_onto_plane(
    pt: jax.Array, plane_pt: jax.Array, plane_normal: jax.Array
) -> jax.Array:
  """Projects a point onto a plane along the plane normal."""
  dist = (pt - plane_pt).dot(plane_normal)
  return pt - dist * plane_normal


def _project_poly_onto_plane(
    poly: jax.Array, plane_pt: jax.Array, plane_normal: jax.Array
) -> jax.Array:
  """Projects a polygon onto a plane using the plane normal."""
  return jax.vmap(project_pt_onto_plane, in_axes=[0, None, None])(
      poly, plane_pt, math.normalize(plane_normal)[0]
  )


def _project_poly_onto_poly_plane(
    poly1: jax.Array, norm1: jax.Array, poly2: jax.Array, norm2: jax.Array
) -> jax.Array:
  """Projects poly1 onto the poly2 plane along poly1's normal."""
  d = poly2[0].dot(norm2)
  denom = norm1.dot(norm2)
  t = (d - poly1.dot(norm2)) / (denom + 1e-6 * (denom == 0.0))
  new_poly = poly1 + t.reshape(-1, 1) * norm1
  return new_poly


def point_in_front_of_plane(
    plane_pt: jax.Array, plane_normal: jax.Array, pt: jax.Array
) -> bool:
  """Checks if a point is strictly in front of a plane."""
  return (pt - plane_pt).dot(plane_normal) > 1e-6  # pytype: disable=bad-return-type  # jax-ndarray


def clip_edge_to_planes(
    edge_p0: jax.Array,
    edge_p1: jax.Array,
    plane_pts: jax.Array,
    plane_normals: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Clips an edge against side planes.

  We return two clipped points, and a mask to include the new edge or not.

  Args:
    edge_p0: the first point on the edge
    edge_p1: the second point on the edge
    plane_pts: side plane points
    plane_normals: side plane normals

  Returns:
    new_ps: new edge points that are clipped against side planes
    mask: a boolean mask, True if an edge point is a valid clipped point and
    False otherwise
  """
  p0, p1 = edge_p0, edge_p1
  p0_in_front = jax.vmap(jp.dot)(p0 - plane_pts, plane_normals) > 1e-6
  p1_in_front = jax.vmap(jp.dot)(p1 - plane_pts, plane_normals) > 1e-6

  # Get candidate clipped points along line segment (p0, p1) by clipping against
  # all clipping planes.
  candidate_clipped_ps = jax.vmap(
      closest_segment_point_plane, in_axes=[None, None, 0, 0]
  )(p0, p1, plane_pts, plane_normals)

  def clip_edge_point(p0, p1, p0_in_front, clipped_ps):
    @jax.vmap
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
  mask = jp.where((p0 - p1).dot(new_ps[0] - new_ps[1]) < 0, False, mask)
  return new_ps, jp.array([mask, mask])


def clip(
    clipping_poly: jax.Array,
    subject_poly: jax.Array,
    clipping_normal: jax.Array,
    subject_normal: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
  """Clips a subject polygon against a clipping polygon.

  A parallelized clipping algorithm for convex polygons. The result is a set of
  vertices on the clipped subject polygon in the subject polygon plane.

  Args:
    clipping_poly: the polygon that we use to clip the subject polygon against
    subject_poly: the polygon that gets clipped
    clipping_normal: normal of the clipping polygon
    subject_normal: normal of the subject polygon

  Returns:
    clipped_pts: points on the clipped polygon
    mask: True if a point is in the clipping polygon, False otherwise
  """
  # Get clipping edge points, edge planes, and edge normals.
  clipping_p0 = jp.roll(clipping_poly, 1, axis=0)
  clipping_plane_pts = clipping_p0
  clipping_p1 = clipping_poly
  clipping_plane_normals = jax.vmap(jp.cross, in_axes=[0, None])(
      clipping_p1 - clipping_p0,
      clipping_normal,
  )

  # Get subject edge points, edge planes, and edge normals.
  subject_edge_p0 = jp.roll(subject_poly, 1, axis=0)
  subject_plane_pts = subject_edge_p0
  subject_edge_p1 = subject_poly
  subject_plane_normals = jax.vmap(jp.cross, in_axes=[0, None])(
      subject_edge_p1 - subject_edge_p0,
      subject_normal,
  )

  # Clip all edges of the subject poly against clipping side planes.
  clipped_edges0, masks0 = jax.vmap(
      clip_edge_to_planes, in_axes=[0, 0, None, None]
  )(
      subject_edge_p0,
      subject_edge_p1,
      clipping_plane_pts,
      clipping_plane_normals,
  )

  # Project the clipping poly onto the subject plane.
  clipping_p0_s = _project_poly_onto_poly_plane(
      clipping_p0, clipping_normal, subject_poly, subject_normal
  )
  # TODO consider doing a roll here instead of projection.
  clipping_p1_s = _project_poly_onto_poly_plane(
      clipping_p1, clipping_normal, subject_poly, subject_normal
  )

  # Clip all edges of the clipping poly against subject planes.
  clipped_edges1, masks1 = jax.vmap(
      clip_edge_to_planes, in_axes=[0, 0, None, None]
  )(clipping_p0_s, clipping_p1_s, subject_plane_pts, subject_plane_normals)

  # Merge the points and reshape.
  clipped_edges = jp.concatenate([clipped_edges0, clipped_edges1])
  masks = jp.concatenate([masks0, masks1])
  clipped_points = clipped_edges.reshape((-1, 3))
  mask = masks.reshape(-1)

  return clipped_points, mask


def manifold_points(
    poly: jax.Array, poly_mask: jax.Array, poly_norm: jax.Array
) -> jax.Array:
  """Chooses four points on the polygon with approximately maximal area."""
  dist_mask = jp.where(poly_mask, 0.0, -1e6)
  a_idx = jp.argmax(dist_mask)
  a = poly[a_idx]
  # choose point b furthest from a
  b_idx = (((a - poly) ** 2).sum(axis=1) + dist_mask).argmax()
  b = poly[b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = jp.cross(poly_norm, a - b)
  ap = a - poly
  c_idx = (jp.abs(ap.dot(ab)) + dist_mask).argmax()
  c = poly[c_idx]
  # choose point d furthest from the other two triangle edges
  ac = jp.cross(poly_norm, a - c)
  bc = jp.cross(poly_norm, b - c)
  bp = b - poly
  dist_bp = jp.abs(bp.dot(bc)) + dist_mask
  dist_ap = jp.abs(ap.dot(ac)) + dist_mask
  d_idx = jp.concatenate([dist_bp, dist_ap]).argmax() % poly.shape[0]
  return jp.array([a_idx, b_idx, c_idx, d_idx])


def _create_contact_manifold(
    clipping_poly: jax.Array,
    subject_poly: jax.Array,
    clipping_norm: jax.Array,
    subject_norm: jax.Array,
    sep_axis: jax.Array,
) -> Contact:
  """Creates a contact manifold between two convex polygons.

  The polygon faces are expected to have a counter clockwise winding order so
  that clipping plane normals point away from the polygon center.

  Args:
    clipping_poly: The reference polygon to clip the contact against.
    subject_poly: The subject polygon to clip contacts onto.
    clipping_norm: The clipping polygon normal.
    subject_norm: The subject polygon normal.
    sep_axis: The separating axis

  Returns:
    contact: Contact object.
  """
  # Clip the subject (incident) face onto the clipping (reference) face.
  # The incident points are clipped points on the subject polygon.
  poly_incident, mask = clip(
      clipping_poly, subject_poly, clipping_norm, subject_norm
  )
  # The reference points are clipped points on the clipping polygon.
  poly_ref = _project_poly_onto_plane(
      poly_incident, clipping_poly[0], clipping_norm
  )
  behind_clipping_plane = point_in_front_of_plane(
      clipping_poly[0], -clipping_norm, poly_incident
  )
  mask = mask & behind_clipping_plane

  # Choose four contact points.
  best = manifold_points(poly_ref, mask, clipping_norm)
  contact_pts = jp.take(poly_ref, best, axis=0)
  mask_pts = jp.take(mask, best, axis=0)
  penetration_dir = jp.take(poly_incident, best, axis=0) - contact_pts
  penetration = penetration_dir.dot(-clipping_norm)
  penetration = jp.where(mask_pts, penetration, -jp.ones_like(penetration))

  contact = Contact(  # pytype: disable=wrong-arg-types  # jnp-type
      pos=contact_pts,
      normal=jp.stack([sep_axis] * 4, 0),
      penetration=penetration,
      friction=jp.array([]),
      elasticity=jp.array([]),
      solver_params=jp.array([]),
      link_idx=jp.array([]),
  )

  return contact


def sat_hull_hull(
    faces_a: jax.Array,
    faces_b: jax.Array,
    vertices_a: jax.Array,
    vertices_b: jax.Array,
    normals_a: jax.Array,
    normals_b: jax.Array,
    unique_edges_a: jax.Array,
    unique_edges_b: jax.Array,
) -> Contact:
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
    unique_edges_a: Unique edges for hull A.
    unique_edges_b: Unique edges for hull B.

  Returns:
    contact: A contact.
  """
  # get the separating axes
  edge_dir_a = unique_edges_a[:, 0] - unique_edges_a[:, 1]
  edge_dir_b = unique_edges_b[:, 0] - unique_edges_b[:, 1]
  edge_dir_a_r = jp.tile(edge_dir_a, reps=(unique_edges_b.shape[0], 1))
  edge_dir_b_r = jp.repeat(edge_dir_b, repeats=unique_edges_a.shape[0], axis=0)
  edge_edge_axes = jax.vmap(jp.cross)(edge_dir_a_r, edge_dir_b_r)
  edge_edge_axes = jax.vmap(lambda x: math.normalize(x, axis=0)[0])(
      edge_edge_axes
  )

  axes = jp.concatenate([normals_a, normals_b, edge_edge_axes])

  # for each separating axis, get the support
  @jax.vmap
  def get_support(axis):
    support_a = jax.vmap(jp.dot, in_axes=[None, 0])(axis, vertices_a)
    support_b = jax.vmap(jp.dot, in_axes=[None, 0])(axis, vertices_b)
    dist1 = support_a.max() - support_b.min()
    dist2 = support_b.max() - support_a.min()
    sign = jp.where(dist1 > dist2, -1, 1)
    dist = jp.minimum(dist1, dist2)
    dist = jp.where(~jp.all(axis == 0.0), dist, 1e6)  # degenerate axis
    return dist, sign

  support, sign = get_support(axes)

  # choose the best separating axis
  best_idx = jp.argmin(support)
  best_sign = sign[best_idx]
  best_axis = axes[best_idx]
  is_edge_contact = best_idx >= (normals_a.shape[0] + normals_b.shape[0])

  # get the (reference) face most aligned with the separating axis
  dist_a = jax.vmap(jp.dot, in_axes=[None, 0])(best_axis, normals_a)
  dist_b = jax.vmap(jp.dot, in_axes=[None, 0])(best_axis, normals_b)
  a_max = dist_a.argmax()
  b_max = dist_b.argmax()
  a_min = dist_a.argmin()
  b_min = dist_b.argmin()

  ref_face = jp.where(best_sign > 0, faces_a[a_max], faces_b[b_max])
  ref_face_norm = jp.where(best_sign > 0, normals_a[a_max], normals_b[b_max])
  incident_face = jp.where(best_sign > 0, faces_b[b_min], faces_a[a_min])
  incident_face_norm = jp.where(
      best_sign > 0, normals_b[b_min], normals_a[a_min]
  )

  contact = _create_contact_manifold(
      ref_face,
      incident_face,
      ref_face_norm,
      incident_face_norm,
      -best_sign * best_axis,
  )

  # For edge contacts, we use the clipped face point, mainly for performance
  # reasons. For small penetration, the clipped face point is roughly the edge
  # contact point.
  # TODO revisit edge contact pos (for deep penetration) with same perf
  idx = contact.penetration.argmax()
  contact = contact.replace(
      penetration=jp.where(
          is_edge_contact,
          jp.array([contact.penetration[idx], -1, -1, -1]),
          contact.penetration,
      ),
      pos=jp.where(
          is_edge_contact, jp.tile(contact.pos[idx], (4, 1)), contact.pos
      ),
  )

  return contact
