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

"""Geometry functions."""

from typing import Tuple
from brax.v2 import math
from jax import numpy as jp


def closest_segment_point(
    a: jp.ndarray, b: jp.ndarray, pt: jp.ndarray
) -> jp.ndarray:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-6)
  return a + jp.clip(t, 0.0, 1.0) * ab


def closest_segment_point_and_dist(
    a: jp.ndarray, b: jp.ndarray, pt: jp.ndarray
) -> Tuple[jp.ndarray, jp.ndarray]:
  """Returns closest point on the line segment and the distance squared."""
  closest = closest_segment_point(a, b, pt)
  dist = (pt - closest).dot(pt - closest)
  return closest, dist


def closest_segment_to_segment_points(
    a0: jp.ndarray, a1: jp.ndarray, b0: jp.ndarray, b1: jp.ndarray
) -> Tuple[jp.ndarray, jp.ndarray]:
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
    a: jp.ndarray, b: jp.ndarray, p0: jp.ndarray, plane_normal: jp.ndarray
) -> jp.ndarray:
  """Gets the closest point between a line segment and a plane."""
  # If a line segment is parametrized as S(t) = a + t * (b - a), we can
  # plug it into the plane equation dot(n, S(t)) - d = 0, then solve for t to
  # get the line-plane intersection. We then clip t to be in [0, 1] to be on
  # the line segment.
  n = plane_normal
  d = jp.dot(p0, n)  # shortest distance from origin to plane
  denom = jp.dot(n, b - a)
  t = (d - jp.dot(n, a)) / (denom + 1e-6 * (denom == 0.0))
  t = jp.clip(t, 0, 1)
  segment_point = a + t * (b - a)

  return segment_point


def closest_triangle_point(
    p0: jp.ndarray, p1: jp.ndarray, p2: jp.ndarray, pt: jp.ndarray
) -> jp.ndarray:
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
    a: jp.ndarray,
    b: jp.ndarray,
    p0: jp.ndarray,
    p1: jp.ndarray,
    p2: jp.ndarray,
    triangle_normal: jp.ndarray,
) -> Tuple[jp.ndarray, jp.ndarray]:
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
