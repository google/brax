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

"""Tests for geometry."""

from absl.testing import absltest
from absl.testing import parameterized
from brax.v2 import math
from brax.v2.geometry import math as geom_math
import jax
import jax.numpy as jp
import numpy as np
from scipy import spatial


def _get_rand_point(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(1, 3)
  return verts[0, :]


def _get_rand_line_segment(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(2, 3)
  return verts[0, :], verts[1, :]


def _get_rand_triangle_vertices(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(3, 3)
  return verts[0, :], verts[1, :], verts[2, :]


def _get_rand_dir(seed=None):
  if seed is not None:
    np.random.seed(seed)
  r = np.random.randn(1)
  theta = np.random.random(1) * 2 * np.pi
  a = (np.random.random(1) - 0.5) * 2.0
  phi = np.arccos(a)
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return jp.array([x, y, z]).squeeze()


def _get_rand_convex_polygon(seed=None):
  if seed is not None:
    np.random.seed(seed)
  points = np.random.random((15, 2))
  hull = spatial.ConvexHull(points)
  points = points[hull.vertices]
  points = np.array([[p[0], p[1], 0] for p in points])
  return points


def _minimize(fn, sample_fn, lb, ub, tol, max_iter=20, seed=42):
  """Minimize a function, roughly, using the cross-entropy method."""
  # We can alternatively use scipy.optimize with non-linear constraints here.
  assert lb.shape == ub.shape, 'bounds need to have the same shape'
  np.random.seed(seed)

  i, n = 0, 1_000
  mu = (ub + lb) * 0.5
  sigma = (ub - lb) * 0.5
  size = lb.shape[0]
  val, prev_val = fn(mu), None

  while prev_val is None or np.abs(val - prev_val) > tol:
    params = sample_fn(mu, sigma, n, size, lb, ub)
    vals = np.array([fn(p) for p in params])
    if val < vals.min():  # early exit
      return mu
    idx = vals.argsort()
    best_idx = idx[: int(n * 0.05)]
    mu = params[best_idx].mean(axis=0)
    sigma = params[best_idx].std(axis=0) + 1e-10

    prev_val = val
    val = fn(mu)

    i += 1
    if i == max_iter:
      break

  return mu


def _closest_segment_to_segment_points(a0, a1, b0, b1):
  dir_a = a1 - a0
  len_a = np.sqrt(dir_a.dot(dir_a))
  half_len_a = len_a / 2
  dir_a = dir_a / len_a

  dir_b = b1 - b0
  len_b = np.sqrt(dir_b.dot(dir_b))
  half_len_b = len_b / 2
  dir_b = dir_b / len_b

  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Parametrize both line segments.
  def fn(t):
    best_a = a_mid + dir_a * t[0]
    best_b = b_mid + dir_b * t[1]
    return (best_a - best_b).dot(best_a - best_b)

  def sample_fn(mu, sigma, n, size, lb, ub):
    params = np.random.normal(mu, sigma, size=(n, size))
    params = np.clip(params, lb, ub)
    return params

  lb = np.array([-half_len_a, -half_len_b])
  ub = np.array([half_len_a, half_len_b])
  ta, tb = _minimize(fn, sample_fn, lb, ub, tol=1e-4)
  best_a = a_mid + dir_a * ta
  best_b = b_mid + dir_b * tb
  return best_a, best_b


def _closest_triangle_point(p0, p1, p2, pt):
  # Parametrize the triangle using barycentric coordinates where a point in
  # the triangle Q = w * V_0 + y * V_1 + v * V_2
  # and u + v + w = 1, or w = 1 - (u + v), (u + v) <= 1.
  def fn(t):
    u, v = t[0], t[1]
    w = 1 - u - v
    best_q = u * p0 + v * p1 + w * p2
    return (pt - best_q).dot(pt - best_q)

  def sample_fn(mu, sigma, n, size, lb, ub):
    params = np.random.normal(mu, sigma, size=(n, size))
    params = np.clip(params, lb, ub)
    params[:, 1] = np.clip(params[:, 1], 0.0, 1.0 - params[:, 0])
    return params

  lb = np.array([0, 0])
  ub = np.array([1, 1])
  u, v = _minimize(fn, sample_fn, lb, ub, tol=1e-10)
  best_q = u * p0 + v * p1 + (1 - u - v) * p2
  return best_q


def _closest_segment_triangle_points(a, b, p0, p1, p2):
  # Parametrize line segment.
  dir_ls = b - a
  len_ls = jp.sqrt(dir_ls.dot(dir_ls))
  half_len_ls = len_ls / 2
  dir_ls = dir_ls / len_ls
  ls_mid = a + dir_ls * half_len_ls

  # Parametrize the triangle using barycentric coordinates.
  def fn(t):
    best_ls = ls_mid + dir_ls * t[0]
    u, v = t[1], t[2]
    w = 1 - u - v
    best_q = u * p0 + v * p1 + w * p2
    return (best_ls - best_q).dot(best_ls - best_q)

  def sample_fn(mu, sigma, n, size, lb, ub):
    params = np.random.normal(mu, sigma, size=(n, size))
    params = np.clip(params, lb, ub)
    params[:, 2] = np.clip(params[:, 2], 0.0, 1.0 - params[:, 1])
    return params

  lb = np.array([-half_len_ls, 0, 0])
  ub = np.array([half_len_ls, 1, 1])
  t_ls, u, v = _minimize(fn, sample_fn, lb, ub, tol=1e-10)
  best_ls = ls_mid + dir_ls * t_ls
  best_q = u * p0 + v * p1 + (1 - u - v) * p2
  return best_ls, best_q


class ClosestSegmentSegmentPointsTest(absltest.TestCase):
  """Tests for closest segment-to-segment points."""

  def test_closest_segments_points(self):
    a0 = jp.array([0.73432405, 0.12372768, 0.20272314])
    a1 = jp.array([1.10600128, 0.88555209, 0.65209485])
    b0 = jp.array([0.85599262, 0.61736299, 0.9843583])
    b1 = jp.array([1.84270939, 0.92891793, 1.36343326])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [1.09063, 0.85404, 0.63351], 5)
    self.assertSequenceAlmostEqual(best_b, [0.99596, 0.66156, 1.03813], 5)

  def test_intersecting_segments(self):
    """Tests segments that intersect."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([-1.0, 0.0, 0.0]), jp.array([1.0, 0.0, 0.0])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)

  def test_intersecting_lines(self):
    """Tests that intersecting lines (not segments) get clipped."""
    a0, a1 = jp.array([0.2, 0.2, 0.0]), jp.array([1.0, 1.0, 0.0])
    b0, b1 = jp.array([0.2, 0.4, 0.0]), jp.array([1.0, 2.0, 0.0])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.3, 0.3, 0.0], 2)
    self.assertSequenceAlmostEqual(best_b, [0.2, 0.4, 0.0], 2)

  def test_parallel_segments(self):
    """Tests that parallel segments have closest points at the midpoint."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([1.0, 0.0, -1.0]), jp.array([1.0, 0.0, 1.0])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.0], 5)

  def test_parallel_offset_segments(self):
    """Tests that offset parallel segments are close at segment endpoints."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([1.0, 0.0, 1.0]), jp.array([1.0, 0.0, 3.0])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 1.0], 5)

  def test_zero_length_segments(self):
    """Test that zero length segments don't return NaNs."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, -1.0])
    b0, b1 = jp.array([1.0, 0.0, 0.1]), jp.array([1.0, 0.0, 0.1])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, -1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.1], 5)

  def test_overlapping_segments(self):
    """Tests that perfectly overlapping segments intersect at the midpoints."""
    a0, a1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    b0, b1 = jp.array([0.0, 0.0, -1.0]), jp.array([0.0, 0.0, 1.0])
    best_a, best_b = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)


class GeometryScipyTest(parameterized.TestCase):
  """Tests for geometry functions against scipy equivalents."""

  params = list(zip(np.repeat(np.arange(10), 10), np.tile(np.arange(10), 10)))

  @parameterized.parameters(*params)
  def test_closest_segment_to_segment_points(self, i, j):
    a0, a1 = _get_rand_line_segment(i)
    b0, b1 = _get_rand_line_segment(j)
    expected = _closest_segment_to_segment_points(a0, a1, b0, b1)
    ans = geom_math.closest_segment_to_segment_points(a0, a1, b0, b1)
    expected_dist = (expected[0] - expected[1]).dot(expected[0] - expected[1])
    test_dist = (ans[0] - ans[1]).dot(ans[0] - ans[1])
    self.assertAlmostEqual(expected_dist, test_dist, 4)

  @parameterized.parameters(*params)
  def test_closest_triangle_point(self, i, j):
    pt = _get_rand_point(i)
    p0, p1, p2 = _get_rand_triangle_vertices(j)
    expected = _closest_triangle_point(p0, p1, p2, pt)
    ans = geom_math.closest_triangle_point(p0, p1, p2, pt)
    self.assertSequenceAlmostEqual(expected, ans, 4)

  @parameterized.parameters(*params)
  def test_closest_segment_triangle_points(self, i, j):
    a, b = _get_rand_line_segment(i)
    p0, p1, p2 = _get_rand_triangle_vertices(j)
    expected = _closest_segment_triangle_points(a, b, p0, p1, p2)
    triangle_normal, _ = math.normalize(jp.cross(p0 - p1, p2 - p1))
    ans = geom_math.closest_segment_triangle_points(
        a, b, p0, p1, p2, triangle_normal
    )
    expected_dist = (expected[0] - expected[1]).dot(expected[0] - expected[1])
    test_dist = (ans[0] - ans[1]).dot(ans[0] - ans[1])
    self.assertAlmostEqual(expected_dist, test_dist, 1)
    # Guarantee we are lower than an upper bound.
    self.assertLessEqual(test_dist, expected_dist + 1e-5)


def _check_eq_pts(pts1, pts2, atol=1e-6):
  # For every point in pts1, make sure we have a point in pts2
  # that is close within `atol`, and vice versa.
  if not pts1.size and not pts2.size:
    return True
  elif not pts1.size:
    return False
  elif not pts2.size:
    return False
  if pts1.shape[-1] != 3 or pts2.shape[-1] != 3:
    raise AssertionError('Points should be three dimensional.')
  eq = True
  for p1 in pts1:
    eq = eq and jp.any(jp.sum(jp.abs(p1 - pts2) < atol, axis=-1) == 3)
  for p2 in pts2:
    eq = eq and jp.any(jp.sum(jp.abs(p2 - pts1) < atol, axis=-1) == 3)
  return eq


def _clip(clipping_polygon, subject_polygon):
  """Returns the clipped subject polygon, using Sutherland-Hodgman Clipping."""
  polygon1 = clipping_polygon
  polygon1_normal = jp.cross(
      polygon1[-1] - polygon1[0], polygon1[1] - polygon1[0]
  )
  clipping_edges = [
      (polygon1[i - 1], polygon1[i]) for i in range(len(polygon1))
  ]
  # Clipping plane normals point away from the clipping poly center (polygon
  # points assumed to have a clockwise winding order).
  clipping_planes = [
      (e0, jp.cross(polygon1_normal, e1 - e0)) for e0, e1 in clipping_edges
  ]

  output_polygon = subject_polygon
  for clipping_plane in clipping_planes:
    input_ = output_polygon
    output_polygon = []

    starting_pt = input_[-1]
    clipping_plane_pt, clipping_plane_normal = clipping_plane

    for endpt in input_:
      intersection_pt = geom_math.closest_segment_point_plane(
          starting_pt, endpt, clipping_plane_pt, clipping_plane_normal
      )
      starting_pt_front = geom_math.point_in_front_of_plane(
          clipping_plane_pt, clipping_plane_normal, starting_pt
      )
      endpt_front = geom_math.point_in_front_of_plane(
          clipping_plane_pt, clipping_plane_normal, endpt
      )

      if not endpt_front and not starting_pt_front:
        output_polygon.append(endpt)
      elif not endpt_front and starting_pt_front:
        output_polygon.append(intersection_pt)
        output_polygon.append(endpt)
      elif not starting_pt_front:
        output_polygon.append(intersection_pt)

      starting_pt = endpt

    if not output_polygon:
      # All clipping points outside the subject polygon.
      return jp.array(output_polygon)

  return jp.array(output_polygon)


class ClippingTest(parameterized.TestCase):
  """Tests that the clipping algorithm matches a baseline implementation."""

  clip_vectorized = jax.jit(geom_math.clip)

  @parameterized.parameters(range(100))
  def test_clipped_triangles(self, i):
    subject_poly = jp.array(_get_rand_triangle_vertices(i))
    clipping_poly = jp.array(_get_rand_triangle_vertices(i + 1))

    poly_out = _clip(clipping_poly, subject_poly)

    clipping_normal = jp.cross(
        clipping_poly[1] - clipping_poly[0],
        clipping_poly[-1] - clipping_poly[0],
    )
    subject_normal = jp.cross(
        subject_poly[1] - subject_poly[0], subject_poly[-1] - subject_poly[0]
    )
    poly_out_jax, mask = ClippingTest.clip_vectorized(
        clipping_poly, subject_poly, clipping_normal, subject_normal
    )

    self.assertTrue(
        _check_eq_pts(poly_out, poly_out_jax[mask], atol=1e-4),
        f'Clipped triangles {i} did not match.',
    )

  @parameterized.parameters(range(100))
  def test_clipped_hulls(self, i):
    # The hulls are all in the x-y plane, unlike in `test_clipped_triangles`.
    subject_poly = jp.array(_get_rand_convex_polygon(i))
    clipping_poly = jp.array(_get_rand_convex_polygon(i + 1))

    poly_out = _clip(clipping_poly, subject_poly)

    clipping_normal = jp.cross(
        clipping_poly[1] - clipping_poly[0],
        clipping_poly[-1] - clipping_poly[0],
    )
    subject_normal = jp.cross(
        subject_poly[1] - subject_poly[0],
        subject_poly[-1] - subject_poly[0],
    )
    poly_out_jax, mask = ClippingTest.clip_vectorized(
        clipping_poly, subject_poly, clipping_normal, subject_normal
    )

    self.assertTrue(
        _check_eq_pts(poly_out, poly_out_jax[mask], atol=1e-2),
        f'Clipped hulls {i} did not match.',
    )


class ManifoldPointsTest(parameterized.TestCase):
  """Tests manifold point selection."""

  def test_manifold_points(self):
    poly = jp.array([
        [0.99999994, 0.14842263, 0.39985055],
        [0.8585786, 0.00145163, 0.39985055],
        [1.0, -0.14551926, 0.39985055],
        [1.1414213, 0.00145174, 0.39985055],
    ])
    poly_mask = jp.array([False, True, True, True])
    poly_norm = jp.array([0.0, 0.0, 1.0])
    idx = geom_math.manifold_points(poly, poly_mask, poly_norm)
    self.assertSequenceEqual(idx.tolist(), [1, 3, 1, 2])


if __name__ == '__main__':
  absltest.main()
