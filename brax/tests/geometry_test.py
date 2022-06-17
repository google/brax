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

"""Tests for brax.physics.geometry."""

from absl.testing import absltest
import brax
from brax import jumpy as jp
from brax.physics import geometry as geom
import numpy as np
import scipy


def _get_rand_triangle_vertices(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(3, 3)
  return verts[0, :], verts[1, :], verts[2, :]


def _get_rand_line_segment(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(2, 3)
  return verts[0, :], verts[1, :]


def _get_rand_point(seed=None):
  if seed is not None:
    np.random.seed(seed)
  verts = np.random.randn(1, 3)
  return verts[0, :]


def _scipy_closest_triangle_point(p0, p1, p2, pt):
  # Parametrize the triangle using barycentric coordinates where a point in
  # the triangle Q = w * V_0 + y * V_1 + v * V_2
  # and u + v + w = 1, or w = 1 - (u + v), (u + v) <= 1.
  def fn(t):
    u, v = t[0], t[1]
    best_q = u * p0 + v * p1 + (1 - u - v) * p2
    return (pt - best_q).dot(pt - best_q)

  constraint = scipy.optimize.NonlinearConstraint(
      lambda x: [x[0], x[1], 1 - x[0] - x[1]],
      lb=np.array([0, 0, 0]),
      ub=np.array([1, 1, 1]))
  u, v = scipy.optimize.minimize(
      fn, jp.zeros(2), tol=1e-10, constraints=[constraint]).x
  best_q = u * p0 + v * p1 + (1 - u - v) * p2
  return best_q


def _scipy_closest_segment_to_segment_points(a0, a1, b0, b1):
  dir_a = a1 - a0
  len_a = jp.sqrt(dir_a.dot(dir_a))
  half_len_a = len_a / 2
  dir_a = dir_a / len_a

  dir_b = b1 - b0
  len_b = jp.sqrt(dir_b.dot(dir_b))
  half_len_b = len_b / 2
  dir_b = dir_b / len_b

  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Parametrize both line segments.
  def fn(t):
    best_a = a_mid + dir_a * t[0]
    best_b = b_mid + dir_b * t[1]
    return (best_a - best_b).dot(best_a - best_b)

  constraint = scipy.optimize.NonlinearConstraint(
      lambda x: x,
      lb=np.array([-half_len_a, -half_len_b]),
      ub=np.array([half_len_a, half_len_b]))
  ta, tb = scipy.optimize.minimize(
      fn, np.zeros(2), tol=1e-10, constraints=[constraint]).x
  best_a = a_mid + dir_a * ta
  best_b = b_mid + dir_b * tb
  return best_a, best_b


def _scipy_closest_segment_triangle_points(a, b, p0, p1, p2):
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
    best_q = u * p0 + v * p1 + (1 - u - v) * p2
    return (best_ls - best_q).dot(best_ls - best_q)

  constraint = scipy.optimize.NonlinearConstraint(
      lambda x: [x[0], x[1], x[2], 1 - x[1] - x[2]],
      lb=np.array([-half_len_ls, 0, 0, 0]),
      ub=np.array([half_len_ls, 1, 1, 1]))
  t_ls, u, v = scipy.optimize.minimize(
      fn, jp.zeros(3), tol=1e-10, constraints=[constraint]).x
  best_ls = ls_mid + dir_ls * t_ls
  best_q = u * p0 + v * p1 + (1 - u - v) * p2
  return best_ls, best_q


class ClosestSegmentPointsTest(absltest.TestCase):
  """Tests for closest segment points."""

  def test_closest_segments_points(self):
    a0 = jp.array([0.73432405, 0.12372768, 0.20272314])
    a1 = jp.array([1.10600128, 0.88555209, 0.65209485])
    b0 = jp.array([0.85599262, 0.61736299, 0.9843583])
    b1 = jp.array([1.84270939, 0.92891793, 1.36343326])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [1.09063, 0.85404, 0.63351], 5)
    self.assertSequenceAlmostEqual(best_b, [0.99596, 0.66156, 1.03813], 5)

  def test_intersecting_segments(self):
    """Tests segments that intersect."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([-1., 0., 0.]), jp.array([1., 0., 0.])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [0., 0., 0.], 5)

  def test_intersecting_lines(self):
    """Tests that intersecting lines (not segments) get clipped."""
    a0, a1 = jp.array([0.2, 0.2, 0.]), jp.array([1., 1., 0.])
    b0, b1 = jp.array([0.2, 0.4, 0.]), jp.array([1., 2., 0.])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.3, 0.3, 0.], 2)
    self.assertSequenceAlmostEqual(best_b, [0.2, 0.4, 0.], 2)

  def test_parallel_segments(self):
    """Tests that parallel segments have closest points at the midpoint."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([1., 0., -1.]), jp.array([1., 0., 1.])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 0.], 5)

  def test_parallel_offset_segments(self):
    """Tests that offset parallel segments are close at segment endpoints."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([1., 0., 1.]), jp.array([1., 0., 3.])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 1.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 1.], 5)

  def test_zero_length_segments(self):
    """Test that zero length segments don't return NaNs."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., -1.])
    b0, b1 = jp.array([1., 0., 0.1]), jp.array([1., 0., 0.1])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., -1.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 0.1], 5)

  def test_overlapping_segments(self):
    """Tests that perfectly overlapping segments intersect at the midpoints."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    best_a, best_b = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [0., 0., 0.], 5)


class GeometryScipyTest(absltest.TestCase):
  """Tests for geometry functions against scipy equivalents."""

  def test_closest_triangle_point(self):
    for i in range(10):
      for j in range(10):
        pt = _get_rand_point(i)
        p0, p1, p2 = _get_rand_triangle_vertices(j)
        expected = _scipy_closest_triangle_point(p0, p1, p2, pt)
        ans = geom.closest_triangle_point(p0, p1, p2, pt)
        self.assertSequenceAlmostEqual(expected, ans, 4)

  def test_closest_segment_to_segment_points(self):
    for i in range(10):
      for j in range(10):
        a0, a1 = _get_rand_line_segment(i)
        b0, b1 = _get_rand_line_segment(j)
        expected = _scipy_closest_segment_to_segment_points(a0, a1, b0, b1)
        ans = geom.closest_segment_to_segment_points(a0, a1, b0, b1)
        expected_dist = (expected[0] - expected[1]).dot(expected[0] -
                                                        expected[1])
        test_dist = (ans[0] - ans[1]).dot(ans[0] - ans[1])
        self.assertAlmostEqual(expected_dist, test_dist, 4)

  def test_closest_segment_triangle_points(self):
    for i in range(10):
      for j in range(10):
        a, b = _get_rand_line_segment(i)
        p0, p1, p2 = _get_rand_triangle_vertices(j)
        expected = _scipy_closest_segment_triangle_points(a, b, p0, p1, p2)
        triangle_normal = brax.math.normalize(jp.cross(p0 - p1, p2 - p1))
        ans = geom.closest_segment_triangle_points(a, b, p0, p1, p2,
                                                   triangle_normal)
        expected_dist = (expected[0] - expected[1]).dot(expected[0] -
                                                        expected[1])
        test_dist = (ans[0] - ans[1]).dot(ans[0] - ans[1])
        self.assertAlmostEqual(expected_dist, test_dist, 4)


if __name__ == '__main__':
  absltest.main()
