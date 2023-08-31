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

# pylint:disable=g-multiple-import
"""Tests for mesh.py."""

from absl.testing import absltest
from brax.base import Box, Convex, Mesh, Transform
from brax.geometry import mesh
import numpy as np


class BoxTest(absltest.TestCase):

  def test_box(self):
    """Tests a triangulated box."""
    b = Box(
        halfsize=np.repeat(0.5, 3),
        link_idx=None,
        transform=None,
        friction=0.42,
        elasticity=1,
        solver_params=None,
    )
    m = mesh.box_tri(b)
    self.assertIsInstance(m, Mesh)
    self.assertSequenceEqual(m.vert.shape, (8, 3))  # eight box corners
    self.assertEqual(np.unique(np.abs(m.vert)), 0.5)
    self.assertSequenceEqual(m.face.shape, (12, 3))  # two triangles per face
    self.assertEqual(m.friction, 0.42)

    expected_face_norm = [
        [0, -1.0, 0],  # left
        [0, -1.0, 0],
        [0, 0, -1.0],  # bottom
        [0, 0, -1.0],
        [+1.0, 0, 0],  # front
        [+1.0, 0, 0],
        [0, +1.0, 0],  # right
        [0, +1.0, 0],
        [0, 0, +1.0],  # top
        [0, 0, +1.0],
        [-1.0, 0, 0],  # back
        [-1.0, 0, 0],
    ]
    face_norm = mesh.get_face_norm(m.vert, m.face)
    np.testing.assert_array_almost_equal(face_norm, expected_face_norm)

  def test_box_hull(self):
    """Tests a polygon box."""
    b = Box(
        halfsize=np.repeat(0.5, 3).reshape(3),
        link_idx=None,
        transform=None,
        friction=0.42,
        elasticity=1,
        solver_params=None,
    )
    h = mesh.convex_hull(b)
    self.assertIsInstance(h, Convex)
    self.assertSequenceEqual(h.vert.shape, (8, 3))
    self.assertEqual(np.unique(np.abs(h.vert)), 0.5)
    np.testing.assert_array_equal(h.unique_edge, [[0, 1], [0, 2], [0, 4]])
    self.assertSequenceEqual(h.face.shape, (6, 4))  # one rectangle per face
    self.assertEqual(h.friction, 0.42)

    expected_face_norm = [
        [0, -1.0, 0],  # left
        [0, 0, -1.0],  # bottom
        [+1.0, 0, 0],  # front
        [0, +1.0, 0],  # right
        [0, 0, +1.0],  # top
        [-1.0, 0, 0],  # back
    ]
    face_norm = mesh.get_face_norm(h.vert, h.face)
    np.testing.assert_array_almost_equal(face_norm, expected_face_norm)


class ConvexTest(absltest.TestCase):

  def test_pyramid(self):
    """Tests a pyramid convex hull."""
    vert = np.array([
        [-0.025, 0.05, 0.05],
        [-0.025, -0.05, -0.05],
        [-0.025, -0.05, 0.05],
        [-0.025, 0.05, -0.05],
        [0.075, 0.0, 0.0],
    ])
    face = np.array(
        [[0, 1, 2], [0, 3, 1], [0, 4, 3], [0, 2, 4], [2, 1, 4], [1, 3, 4]]
    )
    pyramid = Mesh(
        link_idx=0,
        transform=Transform.zero((1,)),
        vert=vert,
        face=face,
        friction=1,
        elasticity=0,
        solver_params=None,
    )
    h = mesh.convex_hull(pyramid)

    self.assertIsInstance(h, Convex)

    # get index of h.vert for vertices in vert
    dist = np.repeat(vert, vert.shape[0], axis=0) - np.tile(
        h.vert, (vert.shape[0], 1)
    )
    dist = (dist**2).sum(axis=1).reshape((vert.shape[0], -1))
    vidx = np.argmin(dist, axis=0)

    # check verts
    np.testing.assert_array_equal(h.vert, vert[vidx])

    # check face vertices
    map_ = {v: k for k, v in enumerate(vidx)}
    h_face = np.vectorize(map_.get)(h.face)
    face_verts = sorted([tuple(sorted(set(s))) for s in h_face.tolist()])
    expected_face_verts = sorted([
        (0, 3, 4), (1, 3, 4), (0, 2, 4), (0, 1, 2, 3), (1, 2, 4)])
    self.assertSequenceEqual(
        face_verts,
        expected_face_verts,
    )

    # check edges
    unique_edge = np.vectorize(map_.get)(h.unique_edge)
    unique_edge = np.array(sorted(unique_edge.tolist()))
    np.testing.assert_array_equal(
        unique_edge,
        np.array([[0, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]]),
    )
    self.assertEqual(h.friction, 1)


class ConvexHull2DTest(absltest.TestCase):

  def test_convex_hull_2d_axis1(self):
    """Tests for the correct winding order for a polgyon with +y normal."""
    pts = np.array([
        [-0.04634297, -0.06652775, 0.05853534],
        [-0.01877651, -0.08309858, -0.05236476],
        [0.02362804, -0.08010745, 0.05499557],
        [0.04066505, -0.09034877, -0.01354446],
        [-0.07255043, -0.06837638, -0.00781699],
    ])
    normal = np.array([-0.18467607, -0.97768016, 0.10018111])
    idx = mesh.convex_hull_2d(pts, normal)
    expected = np.cross(pts[idx][1] - pts[idx][0], pts[idx][2] - pts[idx][0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_array_almost_equal(normal, expected)

  def test_convex_hull_2d_axis2(self):
    """Tests for the correct winding order for a polgyon with +z normal."""
    pts = np.array([
        [0.08607829, -0.03881998, -0.03291714],
        [-0.01877651, -0.08309858, -0.05236476],
        [0.05470364, 0.00027677, -0.08371042],
        [-0.01010019, -0.02708892, -0.0957297],
        [0.04066505, -0.09034877, -0.01354446],
    ])
    normal = np.array([0.3839915, -0.60171936, -0.70034587])
    idx = mesh.convex_hull_2d(pts, normal)
    expected = np.cross(pts[idx][1] - pts[idx][0], pts[idx][2] - pts[idx][0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_array_almost_equal(normal, expected)


class UniqueEdgesTest(absltest.TestCase):

  def test_tetrahedron_edges(self):
    """Tests unique edges for a tetrahedron."""
    vert = np.array(
        [[-0.1, 0.0, -0.1], [0.0, 0.1, 0.1], [0.1, 0.0, -0.1], [0.0, -0.1, 0.1]]
    )
    face = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])
    idx = mesh.get_unique_edges(vert, face)
    np.testing.assert_array_equal(
        idx, np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    )


if __name__ == '__main__':
  absltest.main()
