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
"""Useful functions for creating and processing meshes."""

import itertools

from brax.v2.base import Box, Mesh
from jax import numpy as jp


_BOX_CORNERS = list(itertools.product((-1, 1), (-1, 1), (-1, 1)))

# pyformat: disable
# The faces of a triangulated box, i.e. the indices in _BOX_CORNERS of the
# vertices of the 12 triangles (two triangles for each side of the box).
_TRIANGULATED_BOX_FACES = [
    0, 1, 4, 4, 1, 5,  # front
    0, 4, 2, 2, 4, 6,  # bottom
    6, 4, 5, 6, 5, 7,  # right
    2, 6, 3, 3, 6, 7,  # back
    1, 3, 5, 5, 3, 7,  # top
    0, 2, 1, 1, 2, 3,  # left
]
# pyformat: enable


def box(b: Box) -> Mesh:
  """Creates a mesh from a box geometry."""
  assert len(b.halfsize.shape) == 2 and b.halfsize.shape[-1] == 3, (
      'Box halfsize should have a batch dimension and have length 3, '
      f'got {b.halfsize.shape}'
  )
  n_boxes = b.halfsize.shape[0]
  box_corners = jp.array([_BOX_CORNERS] * n_boxes)
  vert = box_corners * b.halfsize.reshape(n_boxes, -1, 3)
  face = jp.array([_TRIANGULATED_BOX_FACES] * n_boxes).reshape(n_boxes, -1, 3)
  return Mesh(
      face=face,
      vert=vert,
      link_idx=b.link_idx,
      transform=b.transform,
      friction=b.friction,
      elasticity=b.elasticity,
  )


def get_face_norm(vert: jp.ndarray, face: jp.ndarray) -> jp.ndarray:
  """Calculates face normals given vertices and face indexes."""
  face_vert = jp.take(vert, face, axis=0)
  # use CCW winding order convention
  edge0 = face_vert[:, 0, :] - face_vert[:, 2, :]
  edge1 = face_vert[:, 0, :] - face_vert[:, 1, :]
  face_norm = jp.cross(edge0, edge1)
  face_norm = face_norm / jp.linalg.norm(face_norm, axis=1).reshape((-1, 1))
  return face_norm
