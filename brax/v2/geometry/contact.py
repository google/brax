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
"""Calculations for generating contacts."""

from typing import Any, Callable, List, Optional, Tuple, TypeVar

from brax.v2 import math
from brax.v2.base import (
    Box,
    Capsule,
    Contact,
    Convex,
    Geometry,
    Mesh,
    Plane,
    Sphere,
    System,
    Transform,
)
from brax.v2.geometry import math as geom_math
from brax.v2.geometry import mesh as geom_mesh
import jax
from jax import numpy as jp
from jax.tree_util import tree_map


Geom = TypeVar('Geom', bound=Geometry)


def _combine(
    geom_a: Geometry, geom_b: Geometry
) -> Tuple[float, float, Tuple[int, int]]:
  # default is to take maximum, but can override
  friction = jp.maximum(geom_a.friction, geom_b.friction)
  elasticity = jp.maximum(geom_a.elasticity, geom_b.elasticity)
  link_idx = (
      geom_a.link_idx,
      geom_b.link_idx if geom_b.link_idx is not None else -1,
  )
  return friction, elasticity, link_idx


def _sphere_plane(sphere: Sphere, plane: Plane) -> Contact:
  """Calculates one contact between a sphere and a plane."""
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
  t = jp.dot(sphere.transform.pos - plane.transform.pos, n)
  penetration = sphere.radius - t
  # halfway between contact points on sphere and on plane
  pos = sphere.transform.pos - n * (sphere.radius - 0.5 * penetration)
  c = Contact(pos, n, penetration, *_combine(sphere, plane))
  # add a batch dimension of size 1
  return tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_sphere(s_a: Sphere, s_b: Sphere) -> Contact:
  """Calculates one contact between two spheres."""
  n, dist = math.normalize(s_a.transform.pos - s_b.transform.pos)
  penetration = s_a.radius + s_b.radius - dist
  s_a_pos = s_a.transform.pos - n * s_a.radius
  s_b_pos = s_b.transform.pos + n * s_b.radius
  pos = (s_a_pos + s_b_pos) * 0.5
  c = Contact(pos, n, penetration, *_combine(s_a, s_b))
  # add a batch dimension of size 1
  return tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_capsule(sphere: Sphere, capsule: Capsule) -> Contact:
  """Calculates one contact between a sphere and a capsule."""
  segment = jp.array([0.0, 0.0, capsule.length * 0.5])
  segment = math.rotate(segment, capsule.transform.rot)
  pt = geom_math.closest_segment_point(
      capsule.transform.pos - segment,
      capsule.transform.pos + segment,
      sphere.transform.pos,
  )
  n, dist = math.normalize(sphere.transform.pos - pt)
  penetration = sphere.radius + capsule.radius - dist

  sphere_pos = sphere.transform.pos - n * sphere.radius
  cap_pos = pt + n * capsule.radius
  pos = (sphere_pos + cap_pos) * 0.5

  c = Contact(pos, n, penetration, *_combine(sphere, capsule))
  # add a batch dimension of size 1
  return tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_mesh(sphere: Sphere, mesh: Mesh) -> Contact:
  """Calculates contacts between a sphere and a mesh."""

  @jax.vmap
  def sphere_face(face):
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        face, mesh.transform.rot
    )
    p0, p1, p2 = pt[0, :], pt[1, :], pt[2, :]

    tri_p = geom_math.closest_triangle_point(p0, p1, p2, sphere.transform.pos)
    n = sphere.transform.pos - tri_p
    n, dist = math.normalize(n)
    penetration = sphere.radius - dist
    sph_p = sphere.transform.pos - n * sphere.radius
    pos = (tri_p + sph_p) * 0.5
    return Contact(pos, n, penetration, *_combine(sphere, mesh))

  return sphere_face(jp.take(mesh.vert, mesh.face, axis=0))


def _capsule_plane(capsule: Capsule, plane: Plane) -> Contact:
  """Calculates two contacts between a capsule and a plane."""
  segment = jp.array([0.0, 0.0, capsule.length * 0.5])
  segment = math.rotate(segment, capsule.transform.rot)

  results = []
  for off in [segment, -segment]:
    sphere = Sphere(
        link_idx=capsule.link_idx,
        transform=Transform.create(pos=capsule.transform.pos + off),
        friction=capsule.friction,
        elasticity=capsule.elasticity,
        radius=capsule.radius,
    )
    results.append(_sphere_plane(sphere, plane))

  return tree_map(lambda *x: jp.concatenate(x), *results)


def _capsule_capsule(cap_a: Capsule, cap_b: Capsule) -> Contact:
  """Calculates one contact between two capsules."""
  seg_a = jp.array([0.0, 0.0, cap_a.length * 0.5])
  seg_a = math.rotate(seg_a, cap_a.transform.rot)
  seg_b = jp.array([0.0, 0.0, cap_b.length * 0.5])
  seg_b = math.rotate(seg_b, cap_b.transform.rot)
  pt_a, pt_b = geom_math.closest_segment_to_segment_points(
      cap_a.transform.pos - seg_a,
      cap_a.transform.pos + seg_a,
      cap_b.transform.pos - seg_b,
      cap_b.transform.pos + seg_b,
  )
  n, dist = math.normalize(pt_a - pt_b)
  penetration = cap_a.radius + cap_b.radius - dist

  cap_a_pos = pt_a - n * cap_a.radius
  cap_b_pos = pt_b + n * cap_b.radius
  pos = (cap_a_pos + cap_b_pos) * 0.5

  c = Contact(pos, n, penetration, *_combine(cap_a, cap_b))
  # add a batch dimension of size 1
  return tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _capsule_mesh(capsule: Capsule, mesh: Mesh) -> Contact:
  """Calculates contacts between a capsule and a mesh."""

  @jax.vmap
  def capsule_face(face, face_norm):
    seg = jp.array([0.0, 0.0, capsule.length * 0.5])
    seg = math.rotate(seg, capsule.transform.rot)
    end_a, end_b = capsule.transform.pos - seg, capsule.transform.pos + seg

    tri_norm = math.rotate(face_norm, mesh.transform.rot)
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        face, mesh.transform.rot
    )
    p0, p1, p2 = pt[..., 0, :], pt[..., 1, :], pt[..., 2, :]

    seg_p, tri_p = geom_math.closest_segment_triangle_points(
        end_a, end_b, p0, p1, p2, tri_norm
    )
    n = seg_p - tri_p
    n, dist = math.normalize(n)
    penetration = capsule.radius - dist
    cap_p = seg_p - n * capsule.radius
    pos = (tri_p + cap_p) * 0.5
    return Contact(pos, n, penetration, *_combine(capsule, mesh))

  face_vert = jp.take(mesh.vert, mesh.face, axis=0)
  face_norm = geom_mesh.get_face_norm(mesh.vert, mesh.face)
  return capsule_face(face_vert, face_norm)


def _convex_convex(convex_a: Convex, convex_b: Convex) -> Contact:
  """Calculates contacts between two convex objects."""
  normals_a = geom_mesh.get_face_norm(convex_a.vert, convex_a.face)
  normals_b = geom_mesh.get_face_norm(convex_b.vert, convex_b.face)
  faces_a = jp.take(convex_a.vert, convex_a.face, axis=0)
  faces_b = jp.take(convex_b.vert, convex_b.face, axis=0)

  def transform_faces(convex, faces, normals):
    faces = convex.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        faces, convex.transform.rot
    )
    normals = math.rotate(normals, convex.transform.rot)
    return faces, normals

  v_transform_faces = jax.vmap(transform_faces, in_axes=[None, 0, 0])
  faces_a, normals_a = v_transform_faces(convex_a, faces_a, normals_a)
  faces_b, normals_b = v_transform_faces(convex_b, faces_b, normals_b)

  def transform_verts(convex, vertices):
    vertices = convex.transform.pos + math.rotate(
        vertices, convex.transform.rot
    )
    return vertices

  v_transform_verts = jax.vmap(transform_verts, in_axes=[None, 0])
  vertices_a = v_transform_verts(convex_a, convex_a.vert)
  vertices_b = v_transform_verts(convex_b, convex_b.vert)

  unique_edges_a = jp.take(vertices_a, convex_a.unique_edge, axis=0)
  unique_edges_b = jp.take(vertices_b, convex_b.unique_edge, axis=0)

  c = geom_math.sat_hull_hull(
      faces_a,
      faces_b,
      vertices_a,
      vertices_b,
      normals_a,
      normals_b,
      unique_edges_a,
      unique_edges_b,
  )
  friction, elasticity, link_idx = jax.tree_map(
      lambda x: jp.repeat(x, 4), _combine(convex_a, convex_b)
  )

  return Contact(
      c.pos,
      c.normal,
      c.penetration,
      friction,
      elasticity,
      link_idx,
  )


def _mesh_plane(mesh: Mesh, plane: Plane) -> Contact:
  """Calculates contacts between a mesh and a plane."""

  @jax.vmap
  def point_plane(vert):
    n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
    pos = mesh.transform.pos + math.rotate(vert, mesh.transform.rot)
    penetration = jp.dot(plane.transform.pos - pos, n)
    return Contact(pos, n, penetration, *_combine(mesh, plane))

  return point_plane(mesh.vert)


_TYPE_FUN = {
    (Sphere, Plane): jax.vmap(_sphere_plane),
    (Sphere, Sphere): jax.vmap(_sphere_sphere),
    (Sphere, Capsule): jax.vmap(_sphere_capsule),
    (Sphere, Box): jax.vmap(_sphere_mesh),
    (Sphere, Mesh): jax.vmap(_sphere_mesh),
    (Capsule, Plane): jax.vmap(_capsule_plane),
    (Capsule, Capsule): jax.vmap(_capsule_capsule),
    (Capsule, Box): jax.vmap(_capsule_mesh),
    (Capsule, Mesh): jax.vmap(_capsule_mesh),
    (Convex, Convex): jax.vmap(_convex_convex),
    (Mesh, Plane): jax.vmap(_mesh_plane),
}


def _geom_pairs(
    sys, x
) -> List[Tuple[Optional[Callable[[Geom, Geom], Any]], Geom, Geom]]:
  """Transforms geometries and gets contact functions to apply to each pair."""
  geom_pairs = []
  for geom_a, geom_b in sys.contacts:
    # check both type signatures: a, b <> b, a
    fun = _TYPE_FUN.get((type(geom_a), type(geom_b)))
    if fun is None:
      fun = _TYPE_FUN.get((type(geom_b), type(geom_a)))
      if fun is None:
        raise RuntimeError(
            f'unrecognized collider pair: {type(geom_a)}, {type(geom_b)}'
        )
      geom_a, geom_b = geom_b, geom_a

    tx_a = x.take(geom_a.link_idx).vmap().do(geom_a.transform)
    geom_a = geom_a.replace(transform=tx_a)
    # OK for geom b to have no parent links, e.g. static terrain
    if geom_b.link_idx is not None:
      tx_b = x.take(geom_b.link_idx).vmap().do(geom_b.transform)
      geom_b = geom_b.replace(transform=tx_b)

    geom_pairs.append((fun, geom_a, geom_b))  # type: ignore

  return geom_pairs


def contact(sys: System, x: Transform) -> Optional[Contact]:
  """Calculates contacts in the system.

  Args:
    sys: system defining the kinematic tree and other properties
    x: link transforms in world frame

  Returns:
    Contact pytree, one row for each element in sys.contacts

  Raises:
    RuntimeError: if sys.contacts has an invalid type pair
  """

  contacts = []

  for fun, geom_a, geom_b in _geom_pairs(sys, x):
    c = fun(geom_a, geom_b)  # type: ignore
    c = tree_map(jp.concatenate, c)
    contacts.append(c)

  if not contacts:
    return None

  return jax.tree_map(lambda *x: jp.concatenate(x), *contacts)
