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
"""Calculations for generating contacts."""

from typing import Iterator, Optional, Tuple

from brax import math
from brax.base import (
    Capsule,
    Cylinder,
    Contact,
    Convex,
    Geometry,
    Mesh,
    Plane,
    Sphere,
    System,
    Transform,
)
from brax.geometry import math as geom_math
from brax.geometry import mesh as geom_mesh
import jax
from jax import numpy as jp


def _sphere_plane(sphere: Sphere, plane: Plane) -> Contact:
  """Calculates one contact between a sphere and a plane."""
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
  t = jp.dot(sphere.transform.pos - plane.transform.pos, n)
  penetration = sphere.radius - t
  # halfway between contact points on sphere and on plane
  pos = sphere.transform.pos - n * (sphere.radius - 0.5 * penetration)
  c = Contact(pos, n, penetration, *_combine(sphere, plane))
  # returns 1 contact, so add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_sphere(s_a: Sphere, s_b: Sphere) -> Contact:
  """Calculates one contact between two spheres."""
  n, dist = math.normalize(s_a.transform.pos - s_b.transform.pos)
  penetration = s_a.radius + s_b.radius - dist
  s_a_pos = s_a.transform.pos - n * s_a.radius
  s_b_pos = s_b.transform.pos + n * s_b.radius
  pos = (s_a_pos + s_b_pos) * 0.5
  c = Contact(pos, n, penetration, *_combine(s_a, s_b))
  # returns 1 contact, so add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


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
  # returns 1 contact, so add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_circle(sphere: Sphere, circle: Cylinder) -> Contact:
  """Calculates one contact between a sphere and a circle."""
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), circle.transform.rot)

  # orient the normal s.t. it points at the CoM of the sphere
  normal_dir = jp.sign(
      (sphere.transform.pos - circle.transform.pos).dot(n))
  n = n * normal_dir

  pos = sphere.transform.pos - n * sphere.radius
  plane_pt = circle.transform.pos
  penetration = jp.dot(plane_pt - pos, n)

  # check if the sphere radius is within the cylinder in the normal dir of the
  # circle
  plane_pt2 = plane_pt + n
  line_pt = geom_math.closest_line_point(
      plane_pt, plane_pt2, sphere.transform.pos
  )
  in_cylinder = (sphere.transform.pos - line_pt).dot(
      sphere.transform.pos - line_pt
  ) <= circle.radius**2

  # get closest point on circle edge
  perp_dir = jp.cross(n, sphere.transform.pos - plane_pt)
  perp_dir = math.rotate(perp_dir, math.quat_rot_axis(n, -jp.pi / 2.0))  # pytype: disable=wrong-arg-types  # jnp-type
  perp_dir, _ = math.normalize(perp_dir)
  edge_pt = plane_pt + perp_dir * circle.radius
  edge_contact = (sphere.transform.pos - edge_pt).dot(
      sphere.transform.pos - edge_pt
  ) <= sphere.radius**2
  edge_to_sphere = sphere.transform.pos - edge_pt
  edge_to_sphere = math.normalize(edge_to_sphere)[0]

  penetration = jp.where(in_cylinder, penetration, -jp.ones_like(penetration))
  penetration = jp.where(
      edge_contact,
      sphere.radius
      - jp.sqrt(
          (sphere.transform.pos - edge_pt).dot(sphere.transform.pos - edge_pt)
      ),
      penetration,
  )
  n = jp.where(edge_contact, edge_to_sphere, n)
  pos = jp.where(edge_contact, edge_pt, pos)
  c = Contact(pos, n, penetration, *_combine(sphere, circle))  # pytype: disable=wrong-arg-types  # jax-ndarray
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_convex(sphere: Sphere, convex: Convex) -> Contact:
  """Calculates contacts between a sphere and a convex object."""
  normals = geom_mesh.get_face_norm(convex.vert, convex.face)
  faces = jp.take(convex.vert, convex.face, axis=0)

  # Transform the sphere into the convex frame.
  sphere_transform = sphere.transform.to_local(convex.transform)

  # Get support from face normals.
  @jax.vmap
  def get_support(faces, normal):
    sphere_pos = sphere_transform.pos - normal * sphere.radius
    return jp.dot(sphere_pos - faces[0], normal)

  support = get_support(faces, normals)

  # Pick the face with minimal penetration as long as it has support.
  support = jp.where(support >= 0, -1e12, support)
  best_idx = support.argmax()
  face = faces[best_idx]
  normal = normals[best_idx]

  # Get closest point between the polygon face and the sphere center point.
  # Project the sphere center point onto poly plane. If it's inside polygon
  # edge normals, then we're done.
  pt = geom_math.project_pt_onto_plane(sphere_transform.pos, face[0], normal)
  edge_p0 = jp.roll(face, 1, axis=0)
  edge_p1 = face
  edge_normals = jax.vmap(jp.cross, in_axes=(0, None))(
      edge_p1 - edge_p0,
      normal,
  )
  edge_dist = jax.vmap(
      lambda plane_pt, plane_norm: (pt - plane_pt).dot(plane_norm)
  )(edge_p0, edge_normals)
  inside = jp.all(edge_dist <= 0)  # lte to handle degenerate edges

  # If the point is outside edge normals, project onto the closest edge plane
  # that the point is in front of.
  degenerate_edge = jp.all(edge_normals == 0, axis=1)
  behind = edge_dist < 0.0
  edge_dist = jp.where(degenerate_edge | behind, 1e12, edge_dist)
  idx = edge_dist.argmin()
  edge_pt = geom_math.closest_segment_point(edge_p0[idx], edge_p1[idx], pt)

  pt = jp.where(inside, pt, edge_pt)

  # Get the normal, penetration, and contact position.
  n, d = math.normalize(sphere_transform.pos - pt)
  spt = sphere_transform.pos - n * sphere.radius
  penetration = sphere.radius - d
  pos = (pt + spt) * 0.5

  # Go back to world frame.
  n = math.rotate(n, convex.transform.rot)
  pos = convex.transform.pos + math.rotate(pos, convex.transform.rot)

  c = Contact(pos, n, penetration, *_combine(sphere, convex))
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_mesh(sphere: Sphere, mesh: Mesh) -> Contact:
  """Calculates contacts between a sphere and a mesh."""

  @jax.vmap
  def sphere_face(face):
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=(0, None))(
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
        radius=capsule.radius,
        link_idx=capsule.link_idx,
        transform=Transform.create(pos=capsule.transform.pos + off),
        friction=capsule.friction,
        elasticity=capsule.elasticity,
        solver_params=capsule.solver_params,
    )
    results.append(_sphere_plane(sphere, plane))

  return jax.tree_map(lambda *x: jp.concatenate(x), *results)


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
  # returns 1 contact, so add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _capsule_convex(capsule: Capsule, convex: Convex) -> Contact:
  """Calculates contacts between a capsule and a convex object."""
  normals = geom_mesh.get_face_norm(convex.vert, convex.face)
  faces = jp.take(convex.vert, convex.face, axis=0)

  # Transform the capsule into the convex frame.
  cap_transform = capsule.transform.to_local(convex.transform)

  seg = jp.array([0.0, 0.0, capsule.length * 0.5])
  seg = math.rotate(seg, cap_transform.rot)
  cap_pts = jp.array([
      cap_transform.pos - seg,
      cap_transform.pos + seg,
  ])

  # Get support from face normals.
  @jax.vmap
  def get_support(face, normal):
    pts = cap_pts - normal * capsule.radius
    sup = jax.vmap(lambda x: jp.dot(x - face[0], normal))(pts)
    return sup.min()

  support = get_support(faces, normals)
  has_support = jp.all(support < 0)

  # Pick the face with minimal penetration as long as it has support.
  support = jp.where(support >= 0, -1e12, support)
  best_idx = support.argmax()
  face = faces[best_idx]
  normal = normals[best_idx]

  # Clip the edge against side planes and create two contact points against the
  # face.
  edge_p0 = jp.roll(face, 1, axis=0)
  edge_p1 = face
  edge_normals = jax.vmap(jp.cross, in_axes=(0, None))(
      edge_p1 - edge_p0,
      normal,
  )
  cap_pts_clipped, mask = geom_math.clip_edge_to_planes(
      cap_pts[0], cap_pts[1], edge_p0, edge_normals
  )
  cap_pts_clipped = cap_pts_clipped - normal * capsule.radius
  face_pts = jax.vmap(geom_math.project_pt_onto_plane, in_axes=(0, None, None))(
      cap_pts_clipped, face[0], normal
  )
  # Create variables for the face contact.
  pos = (cap_pts_clipped + face_pts) * 0.5
  norm = jp.stack([normal] * 2, 0)
  penetration = jp.where(
      mask & has_support, jp.dot(face_pts - cap_pts_clipped, normal), -1
  )

  # Get a potential edge contact.
  # TODO handle deep edge penetration more gracefully, since edge_axis
  # can point in the wrong direction for deep penetration.
  edge_closest, cap_closest = jax.vmap(
      geom_math.closest_segment_to_segment_points, in_axes=(0, 0, None, None)
  )(edge_p0, edge_p1, cap_pts[0], cap_pts[1])
  e_idx = ((edge_closest - cap_closest) ** 2).sum(axis=1).argmin()
  cap_closest_pt, edge_closest_pt = cap_closest[e_idx], edge_closest[e_idx]
  edge_axis = cap_closest_pt - edge_closest_pt
  edge_axis, edge_dist = math.normalize(edge_axis)
  edge_pos = (
      edge_closest_pt + (cap_closest_pt - edge_axis * capsule.radius)
  ) * 0.5
  edge_norm = edge_axis
  edge_penetration = capsule.radius - edge_dist
  has_edge_contact = edge_penetration > 0

  # Create the contact.
  pos = jp.where(has_edge_contact, pos.at[0].set(edge_pos), pos)
  norm = jp.where(has_edge_contact, norm.at[0].set(edge_norm), norm)

  # Go back to world frame.
  pos = convex.transform.pos + jax.vmap(math.rotate, in_axes=(0, None))(
      pos, convex.transform.rot
  )
  norm = jax.vmap(math.rotate, in_axes=(0, None))(norm, convex.transform.rot)

  penetration = jp.where(
      has_edge_contact, penetration.at[0].set(edge_penetration), penetration
  )
  tile_fn = lambda x: jp.tile(x, (2,) + tuple([1 for _ in x.shape]))
  params = jax.tree_map(tile_fn, _combine(capsule, convex))
  return Contact(pos, norm, penetration, *params)


def _capsule_mesh(capsule: Capsule, mesh: Mesh) -> Contact:
  """Calculates contacts between a capsule and a mesh."""

  @jax.vmap
  def capsule_face(face, face_norm):
    seg = jp.array([0.0, 0.0, capsule.length * 0.5])
    seg = math.rotate(seg, capsule.transform.rot)
    end_a, end_b = capsule.transform.pos - seg, capsule.transform.pos + seg

    tri_norm = math.rotate(face_norm, mesh.transform.rot)
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=(0, None))(
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


def _convex_plane(convex: Convex, plane: Plane) -> Contact:
  """Calculates contacts between a convex object and a plane."""
  # Transform the plane into convex frame.
  plane_transform = plane.transform.to_local(convex.transform)

  vertices = convex.vert
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane_transform.rot)
  support = jax.vmap(jp.dot, in_axes=(None, 0))(
      n, plane_transform.pos - vertices
  )
  idx = geom_math.manifold_points(vertices, support > 0, n)
  pos = vertices[idx]

  # Go back to world frame.
  pos = convex.transform.pos + jax.vmap(math.rotate, in_axes=(0, None))(
      pos, convex.transform.rot
  )
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)

  normal = jp.stack([n] * 4, axis=0)
  unique = jp.tril(idx == idx[:, None]).sum(axis=1) == 1
  penetration = jp.where(unique, support[idx], -1)
  tile_fn = lambda x: jp.tile(x, (4,) + tuple([1 for _ in x.shape]))
  params = jax.tree_map(tile_fn, _combine(convex, plane))
  return Contact(pos, normal, penetration, *params)


def _convex_convex(convex_a: Convex, convex_b: Convex) -> Contact:
  """Calculates contacts between two convex objects."""
  # pad face vertices so that we can broadcast between geom_i and geom_j
  sa, sb = convex_a.face.shape[-1], convex_b.face.shape[-1]
  if sa < sb:
    face = jp.pad(convex_a.face, ((0, sb - sa)), 'edge')
    convex_a = convex_a.replace(face=face)
  elif sb < sa:
    face = jp.pad(convex_b.face, ((0, sa - sb)), 'edge')
    convex_b = convex_b.replace(face=face)

  # ensure that the first object has less verts
  if convex_a.vert.shape[0] > convex_b.vert.shape[0]:
    convex_a, convex_b = convex_b, convex_a

  normals_a = geom_mesh.get_face_norm(convex_a.vert, convex_a.face)
  normals_b = geom_mesh.get_face_norm(convex_b.vert, convex_b.face)
  faces_a = jp.take(convex_a.vert, convex_a.face, axis=0)
  faces_b = jp.take(convex_b.vert, convex_b.face, axis=0)

  to_b_frame = convex_a.transform.to_local(convex_b.transform)

  def transform_faces(faces, normals):
    faces = to_b_frame.pos + jax.vmap(math.rotate, in_axes=(0, None))(
        faces, to_b_frame.rot
    )
    normals = math.rotate(normals, to_b_frame.rot)
    return faces, normals

  faces_a, normals_a = jax.vmap(transform_faces)(faces_a, normals_a)

  def transform_verts(vertices):
    return to_b_frame.pos + math.rotate(vertices, to_b_frame.rot)

  vertices_a = jax.vmap(transform_verts)(convex_a.vert)
  vertices_b = convex_b.vert

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

  # Go back to world frame.
  pos = convex_b.transform.pos + jax.vmap(math.rotate, in_axes=(0, None))(
      c.pos, convex_b.transform.rot
  )
  normal = jax.vmap(math.rotate, in_axes=(0, None))(
      c.normal, convex_b.transform.rot
  )
  penetration = c.penetration

  tile_fn = lambda x: jp.tile(x, (4,) + tuple([1 for _ in x.shape]))
  params = jax.tree_map(tile_fn, _combine(convex_a, convex_b))

  return Contact(pos, normal, penetration, *params)


def _mesh_plane(mesh: Mesh, plane: Plane) -> Contact:
  """Calculates contacts between a mesh and a plane."""

  @jax.vmap
  def point_plane(vert):
    n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
    pos = mesh.transform.pos + math.rotate(vert, mesh.transform.rot)
    penetration = jp.dot(plane.transform.pos - pos, n)
    return Contact(pos, n, penetration, *_combine(mesh, plane))

  return point_plane(mesh.vert)


def _combine(
    geom_a: Geometry, geom_b: Geometry
) -> Tuple[jax.Array, jax.Array, jax.Array, Tuple[jax.Array, jax.Array]]:
  # default is to take maximum, but can override
  friction = jp.maximum(geom_a.friction, geom_b.friction)
  elasticity = (geom_a.elasticity + geom_b.elasticity) * 0.5
  solver_params = (geom_a.solver_params + geom_b.solver_params) * 0.5
  link_idx = (
      jp.array(geom_a.link_idx if geom_a.link_idx is not None else -1),
      jp.array(geom_b.link_idx if geom_b.link_idx is not None else -1),
  )
  return friction, elasticity, solver_params, link_idx


_TYPE_FUN = {
    (Sphere, Plane): _sphere_plane,
    (Sphere, Sphere): _sphere_sphere,
    (Sphere, Capsule): _sphere_capsule,
    (Sphere, Cylinder): _sphere_circle,
    (Sphere, Convex): _sphere_convex,
    (Sphere, Mesh): _sphere_mesh,
    (Capsule, Plane): _capsule_plane,
    (Capsule, Capsule): _capsule_capsule,
    (Capsule, Convex): _capsule_convex,
    (Capsule, Mesh): _capsule_mesh,
    (Convex, Convex): _convex_convex,
    (Convex, Plane): _convex_plane,
    (Mesh, Plane): _mesh_plane,
}


def _geom_pairs(sys: System) -> Iterator[Tuple[Geometry, Geometry]]:
  """Returns geometry pairs that undergo collision."""
  for i in range(len(sys.geoms)):
    for j in range(i, len(sys.geoms)):
      mask_i, mask_j = sys.geom_masks[i], sys.geom_masks[j]
      if (mask_i & mask_j >> 32) | (mask_i >> 32 & mask_j) == 0:
        continue
      geom_i, geom_j = sys.geoms[i], sys.geoms[j]
      if geom_i.link_idx is None and geom_j.link_idx is None:
        # skip collisions between world geoms
        continue
      if geom_i.link_idx is None:
        # geom_j is expected to be the world geom
        geom_i, geom_j = geom_j, geom_i
      if i == j and geom_j.transform.pos.shape[0] == 1:
        # no self-collisions, unless the geom batch dimension is g.t. 1 where
        # self-collisions are more easily filtered out with contype/conaffinity.
        continue
      yield (geom_i, geom_j)


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
  for geom_i, geom_j in _geom_pairs(sys):
    key = (type(geom_i), type(geom_j))
    fun = _TYPE_FUN.get(key)
    if fun is None:
      geom_i, geom_j = geom_j, geom_i
      fun = _TYPE_FUN.get((key[1], key[0]))
      if fun is None:
        raise RuntimeError(f'unrecognized collider pair: {key}')
    tx_i = x.take(geom_i.link_idx).vmap().do(geom_i.transform)

    if geom_i is geom_j:
      geom_i = geom_i.replace(transform=tx_i)
      choose_i, choose_j = jp.triu_indices(geom_i.link_idx.shape[0], 1)
      geom_i, geom_j = geom_i.take(choose_i), geom_i.take(choose_j)
      c = jax.vmap(fun)(geom_i, geom_j)  # type: ignore
      c = jax.tree_map(jp.concatenate, c)
    else:
      geom_i = geom_i.replace(transform=tx_i)
      # OK for geom j to have no parent links, e.g. static terrain
      if geom_j.link_idx is not None:
        tx_j = x.take(geom_j.link_idx).vmap().do(geom_j.transform)
        geom_j = geom_j.replace(transform=tx_j)
      vvfun = jax.vmap(jax.vmap(fun, in_axes=(0, None)), in_axes=(None, 0))
      c = vvfun(geom_i, geom_j)  # type: ignore
      c = jax.tree_map(lambda x: jp.concatenate(jp.concatenate(x)), c)

    contacts.append(c)

  if not contacts:
    return None

  # ignore penetration of two geoms within the same link
  c = jax.tree_map(lambda *x: jp.concatenate(x), *contacts)
  penetration = jp.where(c.link_idx[0] != c.link_idx[1], c.penetration, -1)
  c = c.replace(penetration=penetration)

  return c
