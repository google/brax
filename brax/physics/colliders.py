# Copyright 2021 The Brax Authors.
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
"""Colliders push apart bodies that are in contact."""

import abc
import itertools
from typing import Any, Callable, List, Tuple
import warnings

from brax import jumpy as jp
from brax import math
from brax import pytree
from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics.base import P, QP, vec_to_arr


class Collidable:
  """Part of a body (with geometry and mass/inertia) that can collide.

  Collidables can repeat for a geometry. e.g. a body with a box collider has 8
  corner collidables.
  """

  def __init__(self, collidables: List[config_pb2.Body], body: bodies.Body):
    self.body = jp.take(body, [body.index[c.name] for c in collidables])
    self.pos = jp.array(
        [vec_to_arr(c.colliders[0].position) for c in collidables])
    # assuming first collider's material is body's material
    self.friction = jp.array([c.colliders[0].material.friction for c in collidables])

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

class Cull(abc.ABC):
  """Selects collidable pair candidates for collision detection."""

  @abc.abstractmethod
  def get(self) -> Tuple[Collidable, Collidable]:
    """Returns collidable pair candidates for collision detection."""

  def update(self, qp: QP):
    """Updates candidate statistics given qp world state."""


@pytree.register
class AllPairs(Cull):
  """Naive strategy: returns all possible pairs for collision detection."""

  def __init__(self, col_a: Collidable, col_b: Collidable,
               mask: List[Tuple[int, int]]):
    self.col_a = jp.take(col_a, [a for a, _ in mask])
    self.col_b = jp.take(col_b, [b for _, b in mask])

  def get(self) -> Tuple[Collidable, Collidable]:
    return self.col_a, self.col_b


@pytree.register
class NearNeighbors(Cull):
  """Returns top K nearest neighbor collidables for collision detection."""

  __pytree_ignore__ = ('cutoff',)

  def __init__(self, col_a: Collidable, col_b: Collidable,
               mask: List[Tuple[int, int]], cutoff: int):
    dist_off = jp.zeros(col_a.body.idx.shape + col_b.body.idx.shape)
    # TODO: consider accounting for bounds/radius of a collidable
    dist_mask = dist_off + float('inf')
    mask = (jp.array([a for a, _ in mask]), jp.array([b for _, b in mask]))
    dist_off += jp.index_update(dist_mask, mask, 0)
    self.dist_off = dist_off
    self.cutoff = cutoff
    self.candidate_a, self.candidate_b = col_a, col_b
    self.col_a, self.col_b = self.candidate_a, self.candidate_b

  def update(self, qp: QP):
    if self.candidate_a is self.candidate_b:
      pos = self.candidate_a.position(qp)
      dist = jp.vmap(lambda pt: jp.norm(pos - pt, axis=-1))(pos)
    else:
      pos_a = self.candidate_a.position(qp)
      pos_b = self.candidate_b.position(qp)
      dist = jp.vmap(lambda pt: jp.norm(pos_b - pt, axis=-1))(pos_a)
    # add in offset and convert to closeness
    sim = -(dist + self.dist_off)
    # TODO: return a status if more valid candidates than cutoff
    _, idx = jp.top_k(sim.ravel(), self.cutoff)
    self.col_a = jp.take(self.candidate_a, idx // sim.shape[-1])
    self.col_b = jp.take(self.candidate_b, idx % sim.shape[-1])

  def get(self) -> Tuple[Collidable, Collidable]:
    return self.col_a, self.col_b


class Collider(abc.ABC):
  """Calculates impulses given contacts from a contact function."""

  __pytree_ignore__ = ('contact_fn', 'cull', 'friction', 'elasticity',
                       'baumgarte_erp')

  def __init__(self, contact_fn: Callable[[Any, Any, QP, QP], Contact],
               cull: Cull, config: config_pb2.Config):
    """Creates a PairwiseCollider that exhaustively checks for contacts.

    Args:
      contact_fn: a function that returns contacts given collidables and qp
      cull: a cull strategy
      config: for specifying global system config
    """
    self.contact_fn = contact_fn
    self.cull = cull
    self.elasticity = config.elasticity
    self.baumgarte_erp = config.baumgarte_erp * config.substeps / config.dt

  @abc.abstractmethod
  def apply(self, qp: QP) -> P:
    """Returns impulse from any potential contacts between collidables.

    Args:
      qp: Coordinate/velocity frame of the bodies.

    Returns:
      dP: Impulse to apply to the bodies in the collision.
    """


@pytree.register
class OneWayCollider(Collider):
  """Calculates one-way impulses, where the second collidable is static."""

  def apply(self, qp: QP) -> P:
    """Returns impulse from any potential contacts between collidables."""
    col_a, col_b = self.cull.get()
    qp_a = jp.take(qp, col_a.body.idx)
    qp_b = jp.take(qp, col_b.body.idx)
    contact = jp.vmap(self.contact_fn)(col_a, col_b, qp_a, qp_b)
    dp = jp.vmap(self._contact)(qp_a, col_a, contact)

    contact = jp.where(jp.any(dp.vel, axis=-1), 1.0, 0.0)
    contact = jp.segment_sum(contact, col_a.body.idx, qp.pos.shape[0])
    dp_vel = jp.segment_sum(dp.vel, col_a.body.idx, qp.pos.shape[0])
    dp_ang = jp.segment_sum(dp.ang, col_a.body.idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    dp_vel = dp_vel / jp.reshape(1e-8 + contact, (dp_vel.shape[0], 1))
    dp_ang = dp_ang / jp.reshape(1e-8 + contact, (dp_ang.shape[0], 1))

    return P(vel=dp_vel, ang=dp_ang)

  def _contact(self, qp: QP, col: Collidable, contact: Contact) -> P:
    """Calculates impulse on a body due to a contact."""
    rel_pos = contact.pos - qp.pos
    baumgarte_vel = self.baumgarte_erp * contact.penetration
    normal_vel = jp.dot(contact.normal, contact.vel)
    temp1 = jp.matmul(col.body.inertia, jp.cross(rel_pos, contact.normal))
    ang = jp.dot(contact.normal, jp.cross(temp1, rel_pos))
    impulse = (-1. * (1. + self.elasticity) * normal_vel + baumgarte_vel) / (
        (1. / col.body.mass) + ang)
    dp_n = col.body.impulse(qp, impulse * contact.normal, contact.pos)

    # apply drag due to friction acting parallel to the surface contact
    vel_d = contact.vel - normal_vel * contact.normal
    impulse_d = jp.safe_norm(vel_d) / ((1. / (col.body.mass)) + ang)
    # drag magnitude cannot exceed max friction
    impulse_d = jp.minimum(impulse_d, col.friction * impulse)
    dir_d = vel_d / (1e-6 + jp.safe_norm(vel_d))
    dp_d = col.body.impulse(qp, -impulse_d * dir_d, contact.pos)
    # apply collision if penetrating, approaching, and oriented correctly
    apply_n = jp.where(
        (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)

    return dp_n * apply_n + dp_d * apply_d


@pytree.register
class TwoWayCollider(Collider):
  """Calculates two-way impulses on collidable pairs."""

  def apply(self, qp: QP) -> P:
    """Returns impulse from any potential contacts between collidables."""
    col_a, col_b = self.cull.get()
    qp_a = jp.take(qp, col_a.body.idx)
    qp_b = jp.take(qp, col_b.body.idx)
    contact = jp.vmap(self.contact_fn)(col_a, col_b, qp_a, qp_b)
    dp_a, dp_b = jp.vmap(self._contact)(col_a, col_b, qp_a, qp_b, contact)

    body_idx = jp.concatenate((col_a.body.idx, col_b.body.idx))
    dp_vel = jp.concatenate((dp_a.vel, dp_b.vel))
    dp_ang = jp.concatenate((dp_a.ang, dp_b.ang))
    contact = jp.where(jp.any(dp_vel, axis=-1), 1.0, 0.0)
    contact = jp.segment_sum(contact, body_idx, qp.pos.shape[0])
    dp_vel = jp.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jp.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    contact = jp.reshape(1e-8 + contact, (dp_vel.shape[0], 1))
    dp_vel = dp_vel / contact
    dp_ang = dp_ang / contact
    return P(vel=dp_vel, ang=dp_ang)

  def _contact(self, col_a: Collidable, col_b: Collidable, qp_a: QP, qp_b: QP,
               contact: Contact) -> Tuple[P, P]:
    """Calculates impulse on a body due to a contact."""
    rel_pos_a = contact.pos - qp_a.pos
    rel_pos_b = contact.pos - qp_b.pos
    baumgarte_vel = self.baumgarte_erp * contact.penetration
    normal_vel = jp.dot(contact.normal, contact.vel)
    temp1 = jp.matmul(col_a.body.inertia, jp.cross(rel_pos_a, contact.normal))
    temp2 = jp.matmul(col_b.body.inertia, jp.cross(rel_pos_b, contact.normal))
    ang = jp.dot(contact.normal,
                 jp.cross(temp1, rel_pos_a) + jp.cross(temp2, rel_pos_b))
    impulse = (-1. * (1. + self.elasticity) * normal_vel + baumgarte_vel) / (
        (1. / col_a.body.mass) + (1. / col_b.body.mass) + ang)
    dp_n_a = col_a.body.impulse(qp_a, -impulse * contact.normal, contact.pos)
    dp_n_b = col_b.body.impulse(qp_b, impulse * contact.normal, contact.pos)

    # apply drag due to friction acting parallel to the surface contact
    vel_d = contact.vel - normal_vel * contact.normal
    impulse_d = jp.safe_norm(vel_d) / ((1. / col_a.body.mass) +
                                       (1. / col_b.body.mass) + ang)
    # select friction coefficients (combine by multiplication rule by default)
    friction = col_a.friction * col_b.friction
    # drag magnitude cannot exceed max friction
    impulse_d = jp.minimum(impulse_d, friction * impulse)
    dir_d = vel_d / (1e-6 + jp.safe_norm(vel_d))
    dp_d_a = col_a.body.impulse(qp_a, impulse_d * dir_d, contact.pos)
    dp_d_b = col_a.body.impulse(qp_b, -impulse_d * dir_d, contact.pos)
    # apply collision normal if penetrating, approaching, and oriented correctly
    apply_n = jp.where(
        (contact.penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * jp.where(jp.safe_norm(vel_d) > 0.01, 1., 0.)

    dp_a = dp_n_a * apply_n + dp_d_a * apply_d
    dp_b = dp_n_b * apply_n + dp_d_b * apply_d
    return dp_a, dp_b


@pytree.register
class BoxCorner(Collidable):
  """A box corner."""

  def __init__(self, boxes: List[config_pb2.Body], body: bodies.Body):
    super().__init__([boxes[i // 8] for i in range(len(boxes) * 8)], body)
    coords = jp.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))))
    corners = []
    for b in boxes:
      col = b.colliders[0]
      rot = math.euler_to_quat(vec_to_arr(col.rotation))
      box = coords * vec_to_arr(col.box.halfsize)
      box = jp.vmap(math.rotate, include=(True, False))(box, rot)
      box = box + vec_to_arr(col.position)
      corners.extend(box)
    self.corner = jp.array(corners)


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


def box_plane(box: BoxCorner, _: Plane, qp_a: QP, qp_b: QP) -> Contact:
  """Returns contact between a box corner and a plane."""
  pos, vel = qp_a.to_world(box.corner)
  normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
  penetration = jp.dot(qp_b.pos - pos, normal)
  return Contact(pos, vel, normal, penetration)


def box_heightmap(box: BoxCorner, hm: HeightMap, qp_a: QP, qp_b: QP) -> Contact:
  """Returns contact between a box corner and a height map."""
  # Note that this only checks box corners against height map surfaces, and is
  # missing box planes against height map points.
  pos, vel = qp_a.to_world(box.corner)
  pos = math.inv_rotate(pos - qp_b.pos, qp_b.rot)
  uv_pos = pos[:2] / hm.cell_size
  # find the square in the mesh that enclose the candidate point, with mesh
  # indices ux_idx, ux_udx_u, uv_idx_v, uv_idx_uv.
  uv_idx = jp.floor(uv_pos).astype(int)
  uv_idx_u = uv_idx + jp.array([1, 0])
  uv_idx_v = uv_idx + jp.array([0, 1])
  uv_idx_uv = uv_idx + jp.array([1, 1])
  # find the orientation of the triangle of this square that encloses the
  # candidate point
  delta_uv = uv_pos - uv_idx
  # whether the corner lies on the first or secound triangle:
  mu = jp.where(delta_uv[0] + delta_uv[1] < 1, 1, -1)
  # compute the mesh indices of the vertices of this triangle
  p0 = jp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx, uv_idx_uv)
  p1 = jp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_u, uv_idx_v)
  p2 = jp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_v, uv_idx_u)
  h0 = hm.height[p0[0], p0[1]]
  h1 = hm.height[p1[0], p1[1]]
  h2 = hm.height[p2[0], p2[1]]

  raw_normal = jp.array(
      [-mu * (h1 - h0), -mu * (h2 - h0), hm.cell_size])
  normal = raw_normal / jp.norm(raw_normal)
  normal = math.rotate(normal, qp_b.rot)
  height = jp.array(
      [p0[0] * hm.cell_size, p0[1] * hm.cell_size, h0])
  penetration = jp.dot(height - pos, normal)
  return Contact(pos, vel, normal, penetration)


def capsule_plane(cap: CapsuleEnd, _: Plane, qp_a: QP, qp_b: QP) -> Contact:
  """Returns contact between a capsule and a plane."""
  cap_end_world = qp_a.pos + math.rotate(cap.end, qp_a.rot)
  normal = math.rotate(jp.array([0., 0., 1.]), qp_b.rot)
  pos = cap_end_world - normal * cap.radius
  vel = qp_a.vel + jp.cross(qp_a.ang, pos - qp_a.pos)
  penetration = jp.dot(qp_b.pos - pos, normal)
  return Contact(pos, vel, normal, penetration)


def capsule_capsule(cap_a: Capsule, cap_b: Capsule, qp_a: QP,
                    qp_b: QP) -> Contact:
  """Returns contact between two capsules."""
  def endpoints(end, qp, offset):
    pos = qp.pos + math.rotate(offset, qp.rot)
    end = math.rotate(end, qp.rot)
    return pos + end, pos - end

  def closest_segment_point(a, b, pt):
    ab = b - a
    t = jp.dot(pt - a, ab) / (jp.dot(ab, ab) + 1e-10)
    return a + jp.clip(t, 0., 1.) * ab

  a0, a1 = endpoints(cap_a.end, qp_a, cap_a.pos)
  b0, b1 = endpoints(cap_b.end, qp_b, cap_b.pos)
  v0, v1, v2, v3 = b0 - a0, b1 - a0, b0 - a1, b1 - a1
  d0, d1 = jp.dot(v0, v0), jp.dot(v1, v1)
  d2, d3 = jp.dot(v2, v2), jp.dot(v3, v3)
  a_best = jp.where((d2 < d0) | (d2 < d1) | (d3 < d0) | (d3 < d1), a1, a0)
  b_best = closest_segment_point(b0, b1, a_best)
  a_best = closest_segment_point(a0, a1, b_best)

  penetration_vec = b_best - a_best
  dist = jp.safe_norm(penetration_vec)
  normal = penetration_vec / (1e-6 + dist)
  penetration = cap_a.radius + cap_b.radius - dist
  pos = (a_best + b_best) / 2
  vel = qp_b.world_velocity(pos) - qp_a.world_velocity(pos)
  return Contact(pos, vel, normal, penetration)


def get(config: config_pb2.Config, body: bodies.Body) -> List[Collider]:
  """Creates all colliders given a config."""
  def key_fn(x, y):
    return tuple(sorted((body.index.get(x, -1), body.index.get(y, -1))))
  include = {key_fn(f.first, f.second) for f in config.collide_include}
  # exclude colliders for joint parents and children, unless explicitly included
  ignore = {key_fn(j.parent, j.child) for j in config.joints} - include
  # exclude colliders where both bodies are frozen
  frozen = [b.name for b in config.bodies if b.frozen.all]
  ignore.union({key_fn(x, y) for x, y in itertools.combinations(frozen, 2)})

  # flatten and emit one collider per body
  flat_bodies = []
  for b in config.bodies:
    for collider in b.colliders:
      # we treat spheres as sphere-shaped capsules with a single end
      if collider.WhichOneof('type') == 'sphere':
        radius = collider.sphere.radius
        collider = config_pb2.Collider()
        collider.capsule.radius = radius
        collider.capsule.length = 2 * radius
        collider.capsule.end = 1

      new_body = config_pb2.Body()
      new_body.CopyFrom(b)
      new_body.ClearField('colliders')
      new_body.colliders.append(collider)
      flat_bodies.append(new_body)

  # group by type
  type_colliders = {}
  for b in flat_bodies:
    key = b.colliders[0].WhichOneof('type')
    if key not in type_colliders:
      type_colliders[key] = []
    type_colliders[key].append(b)

  # add colliders
  supported_types = {
      ('box', 'plane'): (BoxCorner, Plane, box_plane),
      ('box', 'heightMap'): (BoxCorner, HeightMap, box_heightmap),
      ('capsule', 'plane'): (CapsuleEnd, Plane, capsule_plane),
      ('capsule', 'capsule'): (Capsule, Capsule, capsule_capsule),
  }
  supported_near_neighbors = {('capsule', 'capsule')}
  collidable_cache = {}

  ret = []
  for type_a, type_b in itertools.combinations_with_replacement(
      type_colliders, 2):
    type_a, type_b = sorted((type_a, type_b))

    # calculate all possible body pairs that can collide via these two types
    cols_a, cols_b = type_colliders[type_a], type_colliders[type_b]
    if type_a == type_b:
      candidates = itertools.combinations(cols_a, 2)
    else:
      candidates = itertools.product(cols_a, cols_b)
    pairs = [(body.index[a.name], body.index[b.name]) for a, b in candidates]
    if include:
      pairs = [x for x in pairs if tuple(sorted(x)) in include]
    pairs = [x for x in pairs if tuple(sorted(x)) not in ignore]

    if not pairs:
      continue

    if (type_a, type_b) not in supported_types:
      warnings.warn(f'unsupported collider pair: {type_a}, {type_b}')
      continue

    # create our collidables
    col_cls_a, col_cls_b, contact_fn = supported_types[(type_a, type_b)]
    if col_cls_a not in collidable_cache:
      collidable_cache[col_cls_a] = col_cls_a(cols_a, body)
    if col_cls_b not in collidable_cache:
      collidable_cache[col_cls_b] = col_cls_b(cols_b, body)
    col_a = collidable_cache[col_cls_a]
    col_b = collidable_cache[col_cls_b]

    # convert pairs from body idx to collidable idx
    body_to_collidable_a, body_to_collidable_b = {}, {}
    for i, body_idx in enumerate(col_a.body.idx):
      body_to_collidable_a.setdefault(body_idx, []).append(i)
    for i, body_idx in enumerate(col_b.body.idx):
      body_to_collidable_b.setdefault(body_idx, []).append(i)
    mask = []
    for body_idx_a, body_idx_b in pairs:
      ias = body_to_collidable_a[body_idx_a]
      ibs = body_to_collidable_b[body_idx_b]
      for ia, ib in itertools.product(ias, ibs):
        mask.append((ia, ib))

    if config.collider_cutoff and len(pairs) > config.collider_cutoff and (
        type_a, type_b) in supported_near_neighbors:
      cull = NearNeighbors(col_a, col_b, mask, config.collider_cutoff)
    else:
      cull = AllPairs(col_a, col_b, mask)
    if all(b.frozen.all for b in cols_b):
      collider = OneWayCollider(contact_fn, cull, config)
    else:
      collider = TwoWayCollider(contact_fn, cull, config)

    ret.append(collider)

  return ret
