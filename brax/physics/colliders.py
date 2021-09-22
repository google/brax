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
"""Colliders push bodies apart if they are penetrating."""

import abc
import itertools
from typing import List, Optional, Tuple
import warnings

from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import math
from brax.physics import pytree
from brax.physics.base import P, QP, euler_to_quat, take, vec_to_np
import jax
import jax.numpy as jnp


class PairwiseCollider(abc.ABC):
  """Exhaustively checks for contacts between pairs of colliders."""

  __pytree_ignore__ = ('friction', 'elasticity', 'baumgarte_erp')

  def __init__(self, colliders: List[Tuple[config_pb2.Body, config_pb2.Body]],
               body: bodies.Body, config: config_pb2.Config):
    """Creates a PairwiseCollider that exhaustively checks for contacts.

    Args:
      colliders: list of collider pairs
      body: batched body that contain the bodies to collide
      config: brax Config for specifying global system parameters
    """
    self.friction = config.friction
    self.elasticity = config.elasticity
    self.baumgarte_erp = config.baumgarte_erp * config.substeps / config.dt
    body_indices_a = [body.index[b.name] for b, _ in colliders]
    body_indices_b = [body.index[b.name] for _, b in colliders]
    self.body_a = take(body, jnp.array(body_indices_a))
    self.body_b = take(body, jnp.array(body_indices_b))

  def apply(self, qp: QP) -> P:
    """Returns impulse from a collision between pairwise items.

    Args:
      qp: Coordinate/velocity frame of the bodies.

    Returns:
      dP: Impulse to apply to the bodies in the collision.
    """
    qp_a = take(qp, self.body_a.idx)
    qp_b = take(qp, self.body_b.idx)
    dp_a, dp_b = self.apply_reduced(qp_a, qp_b)

    # sum together all impulse contributions across parents and children
    if dp_b is not None:
      body_idx = jnp.concatenate((self.body_a.idx, self.body_b.idx))
      dp_vel = jnp.concatenate((dp_a.vel, dp_b.vel))
      dp_ang = jnp.concatenate((dp_a.ang, dp_b.ang))
    else:
      body_idx = self.body_a.idx
      dp_vel = dp_a.vel
      dp_ang = dp_a.ang

    contact = jnp.where(jnp.any(dp_vel, axis=-1), 1.0, 0.0)
    contact = jax.ops.segment_sum(contact, body_idx, qp.pos.shape[0])
    dp_vel = jax.ops.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = jax.ops.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    # equally distribute impulse over possible contacts
    dp_vel = dp_vel / jnp.reshape(1e-8 + contact, (dp_vel.shape[0], 1))
    dp_ang = dp_ang / jnp.reshape(1e-8 + contact, (dp_ang.shape[0], 1))

    return P(vel=dp_vel, ang=dp_ang)

  @abc.abstractmethod
  def apply_reduced(self, qp_a: QP, qp_b: QP) -> Tuple[P, Optional[P]]:
    """Returns impulses from a collision between pairwise items.

    Operates in reduced collider space.

    Args:
      qp_a: Body A state data
      qp_b: Body B state data

    Returns:
      dp_a: Body A impulse
      dp_b: Body B impulse
    """

  def _contact_oneway(self, qp: QP, pos: jnp.ndarray, vel: jnp.ndarray,
                      normal: jnp.ndarray, penetration: float) -> P:
    """Calculates impulse on a single body due to a collision.

    Args:
      qp: State for body
      pos: Where the collision is occurring in world space
      vel: How fast the collision is happening
      normal: Normal vector of surface providing collision
      penetration: How far the collider is penetrating into the other

    Returns:
      dP: The impulse on this body result from the collision.
    """
    rel_pos = pos - qp.pos

    baumgarte_vel = self.baumgarte_erp * penetration
    normal_vel = jnp.dot(normal, vel)
    temp1 = jnp.matmul(self.body_a.inertia, jnp.cross(rel_pos, normal))
    ang = jnp.dot(normal, jnp.cross(temp1, rel_pos))
    impulse = (-1. * (1. + self.elasticity) * normal_vel + baumgarte_vel) / (
        (1. / self.body_a.mass) + ang)
    dp_n = self.body_a.impulse(qp, impulse * normal, pos)

    # apply drag due to friction acting parallel to the surface contact
    vel_d = vel - normal_vel * normal
    impulse_d = math.safe_norm(vel_d) / ((1. / (self.body_a.mass)) + ang)
    # drag magnitude cannot exceed max friction
    impulse_d = jnp.minimum(impulse_d, self.friction * impulse)
    dir_d = vel_d / (1e-6 + math.safe_norm(vel_d))
    dp_d = self.body_a.impulse(qp, -impulse_d * dir_d, pos)

    # apply collision if penetrating, approaching, and oriented correctly
    apply_n = jnp.where(
        (penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)

    # apply drag if moving laterally above threshold
    apply_d = apply_n * jnp.where(math.safe_norm(vel_d) > 0.01, 1., 0.)

    return dp_n * apply_n + dp_d * apply_d

  def _contact_twoway(self, qp_a: QP, qp_b: QP, pos: jnp.ndarray,
                      normal: jnp.ndarray, penetration: float) -> Tuple[P, P]:
    """Calculates impulse change due on two bodies to a collision.

    Args:
      qp_a: State for first body
      qp_b: State for second body
      pos: Where the collision is occurring in world space
      normal: Normal vector of surface providing collision
      penetration: How far the collider is penetrating into the other

    Returns:
      dP_a: The impulse on body A resulting from the collision
      dP_b: The impulse on body B resulting from the collision
    """
    rel_pos_a = pos - qp_a.pos
    rel_pos_b = pos - qp_b.pos

    vel = math.world_velocity(qp_b, pos) - math.world_velocity(qp_a, pos)
    baumgarte_vel = self.baumgarte_erp * penetration
    normal_vel = jnp.dot(normal, vel)
    temp1 = jnp.matmul(self.body_a.inertia, jnp.cross(rel_pos_a, normal))
    temp2 = jnp.matmul(self.body_b.inertia, jnp.cross(rel_pos_b, normal))
    ang = jnp.dot(normal,
                  jnp.cross(temp1, rel_pos_a) + jnp.cross(temp2, rel_pos_b))
    impulse = (-1. * (1. + self.elasticity) * normal_vel + baumgarte_vel) / (
        (1. / self.body_a.mass) + (1. / self.body_b.mass) + ang)
    dp_n_a = self.body_a.impulse(qp_a, -impulse * normal, pos)
    dp_n_b = self.body_b.impulse(qp_b, impulse * normal, pos)

    # apply drag due to friction acting parallel to the surface contact
    vel_d = vel - normal_vel * normal
    impulse_d = math.safe_norm(vel_d) / ((1. / self.body_a.mass) +
                                         (1. / self.body_b.mass) + ang)
    # drag magnitude cannot exceed max friction
    impulse_d = jnp.minimum(impulse_d, self.friction * impulse)
    dir_d = vel_d / (1e-6 + math.safe_norm(vel_d))
    dp_d_a = self.body_a.impulse(qp_a, impulse_d * dir_d, pos)
    dp_d_b = self.body_b.impulse(qp_b, -impulse_d * dir_d, pos)

    # apply collision normal if penetrating, approaching, and oriented correctly
    apply_n = jnp.where(
        (penetration > 0.) & (normal_vel < 0) & (impulse > 0.), 1., 0.)

    # apply drag if moving laterally above threshold
    apply_d = apply_n * jnp.where(math.safe_norm(vel_d) > 0.01, 1., 0.)

    dp_a = dp_n_a * apply_n + dp_d_a * apply_d
    dp_b = dp_n_b * apply_n + dp_d_b * apply_d

    return dp_a, dp_b


@pytree.register
class BoxPlane(PairwiseCollider):
  """A collision between a box and a static plane."""

  def __init__(self, colliders: List[Tuple[config_pb2.Body, config_pb2.Body]],
               body: bodies.Body, config: config_pb2.Config):
    super().__init__(colliders, body, config)

    # product of all possible box corners: [-1, -1, -1], [-1, -1, 1], ...
    coords = jnp.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))))
    corners = []
    for b, _ in colliders:
      col = b.colliders[0]
      rot = euler_to_quat(col.rotation)
      box = coords * vec_to_np(col.box.halfsize)
      box = jax.vmap(math.rotate, in_axes=(0, None))(box, rot)
      box = box + vec_to_np(col.position)
      corners.extend(box)

    self.corner = jnp.array(corners)
    # repeat body for each corner
    self.body_a = jax.tree_map(lambda x: jnp.repeat(x, 8, axis=0), self.body_a)
    self.body_b = jax.tree_map(lambda x: jnp.repeat(x, 8, axis=0), self.body_b)

  @jax.vmap
  def apply_reduced(self, qp_a: QP, qp_b: QP) -> Tuple[P, Optional[P]]:
    """Returns impulses from a collision between pairwise items."""
    pos, vel = math.to_world(qp_a, self.corner)
    normal = math.rotate(jnp.array([0., 0., 1.]), qp_b.rot)
    penetration = jnp.dot(qp_b.pos - pos, normal)
    dp = self._contact_oneway(qp_a, pos, vel, normal, penetration)
    return dp, None


@pytree.register
class BoxHeightMap(PairwiseCollider):
  """A collision between a box and a height map.

  Note that this only checks box corners against height map surfaces, and is
  missing box planes against height map points.
  """

  def __init__(self, colliders: List[Tuple[config_pb2.Body, config_pb2.Body]],
               body: bodies.Body, config: config_pb2.Config):
    super().__init__(colliders, body, config)

    # product of all possible box corners: [-1, -1, -1], [1, -1, -1], ...
    corners = list(itertools.product((-1, 1), (-1, 1), (-1, 1)))
    corners = jnp.array(corners, dtype=jnp.float32)
    boxes = []
    for b, _ in colliders:
      col = b.colliders[0]
      rot = euler_to_quat(col.rotation)
      box = corners * vec_to_np(col.box.halfsize)
      box = jax.vmap(math.rotate, in_axes=(0, None))(box, rot)
      box = box + vec_to_np(col.position)
      boxes.append(box)

    heights = []
    cell_sizes = []
    for _, h in colliders:
      col = h.colliders[0]
      mesh_size = int(jnp.sqrt(len(col.heightMap.data)))
      if len(col.heightMap.data) != mesh_size**2:
        raise ValueError('height map data length should be a perfect square.')
      height = jnp.array(col.heightMap.data).reshape((mesh_size, mesh_size))
      heights.append(height)
      cell_sizes.append(col.heightMap.size / (mesh_size - 1))

    self.box = jnp.array(boxes)
    self.height = jnp.array(heights)
    self.cell_size = jnp.array(cell_sizes)

  @jax.vmap
  def apply_reduced(self, qp_a: QP, qp_b: QP) -> Tuple[P, Optional[P]]:
    """Returns impulses from a collision between pairwise items."""
    @jax.vmap
    def each_corner(corner):
      pos, vel = math.to_world(qp_a, corner)
      pos = math.inv_rotate(pos - qp_b.pos, qp_b.rot)
      uv_pos = pos[:2] / self.cell_size

      # find the square in the mesh that enclose the candidate point, with mesh
      # indices ux_idx, ux_udx_u, uv_idx_v, uv_idx_uv.
      uv_idx = jnp.floor(uv_pos).astype(jnp.int32)
      uv_idx_u = uv_idx + jnp.array([1, 0], dtype=jnp.int32)
      uv_idx_v = uv_idx + jnp.array([0, 1], dtype=jnp.int32)
      uv_idx_uv = uv_idx + jnp.array([1, 1], dtype=jnp.int32)

      # find the orientation of the triangle of this square that encloses the
      # candidate point
      delta_uv = uv_pos - uv_idx
      # whether the corner lies on the first or secound triangle:
      mu = jnp.where(delta_uv[0] + delta_uv[1] < 1, 1, -1)

      # compute the mesh indices of the vertices of this triangle
      p0 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx, uv_idx_uv)
      p1 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_u, uv_idx_v)
      p2 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_v, uv_idx_u)

      h0 = self.height[p0[0], p0[1]]
      h1 = self.height[p1[0], p1[1]]
      h2 = self.height[p2[0], p2[1]]

      raw_normal = jnp.array([-mu * (h1 - h0), -mu * (h2 - h0), self.cell_size])
      normal = raw_normal / jnp.linalg.norm(raw_normal)
      normal = math.rotate(normal, qp_b.rot)

      height = jnp.array([p0[0] * self.cell_size, p0[1] * self.cell_size, h0])
      penetration = jnp.dot(height - pos, normal)

      return self._contact_oneway(qp_a, pos, vel, normal, penetration)

    dp = each_corner(self.box)
    # average over all the corner contacts
    contact_count = jnp.sum(jnp.any(dp.vel, axis=-1))
    dp = jax.tree_map(lambda x: jnp.sum(x, axis=0) / (1e-8 + contact_count), dp)
    return dp, None


@pytree.register
class CapsulePlane(PairwiseCollider):
  """A collision between a capsule and a static plane."""

  def __init__(self, colliders: List[Tuple[config_pb2.Body, config_pb2.Body]],
               body: bodies.Body, config: config_pb2.Config):
    super().__init__(colliders, body, config)

    body_idx = []
    ends = []
    radii = []
    for idx, (c, _) in enumerate(colliders):
      col = c.colliders[0]
      for end in [col.capsule.end] if col.capsule.end else [-1, 1]:
        body_idx.append(idx)
        axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
        segment_length = col.capsule.length / 2 - col.capsule.radius
        ends.append(vec_to_np(col.position) + end * axis * segment_length)
        radii.append(col.capsule.radius)

    # re-take bodies to account for multiple ends per body
    self.body_a = take(self.body_a, jnp.array(body_idx))
    self.body_b = take(self.body_b, jnp.array(body_idx))
    self.end = jnp.array(ends)
    self.radius = jnp.array(radii)

  @jax.vmap
  def apply_reduced(self, qp_a: QP, qp_b: QP) -> Tuple[P, Optional[P]]:
    """Returns impulses from a collision between pairwise items."""
    cap_end_world = qp_a.pos + math.rotate(self.end, qp_a.rot)
    normal = math.rotate(jnp.array([0., 0., 1.]), qp_b.rot)
    pos = cap_end_world - normal * self.radius
    rvel = jnp.cross(qp_a.ang, pos - qp_a.pos)
    vel = qp_a.vel + rvel
    penetration = jnp.dot(qp_b.pos - pos, normal)
    dp = self._contact_oneway(qp_a, pos, vel, normal, penetration)
    return dp, None


@pytree.register
class CapsuleCapsule(PairwiseCollider):
  """A collision between two capsules."""

  def __init__(self, colliders: List[Tuple[config_pb2.Body, config_pb2.Body]],
               body: bodies.Body, config: config_pb2.Config):
    super().__init__(colliders, body, config)

    ends = []
    radii = []
    for a, b in colliders:
      col_a, col_b = a.colliders[0], b.colliders[0]
      radii.append((col_a.capsule.radius, col_b.capsule.radius))
      def cap_end(col):
        axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
        segment_length = col.capsule.length / 2. - col.capsule.radius
        return vec_to_np(col.position) + axis * segment_length
      ends.append((cap_end(col_a), cap_end(col_b)))

    self.end_a = jnp.array([a for a, _ in ends])
    self.end_b = jnp.array([b for _, b in ends])
    self.radius_a = jnp.array([a for a, _ in radii])
    self.radius_b = jnp.array([b for _, b in radii])

  @jax.vmap
  def apply_reduced(self, qp_a: QP, qp_b: QP) -> Tuple[P, Optional[P]]:
    """Returns impulses from a collision between pairwise items."""

    def endpoints(end, qp):
      end = math.rotate(end, qp.rot)
      return qp.pos + end, qp.pos - end

    def closest_segment_point(a, b, pt):
      ab = b - a
      t = jnp.dot(pt - a, ab) / jnp.dot(ab, ab)
      return a + jnp.clip(t, 0., 1.) * ab

    a0, a1 = endpoints(self.end_a, qp_a)
    b0, b1 = endpoints(self.end_b, qp_b)
    v0, v1, v2, v3 = b0 - a0, b1 - a0, b0 - a1, b1 - a1
    d0, d1 = jnp.dot(v0, v0), jnp.dot(v1, v1)
    d2, d3 = jnp.dot(v2, v2), jnp.dot(v3, v3)
    a_best = jnp.where((d2 < d0) | (d2 < d1) | (d3 < d0) | (d3 < d1), a1, a0)
    b_best = closest_segment_point(b0, b1, a_best)
    a_best = closest_segment_point(a0, a1, b_best)

    penetration_vec = b_best - a_best
    dist = math.safe_norm(penetration_vec)
    normal = penetration_vec / (1e-6 + dist)
    penetration = self.radius_a + self.radius_b - dist
    pos = (a_best + b_best) / 2

    return self._contact_twoway(qp_a, qp_b, pos, normal, penetration)


def get(config: config_pb2.Config, body: bodies.Body) -> List[PairwiseCollider]:
  """Creates all colliders given a config."""
  include = {tuple(sorted((f.first, f.second))) for f in config.collide_include}

  # exclude colliders for joint parents and children, unless explicitly included
  ignore = {tuple(sorted((j.parent, j.child))) for j in config.joints}
  ignore -= include

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

  # group pair-wise combinations by type
  combos = {}
  for body_a, body_b in itertools.combinations(flat_bodies, 2):
    body_a, body_b = sorted((body_a, body_b), key=lambda b: b.name)
    if include and ((body_a.name, body_b.name) not in include):
      continue
    if (body_a.name, body_b.name) in ignore:
      continue
    if body_a.frozen.all and body_b.frozen.all:
      continue
    type_a = body_a.colliders[0].WhichOneof('type')
    type_b = body_b.colliders[0].WhichOneof('type')
    (type_a, body_a), (type_b, body_b) = sorted(
        ((type_a, body_a), (type_b, body_b)), key=lambda tb: tb[0])
    if (type_a, type_b) not in combos:
      combos[(type_a, type_b)] = []
    combos[(type_a, type_b)].append((body_a, body_b))

  # add colliders
  supported_types = {
      ('box', 'plane'): BoxPlane,
      ('box', 'heightMap'): BoxHeightMap,
      ('capsule', 'plane'): CapsulePlane,
      ('capsule', 'capsule'): CapsuleCapsule
  }
  ret = []
  for type_pair, colliders in combos.items():
    if type_pair in supported_types:
      ret.append(supported_types[type_pair](colliders, body, config))
    else:
      warnings.warn(f'unsupported collider type pair: {type_pair}')

  return ret
