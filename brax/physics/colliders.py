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
"""Calculate collisions of bodies."""

import itertools
from typing import List, Tuple

from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import P, QP, euler_to_quat, take, vec_to_np
from flax import struct
import jax
from jax import ops
import jax.numpy as jnp


class BoxPlane:
  """A collision between a box corner and a static plane."""

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'box', 'plane')
    if not self.pairs:
      return

    body_idx = {b.name: i for i, b in enumerate(config.bodies)}
    box_idxs = []
    plane_idxs = []
    corners = []

    for box, plane in self.pairs:
      if not plane.frozen.all:
        raise ValueError('active planes unsupported: %s' % plane)
      for i in range(8):
        box_idxs.append(body_idx[box.name])
        plane_idxs.append(body_idx[plane.name])
        corner = jnp.array(
            [i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1],
            dtype=jnp.float32)
        col = box.colliders[0]
        corner = corner * vec_to_np(col.box.halfsize)
        corner = math.rotate(corner, euler_to_quat(col.rotation))
        corner = corner + vec_to_np(col.position)
        corners.append(corner)

    body = bodies.Body.from_config(config)
    self.box = take(body, jnp.array(box_idxs))
    self.plane = take(body, jnp.array(plane_idxs))
    self.corner = jnp.array(corners)

  def apply(self, qp: QP, dt: float) -> P:
    """Returns impulse from a collision between box corners and a static plane.

    Note that impulses returned by this function are *not* scaled by dt when
    applied to parts.  Collision impulses are applied directly as velocity and
    angular velocity updates.

    Args:
      qp: Coordinate/velocity frame of the bodies.
      dt: Integration time step length.

    Returns:
      dP: Delta velocity to apply to the box bodies in the collision.
      colliding: Mask for each body: 1 = colliding, 0 = not colliding.
    """
    if not self.pairs:
      return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

    @jax.vmap
    def apply(box, corner, qp_box, qp_plane):
      pos, vel = math.to_world(qp_box, corner)
      normal = math.rotate(jnp.array([0.0, 0.0, 1.0]), qp_plane.rot)
      penetration = jnp.dot(pos - qp_plane.pos, normal)
      dp = _collide(self.config, box, qp_box, pos, vel, normal, penetration, dt)
      collided = jnp.where(penetration < 0., 1., 0.)
      return dp, collided

    qp_box = take(qp, self.box.idx)
    qp_plane = take(qp, self.plane.idx)
    dp, colliding = apply(self.box, self.corner, qp_box, qp_plane)

    # collapse/sum across all corners
    num_bodies = len(self.config.bodies)
    colliding = ops.segment_sum(colliding, self.box.idx, num_bodies)
    vel = ops.segment_sum(dp.vel, self.box.idx, num_bodies)
    ang = ops.segment_sum(dp.ang, self.box.idx, num_bodies)

    # equally distribute contact force over each box
    vel = vel / jnp.reshape(1e-8 + colliding, (vel.shape[0], 1))
    ang = ang / jnp.reshape(1e-8 + colliding, (ang.shape[0], 1))

    return P(vel, ang)


class BoxHeightMap:
  """A collision between box corners and a height map."""

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'box', 'heightMap')
    if not self.pairs:
      return

    body_idx = {b.name: i for i, b in enumerate(config.bodies)}
    box_idxs = []
    height_map_idxs = []
    corners = []
    heights = []
    sizes = []
    mesh_sizes = []

    for box, height_map in self.pairs:
      if not height_map.frozen.all:
        raise ValueError('active height maps unsupported: %s' % height_map)
      for i in range(8):
        box_idxs.append(body_idx[box.name])
        height_map_idxs.append(body_idx[height_map.name])
        corner = jnp.array(
            [i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1],
            dtype=jnp.float32)
        col = box.colliders[0]
        corner = corner * vec_to_np(col.box.halfsize)
        corner = math.rotate(corner, euler_to_quat(col.rotation))
        corner = corner + vec_to_np(col.position)
        corners.append(corner)

        mesh_size = int(
            jnp.round(jnp.sqrt(len(height_map.colliders[0].heightMap.data))))
        if len(height_map.colliders[0].heightMap.data) != mesh_size**2:
          raise ValueError(
              'data length for an height map should be a perfect square.')

        height = jnp.array(height_map.colliders[0].heightMap.data).reshape(
            (mesh_size, mesh_size))
        heights.append(height)
        sizes.append(height_map.colliders[0].heightMap.size)
        mesh_sizes.append(mesh_size)

    body = bodies.Body.from_config(config)
    self.box = take(body, jnp.array(box_idxs))
    self.height_map = take(body, jnp.array(height_map_idxs))
    self.corner = jnp.array(corners)
    self.size = jnp.array(sizes)
    self.mesh_size = jnp.array(mesh_sizes)
    self.heights = jnp.array(heights)

  def apply(self, qp: QP, dt: float) -> P:
    """Returns impulse from a collision between box corners and a static height map.

    Note that impulses returned by this function are *not* scaled by dt when
    applied to parts.  Collision impulses are applied directly as velocity and
    angular velocity updates.
    Args:
      qp: Coordinate/velocity frame of the bodies.
      dt: Integration time step length.

    Returns:
      dP: Delta velocity to apply to the box bodies in the collision.
      colliding: Mask for each body: 1 = colliding, 0 = not colliding.
    """
    if not self.pairs:
      return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

    @jax.vmap
    def apply(box, corner, qp_box, qp_height_map, size, mesh_size, heights):
      world_pos, vel = math.to_world(qp_box, corner)
      pos = math.inv_rotate(world_pos - qp_height_map.pos, qp_height_map.rot)
      uv_pos = (pos[:2]) / size * (mesh_size - 1)

      # find the square in the mesh that enclose the candidate point, with mesh
      # indices ux_idx, ux_udx_u, uv_idx_v, uv_idx_uv.
      uv_idx = jnp.floor(uv_pos).astype(jnp.int32)
      uv_idx_u = uv_idx + jnp.array([1, 0], dtype=jnp.int32)
      uv_idx_v = uv_idx + jnp.array([0, 1], dtype=jnp.int32)
      uv_idx_uv = uv_idx + jnp.array([1, 1], dtype=jnp.int32)

      # find the orientation of the triangle of this square that encloses the
      # candidate point
      delta_uv = uv_pos - uv_idx
      mu = jnp.where(
          delta_uv[0] + delta_uv[1] < 1, 1,
          -1)  # whether the corner lies on the first or secound triangle

      # compute the mesh indices of the vertices of this triangle
      point_0 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx, uv_idx_uv)
      point_1 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_u, uv_idx_v)
      point_2 = jnp.where(delta_uv[0] + delta_uv[1] < 1, uv_idx_v, uv_idx_u)

      h0 = heights[point_0[0], point_0[1]]
      h1 = heights[point_1[0], point_1[1]]
      h2 = heights[point_2[0], point_2[1]]

      raw_normal = jnp.array(
          [-mu * (h1 - h0), -mu * (h2 - h0), 1 * (size / (mesh_size - 1))])
      normal = raw_normal / jnp.linalg.norm(raw_normal)
      rotated_normal = math.rotate(normal, qp_height_map.rot)

      pos_0 = jnp.array([
          point_0[0] * size / (mesh_size - 1),
          point_0[1] * size / (mesh_size - 1), h0
      ])
      penetration = jnp.dot(pos - pos_0, normal)

      dp = _collide(self.config, box, qp_box, pos, vel, rotated_normal,
                    penetration, dt)
      collided = jnp.where(penetration < 0., 1., 0.)
      return dp, collided

    qp_box = take(qp, self.box.idx)
    qp_height_map = take(qp, self.height_map.idx)
    dp, colliding = apply(self.box, self.corner, qp_box, qp_height_map,
                          self.size, self.mesh_size, self.heights)

    # collapse/sum across all corners
    num_bodies = len(self.config.bodies)
    colliding = ops.segment_sum(colliding, self.box.idx, num_bodies)
    vel = ops.segment_sum(dp.vel, self.box.idx, num_bodies)
    ang = ops.segment_sum(dp.ang, self.box.idx, num_bodies)

    # equally distribute contact force over each box (corner ?)
    vel = vel / jnp.reshape(1e-8 + colliding, (vel.shape[0], 1))
    ang = ang / jnp.reshape(1e-8 + colliding, (ang.shape[0], 1))

    return P(vel, ang)


class CapsulePlane:
  """A collision between a capsule and a static plane."""

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'capsule', 'plane')
    self.pairs += _find_body_pairs(config, 'sphere', 'plane')
    if not self.pairs:
      return

    body_idx = {b.name: i for i, b in enumerate(config.bodies)}
    for _, plane in self.pairs:
      if not plane.frozen.all:
        raise ValueError('active planes unsupported: %s' % plane)

    cap_idx = []
    plane_idx = []
    cap_end = []
    cap_radius = []
    for cap, plane in self.pairs:
      if cap.colliders[0].WhichOneof('type') == 'sphere':
        capsule = config_pb2.Collider.Capsule()
        capsule.radius = cap.colliders[0].sphere.radius
        capsule.length = 2 * capsule.radius
        capsule.end = 1
      else:
        capsule = cap.colliders[0].capsule
      ends = [capsule.end] if capsule.end else [-1, 1]
      for end in ends:
        cap_idx.append(body_idx[cap.name])
        plane_idx.append(body_idx[plane.name])
        cap_pos = vec_to_np(cap.colliders[0].position)
        cap_rot = euler_to_quat(cap.colliders[0].rotation)
        cap_axis = math.rotate(jnp.array([0., 0., 1.]), cap_rot)
        cap_arm = capsule.length / 2 - capsule.radius
        cap_end.append(cap_pos + end * cap_axis * cap_arm)
        cap_radius.append(capsule.radius)

    body = bodies.Body.from_config(config)
    self.cap = take(body, jnp.array(cap_idx))
    self.plane = take(body, jnp.array(plane_idx))
    self.cap_end = jnp.array(cap_end)
    self.cap_radius = jnp.array(cap_radius)

  def apply(self, qp: QP, dt: float) -> P:
    """Returns an impulse from a collision between a capsule and a static plane.

    Note that impulses returned by this function are *not* scaled by dt when
    applied to parts.  Collision impulses are applied directly as velocity and
    angular velocity updates.

    Args:
      qp: Coordinate/velocity frame of the bodies.
      dt: Integration time step length.

    Returns:
      dP: Delta velocity to apply to the capsule bodies in the collision.
      colliding: Mask for each body: 1 = colliding, 0 = not colliding.
    """
    if not self.pairs:
      return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

    @jax.vmap
    def apply(cap, cap_end, radius, qp_cap, qp_plane):
      cap_end_world = qp_cap.pos + math.rotate(cap_end, qp_cap.rot)
      normal = math.rotate(jnp.array([0.0, 0.0, 1.0]), qp_plane.rot)
      pos = cap_end_world - normal * radius
      rpos_off = pos - qp_cap.pos
      rvel = jnp.cross(qp_cap.ang, rpos_off)
      vel = qp_cap.vel + rvel

      penetration = jnp.dot(pos - qp_plane.pos, normal)
      dp = _collide(self.config, cap, qp_cap, pos, vel, normal, penetration, dt)
      colliding = jnp.where(penetration < 0., 1., 0.)
      return dp, colliding

    qp_cap = take(qp, self.cap.idx)
    qp_plane = take(qp, self.plane.idx)
    dp, colliding = apply(self.cap, self.cap_end, self.cap_radius, qp_cap,
                          qp_plane)

    # sum across both contact points
    num_bodies = len(self.config.bodies)
    colliding = ops.segment_sum(colliding, self.cap.idx, num_bodies)
    vel = ops.segment_sum(dp.vel, self.cap.idx, num_bodies)
    ang = ops.segment_sum(dp.ang, self.cap.idx, num_bodies)

    # equally distribute contact force over possible collision points
    vel = vel / jnp.reshape(1e-8 + colliding, (vel.shape[0], 1))
    ang = ang / jnp.reshape(1e-8 + colliding, (ang.shape[0], 1))

    return P(vel, ang)


class CapsuleCapsule:
  """A collision between two capsules."""

  @struct.dataclass
  class Capsule:
    body: bodies.Body
    radius: jnp.ndarray
    end: jnp.ndarray

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'capsule', 'capsule')
    if not self.pairs:
      return None

    body_idx = {b.name: i for i, b in enumerate(config.bodies)}
    body_idxs = []
    radii = []
    ends = []

    for body_a, body_b in self.pairs:
      col_a, col_b = body_a.colliders[0], body_b.colliders[0]
      body_idxs.append((body_idx[body_a.name], body_idx[body_b.name]))
      radii.append((col_a.capsule.radius, col_b.capsule.radius))
      def cap_end(col):
        axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
        segment_length = col.capsule.length / 2. - col.capsule.radius
        return vec_to_np(col.position) + axis * segment_length
      ends.append((cap_end(col_a), cap_end(col_b)))

    body = bodies.Body.from_config(config)
    self.cap_a = CapsuleCapsule.Capsule(
        body=take(body, jnp.array([a for a, _ in body_idxs])),
        radius=jnp.array([a for a, _ in radii]),
        end=jnp.array([a for a, _ in ends]))
    self.cap_b = CapsuleCapsule.Capsule(
        body=take(body, jnp.array([b for _, b in body_idxs])),
        radius=jnp.array([b for _, b in radii]),
        end=jnp.array([b for _, b in ends]))

  def apply(self, qp: QP, dt: float) -> P:
    """Returns impulses between capsules in collision.

    Args:
      qp: Coordinate/velocity frame of the bodies.
      dt: Integration time step length.

    Returns:
      dP: Delta velocity to apply to the capsule bodies in the collision.
      colliding: Mask for each body: 1 = colliding, 0 = not colliding.
    """
    if not self.pairs:
      return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

    @jax.vmap
    def apply(cap_a, cap_b, qp_a, qp_b):
      """Extracts collision points and applies collision to capsules."""

      def endpoints(cap, qp):
        end = math.rotate(cap.end, qp.rot)
        return qp.pos + end, qp.pos - end

      def closest_segment_point(a, b, pt):
        ab = b - a
        t = jnp.dot(pt - a, ab) / jnp.dot(ab, ab)
        return a + jnp.clip(t, 0., 1.) * ab

      a_A, a_B = endpoints(cap_a, qp_a)
      b_A, b_B = endpoints(cap_b, qp_b)

      v0 = b_A - a_A
      v1 = b_B - a_A
      v2 = b_A - a_B
      v3 = b_B - a_B

      d0 = jnp.dot(v0, v0)
      d1 = jnp.dot(v1, v1)
      d2 = jnp.dot(v2, v2)
      d3 = jnp.dot(v3, v3)

      bestA = jnp.where((d2 < d0) | (d2 < d1) | (d3 < d0) | (d3 < d1), a_B, a_A)
      bestB = closest_segment_point(b_A, b_B, bestA)
      bestA = closest_segment_point(a_A, a_B, bestB)

      penetration_vec = bestB - bestA
      dist = math.safe_norm(penetration_vec)
      collision_normal = penetration_vec / (1e-6 + dist)
      penetration = dist - cap_a.radius - cap_b.radius
      pos_c = (bestA + bestB) / 2.

      dp_a, dp_b = _collide_pair(self.config, cap_a.body, cap_b.body, qp_a,
                                 qp_b, pos_c, collision_normal, penetration, dt)
      return dp_a, dp_b

    qp_a = take(qp, self.cap_a.body.idx)
    qp_b = take(qp, self.cap_b.body.idx)
    dp_a, dp_b = apply(self.cap_a, self.cap_b, qp_a, qp_b)

    num_bodies = len(self.config.bodies)
    vel_a = ops.segment_sum(dp_a.vel, self.cap_a.body.idx, num_bodies)
    ang_a = ops.segment_sum(dp_a.ang, self.cap_a.body.idx, num_bodies)
    vel_b = ops.segment_sum(dp_b.vel, self.cap_b.body.idx, num_bodies)
    ang_b = ops.segment_sum(dp_b.ang, self.cap_b.body.idx, num_bodies)

    return P(vel_a + vel_b, ang_a + ang_b)


def _find_body_pairs(
    config: config_pb2.Config, type_a: str,
    type_b: str) -> List[Tuple[config_pb2.Body, config_pb2.Body]]:
  """Returns body pairs with colliders matching types."""
  include = {(f.first, f.second) for f in config.collide_include}
  include |= {(f.second, f.first) for f in config.collide_include}

  # exclude collisions between parents and children of joints, unless
  # explicitly included
  ignore = {(j.parent, j.child) for j in config.joints}
  ignore |= {(j.child, j.parent) for j in config.joints}
  ignore -= include

  # flatten and emit one collider per body
  flat_bodies = []
  for body in config.bodies:
    for collider in body.colliders:
      new_body = config_pb2.Body()
      new_body.CopyFrom(body)
      new_body.ClearField('colliders')
      new_body.colliders.append(collider)
      flat_bodies.append(new_body)

  ret = []
  for body_a, body_b in itertools.combinations(flat_bodies, 2):
    if include and ((body_a.name, body_b.name) not in include):
      continue
    if (body_a.name, body_b.name) in ignore:
      continue
    if body_a.frozen.all and body_b.frozen.all:
      continue

    collider_type_a = body_a.colliders[0].WhichOneof('type')
    collider_type_b = body_b.colliders[0].WhichOneof('type')
    if (collider_type_a, collider_type_b) == (type_a, type_b):
      ret.append((body_a, body_b))
    elif (collider_type_b, collider_type_a) == (type_a, type_b):
      ret.append((body_b, body_a))
  return ret


def _collide(config: config_pb2.Config, body: bodies.Body, qp: QP,
             pos_c: jnp.ndarray, vel_c: jnp.ndarray, normal_c: jnp.ndarray,
             penetration: float, dt: float) -> P:
  """Calculates velocity change due to a collision.

  Args:
    config: A brax config.
    body: Body participating in collision
    qp: State for body
    pos_c: Where the collision is occuring in world space
    vel_c: How fast the collision is happening
    normal_c: Normal vector of surface providing collision
    penetration: Amount of penetration between part and plane
    dt: Integration timestep length

  Returns:
    dP: The impulse on this body result from the collision.
  """
  rel_pos_a = pos_c - qp.pos

  baumgarte_rel_vel = config.baumgarte_erp * penetration / dt
  normal_rel_vel = jnp.dot(normal_c, vel_c)

  temp1 = jnp.matmul(body.inertia, jnp.cross(rel_pos_a, normal_c))
  ang = jnp.dot(normal_c, jnp.cross(temp1, rel_pos_a))
  impulse = (-1. *
             (1. + config.elasticity) * normal_rel_vel - baumgarte_rel_vel) / (
                 (1. / body.mass) + ang)

  dp_n = body.impulse(qp, impulse * normal_c, pos_c)

  # apply drag due to friction acting parallel to the surface contact
  rel_vel_d = vel_c - normal_rel_vel * normal_c
  impulse_d = math.safe_norm(rel_vel_d) / ((1. / (body.mass)) + ang)
  # drag magnitude cannot exceed max friction
  impulse_d = jnp.where(impulse_d < config.friction * impulse, impulse_d,
                        config.friction * impulse)
  dir_d = rel_vel_d / (1e-6 + math.safe_norm(rel_vel_d))
  dp_d = body.impulse(qp, -impulse_d * dir_d, pos_c)

  # apply collision normal if penetrating, approaching, and oriented correctly
  colliding_n = jnp.where(penetration < 0., 1., 0.)
  colliding_n *= jnp.where(normal_rel_vel < 0, 1., 0.)
  colliding_n *= jnp.where(impulse > 0., 1., 0.)

  # apply drag if moving laterally above threshold
  colliding_d = colliding_n
  colliding_d *= jnp.where(math.safe_norm(rel_vel_d) > (1. / 100.), 1., 0.)

  return dp_n * colliding_n + dp_d * colliding_d


def _collide_pair(config: config_pb2.Config, body_a: bodies.Body,
                  body_b: bodies.Body, qp_a: QP, qp_b: QP, pos_c: jnp.ndarray,
                  normal_c: jnp.ndarray, penetration: float,
                  dt: float) -> Tuple[P, P]:
  """Calculates velocity change due to a collision.

  Args:
    config: A brax system config.
    body_a: First colliding body
    body_b: Second colliding body
    qp_a: State for first body
    qp_b: State for second body
    pos_c: Where the collision is occuring in world space
    normal_c: Normal vector of surface providing collision
    penetration: Amount of penetration between part and plane
    dt: Integration timestep length

  Returns:
    dP: The impulse on this body result from the collision.
  """
  # TODO: possibly refactor into single collide fn?
  rel_pos_a = pos_c - qp_a.pos
  rel_pos_b = pos_c - qp_b.pos

  vel_a_c = math.world_velocity(qp_a, pos_c)
  vel_b_c = math.world_velocity(qp_b, pos_c)

  rel_vel = vel_b_c - vel_a_c

  baumgarte_rel_vel = config.baumgarte_erp * penetration / dt
  normal_rel_vel = jnp.dot(normal_c, rel_vel)

  temp1 = jnp.matmul(body_a.inertia, jnp.cross(rel_pos_a, normal_c))
  temp2 = jnp.matmul(body_b.inertia, jnp.cross(rel_pos_b, normal_c))

  ang = jnp.dot(normal_c,
                jnp.cross(temp1, rel_pos_a) + jnp.cross(temp2, rel_pos_b))
  impulse = (-1. *
             (1. + config.elasticity) * normal_rel_vel - baumgarte_rel_vel) / (
                 (1. / body_a.mass) + (1. / body_b.mass) + ang)

  dp_n_a = body_a.impulse(qp_a, -impulse * normal_c, pos_c)
  dp_n_b = body_b.impulse(qp_b, impulse * normal_c, pos_c)

  # apply drag due to friction acting parallel to the surface contact
  rel_vel_d = rel_vel - normal_rel_vel * normal_c
  impulse_d = math.safe_norm(rel_vel_d) / ((1. / body_a.mass) +
                                           (1. / body_b.mass) + ang)
  # drag magnitude cannot exceed max friction
  impulse_d = jnp.where(impulse_d < config.friction * impulse, impulse_d,
                        config.friction * impulse)
  dir_d = rel_vel_d / (1e-6 + math.safe_norm(rel_vel_d))
  dp_d_a = body_a.impulse(qp_a, impulse_d * dir_d, pos_c)
  dp_d_b = body_b.impulse(qp_b, -impulse_d * dir_d, pos_c)

  # apply collision normal if penetrating, approaching, and oriented correctly
  colliding_n = jnp.where(penetration < 0., 1., 0.)
  colliding_n *= jnp.where(normal_rel_vel < 0, 1., 0.)
  colliding_n *= jnp.where(impulse > 0., 1., 0.)

  # apply drag if moving laterally above threshold
  colliding_d = colliding_n
  colliding_d *= jnp.where(math.safe_norm(rel_vel_d) > (1. / 100.), 1., 0.)

  dp_a = dp_n_a * colliding_n + dp_d_a * colliding_d
  dp_b = dp_n_b * colliding_n + dp_d_b * colliding_d

  return dp_a, dp_b
