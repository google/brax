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

from flax import struct
import jax
from jax import ops
import jax.numpy as jnp

from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import P, QP, euler_to_quat, vec_to_np, take
from brax.physics.math import safe_norm


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


class CapsulePlane:
  """A collision between a capsule and a static plane."""

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'capsule', 'plane')
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
    length: jnp.ndarray
    pos: jnp.ndarray
    axis: jnp.ndarray

  def __init__(self, config: config_pb2.Config):
    self.config = config
    self.pairs = _find_body_pairs(config, 'capsule', 'capsule')
    if not self.pairs:
      return None

    body_idx = {b.name: i for i, b in enumerate(config.bodies)}
    cols_a = [c.colliders[0] for c, _ in self.pairs]
    body_idx_a = jnp.array([body_idx[c.name] for c, _ in self.pairs])
    cols_rot_a = jnp.array([euler_to_quat(c.rotation) for c in cols_a])
    cols_b = [c.colliders[0] for _, c in self.pairs]
    body_idx_b = jnp.array([body_idx[c.name] for _, c in self.pairs])
    cols_rot_b = jnp.array([euler_to_quat(c.rotation) for c in cols_b])
    v_rot = jax.vmap(math.rotate, in_axes=[None, 0])

    body = bodies.Body.from_config(config)
    self.cap_a = CapsuleCapsule.Capsule(
        body=take(body, body_idx_a),
        radius=jnp.array([c.capsule.radius for c in cols_a]),
        length=jnp.array([c.capsule.length for c in cols_a]),
        pos=jnp.array([vec_to_np(c.position) for c in cols_a]),
        axis=v_rot(jnp.array([0., 0., 1.]), cols_rot_a))
    self.cap_b = CapsuleCapsule.Capsule(
        body=take(body, body_idx_b),
        radius=jnp.array([c.capsule.radius for c in cols_b]),
        length=jnp.array([c.capsule.length for c in cols_b]),
        pos=jnp.array([vec_to_np(c.position) for c in cols_b]),
        axis=v_rot(jnp.array([0., 0., 1.]), cols_rot_b))

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
      p1, p2, p1_p2_dist = _find_closest_segment(cap_a, cap_b, qp_a, qp_b)
      radius_sum = cap_a.radius + cap_b.radius

      penetration = p1_p2_dist - radius_sum
      collision_normal = (p2 - p1) / (1e-6 + math.safe_norm(p2 - p1))

      dp_a, dp_b = _collide_pair(self.config, cap_a.body, cap_b.body, qp_a,
                                 qp_b, (p1 * cap_b.radius + p2 * cap_a.radius) /
                                 radius_sum, collision_normal, penetration, dt)
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


def _find_closest_segment(cap_a: CapsuleCapsule.Capsule,
                          cap_b: CapsuleCapsule.Capsule, qp_a: QP,
                          qp_b: QP) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
  """Finds the closest points between two capsules."""

  def endpoints(axis, qp, radius, length, offset):
    segment_length = length / 2. - radius
    segment_point = axis * segment_length
    return qp.pos + offset + math.rotate(segment_point,
                                         qp.rot), qp.pos + offset + math.rotate(
                                             -1. * segment_point, qp.rot)

  def point_on_segment(close_a, close_b, point_b, unit_vec, projection,
                       segment_a_len, segment_b_len):
    """Finds the point on segment B closest to segment A.

    Represented in fn args, this finds the close_b satisfying
    point_b + unit_vec*dot = close_b

    such that close_b is as close as possible to close_a

    Args:
      close_a: The current closest candidate point to segment B on segment A
      close_b: The current closest candidate point to segment A on segment B
      point_b: A point on an endcap of segment B
      unit_vec: A unit vector pointing along the length of segment B, relative
        to point_b
      projection: Distance from segment B to segment A projected along segment
        B's axis
      segment_a_len: Length of segment A
      segment_b_len: Length of segment B

    Returns:
      Closest point on segment B
    """
    dot = jnp.dot(unit_vec, (close_a - point_b))
    dot = jnp.where(
        jnp.less(dot, 0), 0,
        jnp.where(jnp.greater(dot, segment_b_len), segment_b_len, dot))
    close_b = jnp.where(
        jnp.logical_or(
            jnp.less(projection, 0), jnp.greater(projection, segment_a_len)),
        point_b + (unit_vec * dot), close_b)
    return close_b

  a1, a2 = endpoints(cap_a.axis, qp_a, cap_a.radius, cap_a.length, cap_a.pos)
  b1, b2 = endpoints(cap_b.axis, qp_b, cap_b.radius, cap_b.length, cap_b.pos)

  # check if lines are overlapping
  a_segment = a2 - a1
  b_segment = b2 - b1
  a_len = safe_norm(a_segment)
  b_len = safe_norm(b_segment)

  a_unit_vec = a_segment / (1e-10 + a_len)
  b_unit_vec = b_segment / (1e-10 + b_len)

  cross = jnp.cross(a_unit_vec, b_unit_vec)
  denom = safe_norm(cross)**2.

  # closest point test if segments are parallel

  d1 = jnp.dot(a_unit_vec, (b1 - a1))
  d2 = jnp.dot(a_unit_vec, (b2 - a1))

  pa_par = jnp.zeros(3)
  pb_par = jnp.zeros(3)
  closest_dist = 0.

  segments_are_parallel = jnp.less(denom, 1e-6)
  b_is_before_a = jnp.greater_equal(jnp.less_equal(d1, 0), d2)
  a_is_before_b = jnp.less_equal(jnp.greater_equal(d1, a_len), d2)
  orientation_bool = jnp.less(jnp.absolute(d1), jnp.absolute(d2))

  # base case, segments perfectly overlap
  pa_par = jnp.where(segments_are_parallel, qp_a.pos, pa_par)
  pb_par = jnp.where(segments_are_parallel, qp_b.pos, pb_par)
  closest_dist = jnp.where(segments_are_parallel,
                           safe_norm(((d1 * a_unit_vec) + a1) - b1),
                           closest_dist)

  # segments parallel, with segment b before segment a
  pa_par = jnp.where(segments_are_parallel * b_is_before_a, a1, pa_par)
  pb_par = jnp.where(segments_are_parallel * b_is_before_a,
                     jnp.where(orientation_bool, b1, b2), pb_par)
  closest_dist = jnp.where(
      segments_are_parallel * b_is_before_a,
      jnp.where(orientation_bool, safe_norm(a1 - b1),
                safe_norm(a1 - b2)), closest_dist)

  # segments parallel, with segment a before segment b
  pa_par = jnp.where(
      segments_are_parallel * a_is_before_b * (1 - b_is_before_a), a2, pa_par)
  pb_par = jnp.where(
      segments_are_parallel * a_is_before_b * (1 - b_is_before_a),
      jnp.where(orientation_bool, b1, b2), pb_par)
  closest_dist = jnp.where(
      segments_are_parallel * a_is_before_b * (1 - b_is_before_a),
      jnp.where(orientation_bool, safe_norm(a2 - b1),
                safe_norm(a2 - b2)), closest_dist)

  # closest point test if segments are NOT parallel

  t = (b1 - a1)

  det_a = math.det(t, b_unit_vec, cross)
  det_b = math.det(t, a_unit_vec, cross)

  t1 = det_a / (1e-10 + denom)
  t2 = det_b / (1e-10 + denom)

  pa = a1 + (a_unit_vec * t1)  # Projected closest point on segment A
  pb = b1 + (b_unit_vec * t2)  # Projected closest point on segment B

  # if the closest point on the line running through segment_i to segment_j
  # is not actually contained within the segment, then clamp it to the closest
  # endcap of segment_i
  pa = jnp.where(jnp.less(t1, 0), a1, jnp.where(jnp.greater(t1, a_len), a2, pa))
  pb = jnp.where(jnp.less(t2, 0), b1, jnp.where(jnp.greater(t2, b_len), b2, pb))

  pb = point_on_segment(pa, pb, b1, b_unit_vec, t1, a_len, b_len)
  pa = point_on_segment(pb, pa, a1, a_unit_vec, t2, b_len, a_len)

  fa = lambda _: (pa, pb, safe_norm(pa - pb))
  fb = lambda _: (pa_par, pb_par, closest_dist)

  return jax.lax.cond(jnp.equal(segments_are_parallel, 0.), fa, fb, None)


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

  # factor of 2.0 here empirically helps object grip
  # TODO: expose friction physics parameters in config
  return dp_n * colliding_n + dp_d * colliding_d * 2.0


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
