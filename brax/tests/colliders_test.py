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

"""Tests for brax.physics.colliders."""

from absl.testing import absltest
import brax
from brax import jumpy as jp

from google.protobuf import text_format


class ClosestSegmentPointsTest(absltest.TestCase):
  """Tests for closest segment points."""

  def test_closest_segments_points(self):
    a0 = jp.array([0.73432405, 0.12372768, 0.20272314])
    a1 = jp.array([1.10600128, 0.88555209, 0.65209485])
    b0 = jp.array([0.85599262, 0.61736299, 0.9843583])
    b1 = jp.array([1.84270939, 0.92891793, 1.36343326])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [1.09063, 0.85404, 0.63351], 5)
    self.assertSequenceAlmostEqual(best_b, [0.99596, 0.66156, 1.03813], 5)

  def test_intersecting_segments(self):
    """Tests segments that intersect."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([-1., 0., 0.]), jp.array([1., 0., 0.])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [0., 0., 0.], 5)

  def test_intersecting_lines(self):
    """Tests that intersecting lines (not segments) get clipped."""
    a0, a1 = jp.array([0.2, 0.2, 0.]), jp.array([1., 1., 0.])
    b0, b1 = jp.array([0.2, 0.4, 0.]), jp.array([1., 2., 0.])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.3, 0.3, 0.], 2)
    self.assertSequenceAlmostEqual(best_b, [0.2, 0.4, 0.], 2)

  def test_parallel_segments(self):
    """Tests that parallel segments have closest points at the midpoint."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([1., 0., -1.]), jp.array([1., 0., 1.])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 0.], 5)

  def test_parallel_offset_segments(self):
    """Tests that offset parallel segments are close at segment endpoints."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([1., 0., 1.]), jp.array([1., 0., 3.])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 1.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 1.], 5)

  def test_zero_length_segments(self):
    """Test that zero length segments don't return NaNs."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., -1.])
    b0, b1 = jp.array([1., 0., 0.1]), jp.array([1., 0., 0.1])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., -1.], 5)
    self.assertSequenceAlmostEqual(best_b, [1., 0., 0.1], 5)

  def test_overlapping_segments(self):
    """Tests that perfectly overlapping segments intersect at the midpoints."""
    a0, a1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    b0, b1 = jp.array([0., 0., -1.]), jp.array([0., 0., 1.])
    best_a, best_b = brax.physics.colliders._closest_segment_to_segment_points(
        a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0., 0., 0.], 5)
    self.assertSequenceAlmostEqual(best_b, [0., 0., 0.], 5)


class CapsuleCapsuleColliderFnTest(absltest.TestCase):
  """Tests the capsule-capsule collision function."""

  _CONFIG = """
    dt: .01 substeps: 4 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Capsule1" mass: 1
      colliders { capsule { radius: 0.05 length: 0.88 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule2" mass: 1
      colliders { capsule { radius: 0.05 length: 0.75 } }
      inertia { x: 1 y: 1 z: 1 }
      frozen: { all: true }
    }
    defaults {
      qps {
        name: "Capsule1"
        pos {x: 1.0776836 y: 0.46205616 z: 10.814447 }
        rot {x: 37.72620165 y: 65.59518961 z: 63.17161424}
      }
      qps {
        name: "Capsule2"
        pos {x: 0.9702782 y: 0.45527467 z: 10.704371}
        rot {x: 33.0532861 y: 57.08759044 z: 87.28387177}
      }
    }
  """

  def test_collision(self):
    """Tests that two capsules are colliding."""
    config = text_format.Parse(CapsuleCapsuleColliderFnTest._CONFIG,
                               brax.Config())
    sys = brax.System(config)
    capsules = brax.physics.colliders.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)

    self.assertGreaterEqual(contact.penetration, 0)

  # Parallel capsule collision.
  _CONFIG2 = """
    dt: .01 substeps: 4 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Capsule1" mass: 1
      colliders { capsule { radius: 0.1 length: 1.0 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule2" mass: 1
      colliders { capsule { radius: 0.1 length: 1.0 } }
      inertia { x: 1 y: 1 z: 1 }
      frozen: { all: true }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps {
        name: "Capsule1"
        pos {x: 0 y: 0 z: 0.25 }
        rot {x: 0 y: 90 z: 0}
      }
      qps {
        name: "Capsule2"
        pos {x: 0 y: 0 z: 0.1}
        rot {x: 0 y: 90 z: 0}
      }
    }
  """

  def test_parallel_collision(self):
    """Tests that two parallel capsules are colliding at the midpoint."""
    config = text_format.Parse(CapsuleCapsuleColliderFnTest._CONFIG2,
                               brax.Config())
    sys = brax.System(config)
    capsules = brax.physics.colliders.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)
    self.assertAlmostEqual(contact.penetration, 0.05)
    self.assertSequenceAlmostEqual(contact.pos, [0., 0., (0.15 + 0.2) / 2.0], 4)

  def test_collision_with_zero_capsule_length(self):
    """Tests collisions with capsules that have zero length (i.e. spheres)."""
    config = text_format.Parse(CapsuleCapsuleColliderFnTest._CONFIG2,
                               brax.Config())
    sys = brax.System(config)
    capsules = brax.physics.colliders.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    # Set capsule ends such that the capsule lengths are zero.
    capsule1.end = jp.array([0., 0., 0.])
    capsule2.end = jp.array([0., 0., 0.])
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)
    self.assertAlmostEqual(contact.penetration, 0.05)
    self.assertSequenceAlmostEqual(contact.pos, [0., 0., (0.15 + 0.2) / 2.0], 4)


class BoxHeightMapColliderFnTest(absltest.TestCase):
  """Tests the box-heightmap collision function."""
  _CONFIG = """
    dt: .01 substeps: 4 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.3 y: 0.3 z: 0.3 }}}
      inertia { x: 0.1 y: 0.1 z: 0.1 }
    }
    bodies {
      name: "ground"
      frozen: { all: true }
      colliders {
        heightMap {
          size: 10
          data: [2, 2, 2, 2, 2, 2, 2, 2, 2]
        }
      }
    }
    # Place the box in the first quadrant of the height map.
    defaults { qps { name: "box" pos: {x: 2 y: -2 z: 1.7}}}
  """

  def test_collision_position(self):
    """Tests collision occurs at the correct position on the heightmap."""
    config = text_format.Parse(BoxHeightMapColliderFnTest._CONFIG,
                               brax.Config())
    sys = brax.System(config)
    box_collider = brax.physics.colliders.BoxCorner([jp.take(config.bodies, 0)],
                                                    sys.body)
    box_collider = jp.take(box_collider, 0)
    heightmap_collider = brax.physics.colliders.HeightMap(
        [jp.take(config.bodies, 1)], sys.body)
    heightmap_collider = jp.take(heightmap_collider, 0)
    box_qp = jp.take(sys.default_qp(), 0)
    hm_qp = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.box_heightmap(box_collider,
                                                   heightmap_collider, box_qp,
                                                   hm_qp)
    self.assertSequenceAlmostEqual(contact.pos, [2 - 0.3, -2 - 0.3, 1.7 - 0.3],
                                   3)
    self.assertGreater(contact.penetration, 0.0)


if __name__ == '__main__':
  absltest.main()
