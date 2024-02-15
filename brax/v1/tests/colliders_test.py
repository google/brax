# Copyright 2024 The Brax Authors.
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
import brax.v1 as brax
from brax.v1 import jumpy as jp
from google.protobuf import text_format


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
    capsules = brax.physics.geometry.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)

    self.assertGreaterEqual(contact.penetration[0], 0)

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
    capsules = brax.physics.geometry.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)
    self.assertAlmostEqual(contact.penetration[0], 0.05)
    self.assertSequenceAlmostEqual(contact.pos[0], [0., 0., (0.15 + 0.2) / 2.0],
                                   4)

  def test_collision_with_zero_capsule_length(self):
    """Tests collisions with capsules that have zero length (i.e. spheres)."""
    config = text_format.Parse(CapsuleCapsuleColliderFnTest._CONFIG2,
                               brax.Config())
    sys = brax.System(config)
    capsules = brax.physics.geometry.Capsule(config.bodies, sys.body)
    capsule1 = jp.take(capsules, 0)
    capsule2 = jp.take(capsules, 1)
    # Set capsule ends such that the capsule lengths are zero.
    capsule1.end = jp.array([0., 0., 0.])
    capsule2.end = jp.array([0., 0., 0.])
    qp1 = jp.take(sys.default_qp(), 0)
    qp2 = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.capsule_capsule(capsule1, capsule2, qp1,
                                                     qp2)
    self.assertAlmostEqual(contact.penetration[0], 0.05)
    self.assertSequenceAlmostEqual(contact.pos[0], [0., 0., (0.15 + 0.2) / 2.0],
                                   4)


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
    box_collider = brax.physics.geometry.Box([jp.take(config.bodies, 0)],
                                             sys.body)
    box_collider = jp.take(box_collider, 0)
    heightmap_collider = brax.physics.geometry.HeightMap(
        [jp.take(config.bodies, 1)], sys.body)
    heightmap_collider = jp.take(heightmap_collider, 0)
    box_qp = jp.take(sys.default_qp(), 0)
    hm_qp = jp.take(sys.default_qp(), 1)

    contact = brax.physics.colliders.box_heightmap(box_collider,
                                                   heightmap_collider, box_qp,
                                                   hm_qp)
    self.assertSequenceAlmostEqual(contact.pos[0],
                                   [2 - 0.3, -2 - 0.3, 1.7 - 0.3], 3)
    self.assertGreater(contact.penetration[0], 0.0)


class CapsuleMeshTest(absltest.TestCase):
  """Tests the capsule-mesh collision function."""
  _CONFIG = """
    dt: .01 substeps: 4 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Capsule1" mass: 1
      colliders { capsule { radius: 0.5 length: 3.14 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Mesh" mass: 1
      colliders { mesh { name: "Mesh1" scale: 1 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    mesh_geometries {
      name: "Mesh1"
      vertices { x: 0.47 y: -0.68 z: 0.24 }
      vertices { x: -1.70 y: 0.75 z: -1.53 }
      vertices { x: 0.01 y: -0.12 z: -0.81 }
      face_normals { x: 0.51 y: 1.45 z: 0.55 }
      faces: [0, 1, 2]
    }
    defaults {
      qps { name: "Capsule1" pos {x: 0} rot { x: 113 y: 74 z: 152 } }
      qps { name: "Mesh" pos {x: 0} }
    }
  """

  def test_collision_position(self):
    """Tests collision occurs between capsule and triangle."""
    config = text_format.Parse(CapsuleMeshTest._CONFIG, brax.Config())
    sys = brax.System(config)

    capsules = brax.physics.geometry.Capsule(config.bodies, sys.body)
    mesh_geoms = {
        mesh_geom.name: mesh_geom for mesh_geom in config.mesh_geometries
    }
    mesh = brax.physics.geometry.Mesh(
        jp.array([jp.take(config.bodies, 1)]), sys.body, mesh_geoms)
    capsule1 = jp.take(capsules, 0)
    mesh = jp.take(mesh, 0)
    qp = sys.default_qp(0)
    contact = brax.physics.colliders.capsule_mesh(capsule1,
                                                  mesh, jp.take(qp, 0),
                                                  jp.take(qp, 1))

    self.assertGreater(contact.penetration[0], 0.0)


class MultiMeshTest(absltest.TestCase):
  """Tests scene with multiple meshes."""
  _CONFIG = """
    dt: .01 substeps: 4 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Capsule1" mass: 1
      colliders { capsule { radius: 0.5 length: 3.14 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Mesh_1" mass: 1
      colliders { mesh { name: "Mesh1" scale: 1 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    mesh_geometries {
      name: "Mesh1"
      vertices { x: 0.47 y: -0.68 z: 0.24 }
      vertices { x: -1.70 y: 0.75 z: -1.53 }
      vertices { x: 0.01 y: -0.12 z: -0.81 }
      face_normals { x: 0.51 y: 1.45 z: 0.55 }
      faces: [0, 1, 2]
    }
    bodies {
      name: "Mesh_2" mass: 1
      colliders { mesh { name: "Mesh2" scale: 1 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    mesh_geometries {
      name: "Mesh2"
      vertices { x: 0.47 y: -0.68 z: 0.24 }
      vertices { x: -1.70 y: 0.75 z: -1.53 }
      vertices { x: 0.01 y: -0.12 z: -0.81 }
      face_normals { x: 0.51 y: 1.45 z: 0.55 }
      faces: [0, 1, 2]
    }
    defaults {
      qps { name: "Capsule1" pos {x: 0} rot { x: 113 y: 74 z: 152 } }
      qps { name: "Mesh_1" pos {x: 0} }
      qps { name: "Mesh_2" pos {x: 3} }
    }
  """

  def test_mesh_collider_construction(self):
    """Tests collision occurs between capsule and triangle."""
    config = text_format.Parse(MultiMeshTest._CONFIG, brax.Config())
    sys = brax.System(config)
    # builds two collision primitives for each different mesh
    self.assertLen(sys.colliders, 2)
    # checks that collision bodies have correct body indices
    self.assertEqual(sys.colliders[0].cull.col_a.body.idx[0], 0)
    self.assertEqual(sys.colliders[0].cull.col_b.body.idx[0], 1)
    self.assertEqual(sys.colliders[1].cull.col_a.body.idx[0], 0)
    self.assertEqual(sys.colliders[1].cull.col_b.body.idx[0], 2)


if __name__ == '__main__':
  absltest.main()
