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

"""Tests for brax.physics."""

import copy
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import brax.v1 as brax
from brax.v1 import jumpy as jp
import jax

from google.protobuf import text_format


class BodyTest(absltest.TestCase):

  def test_projectile_motion(self):
    """A ball with an initial velocity curves down due to gravity."""
    sys = brax.System(
        text_format.Parse(
            """
    dt: 1 substeps: 1000
    gravity { z: -9.8 }
    bodies { name: "Ball" mass: 1 }
    defaults { qps { name: "Ball" vel {x: 1}}}
    dynamics_mode: "legacy_spring"
    """, brax.Config()))
    qp = sys.default_qp()
    qp, _ = sys.step(qp, jp.array([]))
    # v = v_0 + a * t
    self.assertAlmostEqual(qp.vel[0, 2], -9.8, 2)
    # x = x_0 + v_0 * t + 0.5 * a * t^2
    self.assertAlmostEqual(qp.pos[0, 0], 1, 2)
    self.assertAlmostEqual(qp.pos[0, 2], -9.8 / 2, 2)


class BoxTest(absltest.TestCase):

  _CONFIG = """
    dt: 1.5 substeps: 1000 friction: 0.77459666924 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults { qps { name: "box" pos { z: 1 }}}
    defaults { qps { name: "box" pos { z: 2 } vel {x: 2}}}
    dynamics_mode: "legacy_spring"
  """

  def test_box_hits_ground(self):
    """A box falls onto the ground and stjp."""
    sys = brax.System(text_format.Parse(BoxTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, _ = sys.step(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)

  def test_box_slide(self):
    """A box slides across the ground and comes to a stop."""
    sys = brax.System(text_format.Parse(BoxTest._CONFIG, brax.Config()))
    qp = sys.default_qp(1)
    qp, _ = sys.step(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)
    self.assertGreater(qp.pos[0, 0], 1)  # after sliding for a bit...
    self.assertAlmostEqual(qp.vel[0, 0], 0, 2)  # friction brings it to a stop
    self.assertLess(qp.pos[0, 0], 1.5)  # ... and keeps it from travelling 2m


class BoxBoxTest(absltest.TestCase):
  _CONFIG = """
    dt: 0.5 substeps: 200 friction: 0.8 elasticity: 0.5
    gravity { z: -9.8 }
    bodies {
      name: "box1" mass: 1
      colliders { box { halfsize { x: 0.2 y: 0.2 z: 0.2 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "box2" mass: 1
      colliders { box { halfsize { x: 0.1 y: 0.1 z: 0.1 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps { name: "box1" pos { x: 0 y: 1 z: .2 } rot {z: 0} }
      qps { name: "box2" pos { x: 0.1 y: 1 z: .6 } rot {z: 45} }
    }
    dynamics_mode: "legacy_spring"
  """

  def test_box_box(self):
    """A box falls onto another box."""
    sys = brax.System(text_format.Parse(BoxBoxTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    qp, _ = sys.step(qp, jp.array([]))
    # Boxes are stacked.
    self.assertAlmostEqual(qp.pos[0, 2], 0.2, delta=0.03)
    self.assertAlmostEqual(qp.pos[1, 2], 0.5, delta=0.03)
    # x-y position of the top box is unchanged.
    self.assertAlmostEqual(qp.pos[1, 0], 0.1, delta=0.03)
    self.assertAlmostEqual(qp.pos[1, 1], 1.0, delta=0.03)


class CollisionDebuggerTest(absltest.TestCase):
  _CONFIG = """
    dt: 0.01 substeps: 4 friction: 1
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults { qps { name: "box" pos { z: 0.49 }}}
    dynamics_mode: "legacy_spring"
  """

  def test_system_runs_with_debug_on(self):
    """Tests that the simulation runs with debug_contacts."""
    sys = brax.System(
        text_format.Parse(CollisionDebuggerTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, info = sys.step(qp, jp.array([]))
    self.assertGreater(info.contact_pos.shape[0], 0)


class BoxCapsuleTest(absltest.TestCase):

  _CONFIG = """
    dt: 0.05 substeps: 20 friction: 1 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "capsule" mass: 1
      colliders { capsule { length: 2 radius: 0.2 } }
      inertia { x: 1 y: 1 z: 1 }
    }

    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps { name: "box" pos { z: 2 }}
      qps { name: "capsule" pos: { z: 0.2 } rot: { y: 90 } }
    }
    dynamics_mode: "legacy_spring"
  """

  def test_box_hits_capsule(self):
    """A box falls onto a capsule and stays above it."""
    sys = brax.System(text_format.Parse(BoxCapsuleTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    self.assertAlmostEqual(qp.pos[0, 2], 2, 2)

    step = jax.jit(sys.step)
    for _ in range(50):
      qp, _ = step(qp, jp.array([]))
    # Box should be on the capsule, rather than on the ground.
    self.assertAlmostEqual(qp.pos[0, 2], 0.9, 2)


class HeightMapTest(absltest.TestCase):

  _CONFIG = """
    dt: 2 substeps: 1000 friction: 1 baumgarte_erp: 0.1 elasticity: 0
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
          data: [0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
      }
    }
    defaults { qps { name: "box" pos: {x: 5 y: 5 z: 1}}}
    dynamics_mode: "legacy_spring"
  """

  def test_box_stays_on_heightmap(self):
    """A box falls onto the height map and stjp."""
    sys = brax.System(text_format.Parse(HeightMapTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.3, 2)


class SphereTest(absltest.TestCase):

  _CONFIG = """
    dt: 5 substeps: 50 friction: 0.6 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "Sphere1" mass: 1
      colliders { sphere { radius: 0.25}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {qps { name: "Sphere1" pos {z: 1}}}
    defaults {qps { name: "Sphere1" pos {z: 1} vel {x: 2}}}
    dynamics_mode: "legacy_spring"
  """

  def test_sphere_hits_ground(self):
    """A sphere falls onto the ground and stops."""
    sys = brax.System(text_format.Parse(SphereTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, _ = sys.step(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.25, 2)

  def test_sphere_roll(self):
    """A sphere rolls across the ground."""
    sys = brax.System(text_format.Parse(SphereTest._CONFIG, brax.Config()))
    qp = sys.default_qp(1)
    qp, _ = sys.step(qp, jp.array([]))
    self.assertGreater(qp.ang[0, 1], 0.25)  # sphere is rolling


class CapsuleTest(absltest.TestCase):

  _CONFIG = """
    dt: 20.0 substeps: 10000 friction: 0.6 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "Capsule1" mass: 1
      colliders { capsule { radius: 0.25 length: 1.0 }}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule2" mass: 1
      colliders { rotation { y: 90 } capsule { radius: 0.25 length: 1.0 }}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule3" mass: 1
      colliders { rotation { y: 45 } capsule { radius: 0.25 length: 1.0 }}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule4" mass: 1
      colliders { rotation { x: 45 } capsule { radius: 0.25 length: 1.0 }}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps { name: "Capsule1" pos {z: 1}}
      qps { name: "Capsule2" pos {x:1 z: 1}}
      qps { name: "Capsule3" pos {x:3 z: 1}}
      qps { name: "Capsule4" pos {x:5 z: 1}}
    }
    defaults {
      qps { name: "Capsule1" pos {z: 1}}
      qps { name: "Capsule2" pos {z: 2}}
      qps { name: "Capsule3" pos {x:3 z: 1}}
      qps { name: "Capsule4" pos {x:5 z: 1}}
    }
    dynamics_mode: "legacy_spring"
  """

  def test_capsule_hits_ground(self):
    """A capsule falls onto the ground and stops."""
    sys = brax.System(text_format.Parse(CapsuleTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 0.25, 2)  # lying on its side
    self.assertAlmostEqual(qp.pos[2, 2], 0.25, 2)  # rolls to side from y rot
    self.assertAlmostEqual(qp.pos[3, 2], 0.25, 2)  # rolls to side from x rot

  def test_capsule_hits_capsule(self):
    """A capsule falls onto another capsule and balances on it."""
    config = text_format.Parse(CapsuleTest._CONFIG, brax.Config())
    config.dt = 2.0
    config.substeps = 1000
    sys = brax.System(config)
    qp = sys.default_qp(1)
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 1.25, 2)  # lying on Capsule1

  def test_cull(self):
    """A capsule falls onto another capsule, with NN culling."""
    config = text_format.Parse(CapsuleTest._CONFIG, brax.Config())
    config.collider_cutoff = 1
    config.dt = 2.0
    config.substeps = 1000
    sys = brax.System(config)
    qp = sys.default_qp(1)
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 1.25, 2)  # lying on Capsule1


class MeshTest(absltest.TestCase):

  _CONFIG = """
    dt: 0.05 substeps: 10 friction: 1.0 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "Mesh" mass: 1
      colliders { mesh { name: "Cylinder" scale: 0.1 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Capsule" mass: 1
      colliders { capsule { length: 2 radius: 0.2 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {} } }
    defaults {
      # Initial position is high up in the air.
      qps { name: "Mesh" pos: {x: 0 y: 0 z: 1.5} }

      # Capsule is away from the cylinder.
      qps { name: "Capsule" pos: {x: 10 y: 0 z: 0.2} rot: {x: 0 y: 90 z: 0} }
    }
    mesh_geometries {
      name: "Cylinder"
      path: "cylinder.stl"
    }
    dynamics_mode: "legacy_spring"
  """

  def test_mesh_hits_ground(self):
    """A mesh falls onto the ground."""
    config = text_format.Parse(MeshTest._CONFIG, brax.Config())
    resource_path = os.path.join(
        absltest.get_default_test_srcdir(),
        'brax/v1/tests/testdata')
    sys = brax.System(config, [resource_path])
    qp = sys.default_qp()
    # Cylinder should be up in the air.
    self.assertAlmostEqual(qp.pos[0, 2], 1.5, 2)

    step = jax.jit(sys.step)
    for _ in range(30):
      qp, _ = step(qp, jp.array([]))
    # Cylinder should be on the ground.
    self.assertAlmostEqual(qp.pos[0, 2], 0, 2)

  def test_mesh_hits_capsule(self):
    config = text_format.Parse(MeshTest._CONFIG, brax.Config())
    # Move the capsule under the cylinder.
    config.defaults[0].qps[1].pos.x = 0
    resource_path = os.path.join(
        absltest.get_default_test_srcdir(),
        'brax/v1/tests/testdata')
    sys = brax.System(config, [resource_path])
    qp = sys.default_qp()
    self.assertAlmostEqual(qp.pos[0, 2], 1.5, 2)

    step = jax.jit(sys.step)
    for _ in range(30):
      qp, _ = step(qp, jp.array([]))
    # Cylinder should be on the capsule, rather than on the ground.
    self.assertAlmostEqual(qp.pos[0, 2], 0.38, 2)


class CapsuleClippedPlaneTest(absltest.TestCase):
  """Tests the capsule-clippedPlane collision function."""
  _CONFIG = """
    dt: 2 substeps: 1000 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Sphere1" mass: 1
      colliders { sphere { radius: 0.5 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Sphere2" mass: 1
      colliders { sphere { radius: 0.5 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Sphere3" mass: 1
      colliders { sphere { radius: 0.5 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "ClippedPlane" mass: 1
      colliders {
        clipped_plane { halfsize_x: 3 halfsize_y: 1 }
        position { z: 2 }
      }
      frozen { all: true }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps { name: "Sphere1" pos { z: 3 } }
      qps { name: "Sphere2" pos { z: 3 x: -4 } }
      qps { name: "Sphere3" pos { z: 3 y: -2 } }
      qps { name: "ClippedPlane" pos { x: 0 } }
    }
    dynamics_mode: "legacy_spring"
  """

  def test_collision(self):
    """Tests collisions between spheres and a clipped plane."""
    config = text_format.Parse(CapsuleClippedPlaneTest._CONFIG, brax.Config())
    sys = brax.System(config)

    qp = sys.default_qp()
    step_fn = jax.jit(sys.step)
    qp, _ = step_fn(qp, jp.array([]))
    # sphere1 is on the clipped plane and sphere2/sphere3 are on the ground
    # plane
    self.assertAlmostEqual(qp.pos[0, 2], 2.5, 2)
    self.assertAlmostEqual(qp.pos[1, 2], 0.5, 2)
    self.assertAlmostEqual(qp.pos[2, 2], 0.5, 2)


class JointTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 100000
    dt: .01
    gravity { z: -9.8 }
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }}
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 10000
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
    }
    dynamics_mode: "legacy_spring"
  """

  @parameterized.parameters((2.0, 0.125, 0.0625), (5.0, 0.125, 0.03125),
                            (1.0, 0.0625, 0.1))
  def test_pendulum_period(self, mass, radius, vel):
    """A small spherical mass swings for approximately one period."""
    config = text_format.Parse(JointTest._CONFIG, brax.Config())

    # this length of time comes from the following:
    # inertia_about_anchor = mass * dist_to_anchor^2 + inertia_cm
    # dist_to_anchor = 1.0
    # inertia_cm = 2/5 * mass * radius^2 (solid sphere)
    # T = 2 * pi * sqrt(inertia_about_anchor / (2 * mass * g * dist_to_anchor))
    config.dt = 2 * jp.pi * jp.sqrt((.4 * radius**2 + 1.) / 9.8)
    config.bodies[1].mass = mass
    config.bodies[1].inertia.x = .4 * mass * radius**2
    config.bodies[1].inertia.y = .4 * mass * radius**2
    config.bodies[1].inertia.z = .4 * mass * radius**2
    sys = brax.System(config)

    # initializing system to have a small initial velocity and ang velocity
    # so that small angle approximation is valid
    qp = brax.QP(
        pos=jp.array([[0., 0., 0.], [0., 0., -1.]]),
        rot=jp.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        vel=jp.array([[0., 0., 0.], [0., vel, 0.]]),
        ang=jp.array([[0., 0., 0.], [vel, 0., 0.]]))
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[1, 1], 0., 3)  # returned to the origin

  offsets = [-15, 15, -45, 45, -75, 75]
  axes = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
  ]
  limits = [0, 1]

  @parameterized.parameters(itertools.product(offsets, axes, limits))
  def test_reference_offset(self, offset, axis, limit):
    """Construct joint and check that default_qp generates offsets correctly."""
    config = text_format.Parse(JointTest._CONFIG, brax.Config())

    # loop over different types of joints
    for l in range(3):
      # construct appropriate number of angle_limits for this joint
      if l == 0:
        a_l = config.joints[0].angle_limit[0]
      else:
        a_l = config.joints[0].angle_limit.add()
      # set angle default to either 0 or the offset value
      a_l.min = offset * limit
      a_l.max = offset * limit

      sys_default = brax.System(config)

      this_offset = offset * jp.array(axis)

      # duplicate config deeply
      rotated_config = copy.deepcopy(config)

      rotated_config.joints[0].reference_rotation.x = this_offset[0]
      rotated_config.joints[0].reference_rotation.y = this_offset[1]
      rotated_config.joints[0].reference_rotation.z = this_offset[2]

      # construct a new config with this reference offset applied
      sys_offset = brax.System(rotated_config)
      offset_qp = sys_offset.default_qp()

      # construct joint functions for default and offset systems
      qp_p = jp.take(offset_qp, 0)
      qp_c = jp.take(offset_qp, 1)
      joint_offset = jp.take(sys_offset.joints[0], 0)
      joint_default = jp.take(sys_default.joints[0], 0)

      # calculate joint angles as seen by the default or offset system
      _, angle_offset = joint_offset.axis_angle(qp_p, qp_c)
      _, angle_default = joint_default.axis_angle(qp_p, qp_c)
      angle_offset = (jp.array(angle_offset) / jp.pi) * 180
      angle_default = (jp.array(angle_default) / jp.pi) * 180
      num_offsets = angle_offset.shape[0]

      for a_o, a_d, t_o in zip(angle_offset, angle_default,
                               this_offset[:num_offsets]):
        if limit == 0:
          # default system sees part rotated by offset degrees
          self.assertAlmostEqual(a_d, t_o, 2)
          # offset system sees part at local 0.0
          self.assertAlmostEqual(a_o, 0.0, 2)
        if limit == 1:
          # when default angle is nonzero, there's not a clean relationship
          # between the default and offset frames anymore, because the offset
          # frame now has two rotations applied---1) the rotation that offsets
          # it relative to the parent, and then 2) the rotation placing it at
          # its default angle limit.  it should still be the case, though, that
          # the offset angle agrees with the default angle limit.
          self.assertAlmostEqual(a_o, offset, 2)


class Actuator1DTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 80
    dt: 4.0
    gravity { z: -9.8 }
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }}
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 5000
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
      angular_damping: 20.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 150.0
      angle {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
    dynamics_mode: "legacy_spring"
  """

  @parameterized.parameters(15., 30., 45., 90.)
  def test_1d_angle_actuator(self, target_angle):
    """A simple part actuates to a target angle."""
    config = text_format.Parse(Actuator1DTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = sys.step(qp, jp.array([target_angle]))
    qp_p = jp.take(qp, 0)
    qp_c = jp.take(qp, 1)
    joint = jp.take(sys.joints[0], 0)
    _, (angle,) = joint.axis_angle(qp_p, qp_c)

    self.assertAlmostEqual(target_angle * jp.pi / 180, angle, 2)


class Actuator2DTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 2000
    dt: 2.0
    gravity { z: -9.8 }
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }
    }
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 10000
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
      angle_limit { min: -180 max: 180 }
      angular_damping: 200.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 2000.0
      angle {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
    dynamics_mode: "legacy_spring"
  """

  @parameterized.parameters((15., 30.), (-45., 80), (120, -60.), (-35., -52.))
  def test_2d_angle_actuator(self, target_angle_1, target_angle_2):
    """A simple part actuates 2d-angle actuator to two target angles."""
    config = text_format.Parse(Actuator2DTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = jax.jit(sys.step)(qp, jp.array([target_angle_1, target_angle_2]))
    qp_p = jp.take(qp, 0)
    qp_c = jp.take(qp, 1)
    joint = jp.take(sys.joints[0], 0)
    _, angles = joint.axis_angle(qp_p, qp_c)
    self.assertAlmostEqual(target_angle_1 * jp.pi / 180, angles[0], 2)
    self.assertAlmostEqual(target_angle_2 * jp.pi / 180, angles[1], 2)


class Actuator3DTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 8000
    dt: 20
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }
      colliders { capsule { radius: 0.5 length: 2.0 } }
    }
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 10000
      child_offset { z: 1 }
      angle_limit {
        min: -100
        max: 100
      }
      angle_limit {
        min: -100
        max: 100
      }
      angle_limit {
        min: -100
        max: 100
      }
      angular_damping: 180.0
      limit_strength: 2000.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 40.0
      torque {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
    dynamics_mode: "legacy_spring"
  """
  torques = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]

  @parameterized.parameters(((15, 15, 15), torques), ((35, 40, 75), torques),
                            ((80, 45, 30), torques))
  def test_3d_torque_actuator(self, limits, torques):
    """A simple part actuates 3d-torque actuator to its limits."""
    config = text_format.Parse(Actuator3DTest._CONFIG, brax.Config())
    for t in torques:
      for angle_limit, limit in zip(config.joints[0].angle_limit, limits):
        angle_limit.min = -limit
        angle_limit.max = limit

      sys = brax.System(config=config)
      qp = sys.default_qp()
      qp, _ = jax.jit(sys.step)(qp, jp.array(t))
      qp_p = jp.take(qp, 0)
      qp_c = jp.take(qp, 1)
      joint = jp.take(sys.joints[0], 0)
      _, angles = joint.axis_angle(qp_p, qp_c)
      angles = [a * 180 / jp.pi for a in angles]
      for angle, limit, torque in zip(angles, limits, t):
        if torque != 0:
          self.assertAlmostEqual(angle, limit, 1)  # actuated to target angle


if __name__ == '__main__':
  absltest.main()
