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

"""Tests for brax.physics."""

import copy
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import brax
from brax import jumpy as jp
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
    dt: 1.5 substeps: 2000 friction: 0.77459666924
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      colliders { box { halfsize { x: 1 y: 1 z: 1 }} no_contact: true }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults { qps { name: "box" pos { z: 1 }}}
    defaults { qps { name: "box" pos { z: 2 } vel {x: 2}}}
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


class BoxCapsuleTest(absltest.TestCase):

  _CONFIG = """
    dt: 0.05 substeps: 30 friction: 1
    gravity { z: -9.8 }
    bodies {
      name: "box1" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "capsule1" mass: 1
      colliders { capsule { length: 2 radius: 0.2 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "box2" mass: 10
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "capsule2" mass: 1
      colliders { capsule { length: 2 radius: 0.2 } }
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "box3"
      colliders {
        box {
          halfsize: { x: 0.5 y: 0.5 z: 0.5 }
        }
      }
      mass: 1.0
      frozen { all: true }
    }
    bodies {
      name: "capsule3"
      colliders {
        capsule { radius: 0.5 length: 1.0 }
      }
      inertia { x: 1.0 y: 1.0 z: 1.0 }
      mass: 1.0
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {
      qps {
        name: "capsule1"
        pos { x: 8 y: 0 z: 1 }
      }
      qps {
        name: "box1"
        pos { x: 8 y: 0 z: 3.5 }
      }
      qps {
        name: "capsule2"
        pos { x: 2 y: 0 z: 1 }
      }
      qps {
        name: "box2"
        pos { x: 2 y: 0 z: 3.5 }
      }
      qps { name: "capsule3" pos { x: 0 y: 0 z: 3 } }
    }
    solver_scale_collide: .3
  """

  def test_box_hits_capsule(self):
    """A box falls onto a capsule and stays above it."""
    sys = brax.System(text_format.Parse(BoxCapsuleTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    self.assertAlmostEqual(qp.pos[0, 2], 3.5, 2)
    self.assertAlmostEqual(qp.pos[2, 2], 3.5, 2)

    step = jax.jit(sys.step)
    for _ in range(50):
      qp, _ = step(qp, jp.array([]))
    # Box should be on the capsule, rather than on the ground, for both masses
    # box falls on capsule
    self.assertAlmostEqual(qp.pos[0, 2], 2.5, 2)
    self.assertAlmostEqual(qp.pos[1, 2], 1.0, 2)
    # box falls on capsule with non-unit mass ratio
    self.assertAlmostEqual(qp.pos[2, 2], 2.5, 2)
    self.assertAlmostEqual(qp.pos[3, 2], 1.0, 2)
    # capsule falls on frozen box
    self.assertAlmostEqual(qp.pos[5, 2], 1.5, 2)


class HeightMapTest(absltest.TestCase):

  _CONFIG = """
    dt: 2 substeps: 1000 friction: 1 elasticity: 0
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
  """

  def test_box_stays_on_heightmap(self):
    """A box falls onto the height map and stjp."""
    sys = brax.System(text_format.Parse(HeightMapTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.3, 2)


class SphereTest(absltest.TestCase):

  _CONFIG = """
    dt: 5 substeps: 500 friction: 0.6
    gravity { z: -9.8 }
    bodies {
      name: "Sphere1" mass: 1
      colliders { sphere { radius: 0.25}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {qps { name: "Sphere1" pos {z: 1}}}
    defaults {qps { name: "Sphere1" pos {z: 1} vel {x: 2}}}
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
    dt: 20.0 substeps: 10000 friction: 0.6
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
    config.substeps = 400
    sys = brax.System(config, brax.Config())
    qp = sys.default_qp(1)
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 1.25, 2)  # lying on Capsule1

  def test_cull(self):
    """A capsule falls onto another capsule, with NN culling."""
    config = text_format.Parse(CapsuleTest._CONFIG, brax.Config())
    config.collider_cutoff = 1
    config.dt = 2.0
    config.substeps = 400
    sys = brax.System(config)
    qp = sys.default_qp(1)
    qp, _ = jax.jit(sys.step)(qp, jp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 1.25, 2)  # lying on Capsule1


class MeshTest(absltest.TestCase):

  _CONFIG = """
    dt: 0.05 substeps: 10 friction: 1.0
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
  """

  def test_mesh_hits_ground(self):
    """A mesh falls onto the ground."""
    config = text_format.Parse(MeshTest._CONFIG, brax.Config())
    resource_path = os.path.join(absltest.get_default_test_srcdir(),
                                 'brax/tests/testdata')
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
    resource_path = os.path.join(absltest.get_default_test_srcdir(),
                                 'brax/tests/testdata')
    sys = brax.System(config, [resource_path])
    qp = sys.default_qp()
    self.assertAlmostEqual(qp.pos[0, 2], 1.5, 2)

    step = jax.jit(sys.step)
    for _ in range(30):
      qp, _ = step(qp, jp.array([]))
    # Cylinder should be on the capsule, rather than on the ground.
    self.assertAlmostEqual(qp.pos[0, 2], 0.394, 2)


class JointTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 4000
    dt: .01
    gravity { z: -9.8 }
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }}
    joints {
      name: "Joint" parent: "Anchor" child: "Bob"
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
    }
    solver_scale_pos: .2
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
      num_offsets = l + 1

      for a_o, a_d, t_o in zip(angle_offset[:num_offsets],
                               angle_default[:num_offsets],
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
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }}
    joints {
      name: "Joint" parent: "Anchor" child: "Bob"
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
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies {
      name: "Bob" mass: 1 inertia { x: 1 y: 1 z: 1 }
    }
    joints {
      name: "Joint" parent: "Anchor" child: "Bob"
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
      angle_limit { min: -180 max: 180 }
      angular_damping: 20.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 200.0
      angle {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
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
      name: "Joint" parent: "Anchor" child: "Bob"
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
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 40.0
      torque {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
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


class ForceTest(parameterized.TestCase):

  _CONFIG = """
    dt: 0.1
    substeps: 5000
    bodies { name: "body" mass: 1 inertia { x: 1 y: 1 z: 1 }}
    forces {
      name: "thruster"
      body: "body"
      strength: 2.5
      thruster {}
    }
    forces {
      name: "twister"
      body: "body"
      strength: 2.5
      twister {}
    }
  """

  @parameterized.parameters(1, 5, 10)
  def test_thruster(self, force):
    """A simple part actuates to a target angle."""
    config = text_format.Parse(ForceTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = sys.step(qp, force * jp.array([1., 0., 0., 0., 0., 0]))

    self.assertAlmostEqual(qp.pos[0][0], 0.5 * 2.5 * force * 0.1**2, 3)

  @parameterized.parameters(1, 5, 10)
  def test_twister(self, torque):
    """A simple part actuates to a target angle."""
    config = text_format.Parse(ForceTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = sys.step(qp, torque * jp.array([0., 0., 0., 1., 0., 0]))

    self.assertAlmostEqual(qp.ang[0][0], 2.5 * torque * 0.1, 3)


class ElasticityTest(parameterized.TestCase):
  _CONFIG = """
  dt: 1. substeps: 1000 friction: 0.0 elasticity: 0.5
  gravity { z: -9.8 }
  bodies {
    name: "sphere" mass: 1
    colliders { capsule { radius: .5 length: 1.0 } }
    inertia { x: 1 y: 1 z: 1 }
    }
  bodies {
    name: "boxwall" mass: 1
    colliders { box { halfsize { x: 1 y: 1 z: 1}}}
    inertia { x: 1 y: 1 z: 1}
    frozen { all: true}
  }
  bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
  defaults { qps { name: "sphere" vel {x: 10}}
             qps { name: "boxwall" pos { x: 10 } }}
  """

  @parameterized.parameters(0, .5, 1.)
  def test_ball_bounce(self, elasticity):
    """A ball bounces off a wall where ball and wall have some elasticity."""
    config = text_format.Parse(ElasticityTest._CONFIG, brax.Config())
    config.elasticity = elasticity
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp_init = qp
    qp, _ = sys.step(qp, jp.array([]))

    self.assertAlmostEqual(qp_init.vel[0][0] * (-1) * (elasticity**2.),
                           qp.vel[0][0], 2)


if __name__ == '__main__':
  absltest.main()
