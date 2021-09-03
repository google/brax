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

"""Tests for brax.physics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import brax
from brax.physics.base import take
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
    qp, _ = sys.step(qp, jnp.array([]))
    # v = v_0 + a * t
    self.assertAlmostEqual(qp.vel[0, 2], -9.8, 2)
    # x = x_0 + v_0 * t + 0.5 * a * t^2
    self.assertAlmostEqual(qp.pos[0, 0], 1, 2)
    self.assertAlmostEqual(qp.pos[0, 2], -9.8 / 2, 2)


class BoxTest(absltest.TestCase):

  _CONFIG = """
    dt: 1.5 substeps: 1000 friction: 0.6 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "box" mass: 1
      colliders { box { halfsize { x: 0.5 y: 0.5 z: 0.5 }}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults { qps { name: "box" pos { z: 1 }}}
    defaults { qps { name: "box" pos { z: 2 } vel {x: 2}}}
  """

  def test_box_hits_ground(self):
    """A box falls onto the ground and stops."""
    sys = brax.System(text_format.Parse(BoxTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)

  def test_box_slide(self):
    """A box slides across the ground and comes to a stop."""
    sys = brax.System(text_format.Parse(BoxTest._CONFIG, brax.Config()))
    qp = sys.default_qp(1)
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)
    self.assertGreater(qp.pos[0, 0], 1)  # after sliding for a bit...
    self.assertAlmostEqual(qp.vel[0, 0], 0, 2)  # friction brings it to a stop
    self.assertLess(qp.pos[0, 0], 1.5)  # ... and keeps it from travelling 2m


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
  """

  def test_box_stays_on_heightMap (self):
    """A box falls onto the height map and stops."""
    sys = brax.System(text_format.Parse(HeightMapTest._CONFIG, brax.Config()))
    qp = sys.default_qp()
    qp, _ = sys.step(qp, [])
    self.assertAlmostEqual(qp.pos[0, 2], 0.3, 2)


class SphereTest(absltest.TestCase):

  _CONFIG = """
    dt: 5 substeps: 5000 friction: 0.6 baumgarte_erp: 0.1
    gravity { z: -9.8 }
    bodies {
      name: "Sphere1" mass: 1
      colliders { sphere { radius: 0.25}}
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Ground" frozen: { all: true } colliders { plane {}}}
    defaults {qps { name: "Sphere1" pos {z: 1}}}
  """

  def test_sphere_hits_ground(self):
    """A sphere falls onto the ground and stops."""
    sys = brax.System(text_format.Parse(SphereTest._CONFIG, brax.Config()))
    qp = sys.default_qp(0)
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.25, 2)


class CapsuleTest(absltest.TestCase):

  _CONFIG = """
    dt: 5 substeps: 5000 friction: 0.6 baumgarte_erp: 0.1
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
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 0.25, 2)  # lying on its side
    self.assertAlmostEqual(qp.pos[2, 2], 0.25, 2)  # rolls to side from y rot
    self.assertAlmostEqual(qp.pos[3, 2], 0.25, 2)  # rolls to side from x rot

  def test_capsule_hits_capsule(self):
    """A capsule falls onto another capsule and balances on it."""
    sys = brax.System(text_format.Parse(CapsuleTest._CONFIG, brax.Config()))
    qp = sys.default_qp(1)
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 2], 0.5, 2)  # standing up and down
    self.assertAlmostEqual(qp.pos[1, 2], 1.25, 2)  # lying on Capsule1


class JointTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 100000
    gravity { z: -9.8 }
    bodies {
      name: "Anchor" frozen: { all: true } mass: 1
      inertia { x: 1 y: 1 z: 1 }
    }
    bodies { name: "Bob" }
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 10000
      child_offset { z: 1 }
      angle_limit { min: -180 max: 180 }
    }
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
    config.dt = float(2 * jnp.pi * jnp.sqrt((.4 * radius**2 + 1.) / 9.8))
    config.bodies[1].mass = mass
    config.bodies[1].inertia.x = .4 * mass * radius**2
    config.bodies[1].inertia.y = .4 * mass * radius**2
    config.bodies[1].inertia.z = .4 * mass * radius**2
    sys = brax.System(config)

    # initializing system to have a small initial velocity and ang velocity
    # so that small angle approximation is valid
    qp = brax.QP(
        pos=jnp.array([[0., 0., -1.], [0., 0., 0.]]),
        rot=jnp.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        vel=jnp.array([[0., vel, 0.], [0., 0., 0.]]),
        ang=jnp.array([[.5 * vel, 0., 0.], [0., 0., 0.]]))
    qp, _ = sys.step(qp, jnp.array([]))
    self.assertAlmostEqual(qp.pos[0, 1], 0., 3)  # returned to the origin


class Actuator1DTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 1200
    dt: 4.0
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
      angular_damping: 140.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 15000.0
      angle {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
    defaults { angles { name: "Joint" angle: { x: 60 } } }
  """

  @parameterized.parameters(15., 30., 45., 90.)
  def test_1d_angle_actuator(self, target_angle):
    """A simple part actuates to a target angle."""
    config = text_format.Parse(Actuator1DTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = sys.step(qp, jnp.array([target_angle]))
    qp_p = take(qp, 0)
    qp_c = take(qp, 1)
    joint = take(sys.joint_revolute, 0)
    _, (angle,) = joint.axis_angle(qp_p, qp_c)

    self.assertAlmostEqual(target_angle * jnp.pi / 180, angle, 2)


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
      colliders { capsule { radius: 0.5 length: 2.0 }}
    }
    joints {
      name: "Joint" parent: "Anchor" child: "Bob" stiffness: 10000
      child_offset { z: 1 }
      angle_limit {
        min: -180
        max: 180
      }
      angle_limit {
        min: -180
        max: 180
      }
      angular_damping: 200.0
    }
    actuators {
      name: "Joint"
      joint: "Joint"
      strength: 2000.0
      angle {}
    }
    defaults { qps { name: "Anchor" pos {z: 2}} qps { name: "Bob" pos {z: 1}}}
  """

  @parameterized.parameters((15., 30.), (45., 90.5), (-120, 60.), (30., -120.),
                            (-150, -130), (130, 165))
  def test_2d_angle_actuator(self, target_angle_1, target_angle_2):
    """A simple part actuates 2d-angle actuator to two target angles."""
    config = text_format.Parse(Actuator2DTest._CONFIG, brax.Config())
    sys = brax.System(config=config)
    qp = sys.default_qp()
    qp, _ = sys.step(qp, jnp.array([target_angle_1, target_angle_2]))
    qp_p = take(qp, 0)
    qp_c = take(qp, 1)
    joint = take(sys.joint_universal, 0)
    _, angles = joint.axis_angle(qp_p, qp_c)

    self.assertAlmostEqual(target_angle_1 * jnp.pi / 180, angles[0], 2)
    self.assertAlmostEqual(target_angle_2 * jnp.pi / 180, angles[1], 2)


class Actuator3DTest(parameterized.TestCase):

  _CONFIG = """
    substeps: 8
    dt: .02
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
      angular_damping: 120.0
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

      # cuts down compile time for test to only compile a short step
      for _ in range(1000):
        qp, _ = sys.step(qp, jnp.array(t))

      qp_p = take(qp, 0)
      qp_c = take(qp, 1)
      joint = take(sys.joint_spherical, 0)
      _, angles = joint.axis_angle(qp_p, qp_c)
      angles = [a * 180 / jnp.pi for a in angles]
      for angle, limit, torque in zip(angles, limits, t):
        if torque != 0:
          self.assertAlmostEqual(angle, limit, 1)  # actuated to target angle


if __name__ == '__main__':
  absltest.main()
