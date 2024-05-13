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

# pylint:disable=g-multiple-import
"""Tests for spring physics pipeline."""

from absl.testing import absltest
from brax import com
from brax import kinematics
from brax import test_utils
from brax.base import Transform
from brax.generalized import pipeline as g_pipeline
from brax.spring import pipeline
import jax
from jax import numpy as jp
import numpy as np


class PipelineTest(absltest.TestCase):

  def test_pendulum(self):
    sys = test_utils.load_fixture('triple_pendulum.xml')
    sys = sys.tree_replace({'opt.timestep': 0.0001})
    sys = sys.replace(
        link=sys.link.replace(constraint_stiffness=jp.array([800_000.0] * 3))
    )

    state = pipeline.init(sys, sys.init_q, jp.zeros(sys.qd_size()))
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(10_000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
    x = state.x

    # compare against generalized step
    q, qd = jp.zeros(sys.q_size()), jp.zeros(sys.qd_size())
    state = g_pipeline.init(sys, q, qd)
    j_g_step = jax.jit(g_pipeline.step)
    j_forward = jax.jit(kinematics.forward)
    for _ in range(10_000):
      state = j_g_step(sys, state, jp.zeros(sys.act_size()))
    x_g, _ = j_forward(sys, state.q, state.qd)

    # trajectories should be close after 1 second of simulation
    self.assertLess(jp.linalg.norm(x_g.pos - x.pos), 2e-2)

  def test_universal_pendulum(self):
    sys = test_utils.load_fixture('single_universal_pendulum.xml')

    init_q = jp.array([0.1, 0.2])
    init_qd = jax.random.uniform(jax.random.PRNGKey(0), (2,)) / 10.0

    # set sensible springy defaults
    link = sys.link.replace(constraint_limit_stiffness=jp.array([0.0] * 2))
    link = link.replace(constraint_stiffness=jp.array([100_000.0] * 2))
    link = link.replace(constraint_ang_damping=jp.array([0.0] * 2))
    link = link.replace(constraint_vel_damping=jp.array([200.0] * 2))
    sys = sys.replace(link=link)
    sys = sys.replace(ang_damping=0.0)
    sys = sys.tree_replace({'opt.timestep': 0.001})
    sys = sys.replace(solver_iterations=500)

    state = pipeline.init(sys, init_q, init_qd)
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
    x = state.x

    # compare against generalized step
    state = g_pipeline.init(sys, init_q, init_qd)
    j_g_step = jax.jit(g_pipeline.step)
    j_forward = jax.jit(kinematics.forward)
    for _ in range(1000):
      state = j_g_step(sys, state, jp.zeros(sys.qd_size()))
    x_g, _ = j_forward(sys, state.q, state.qd)

    # trajectories should be close after 1 second of simulation
    self.assertLess(jp.linalg.norm(x_g.rot - x.rot), 1.51e-2)

  def test_spherical_pendulum(self):
    sys = test_utils.load_fixture('single_spherical_pendulum.xml')

    init_q = jp.array([0.1, 0.2, 0.3])
    init_qd = jax.random.uniform(jax.random.PRNGKey(0), (3,)) / 10.0

    # set sensible springy defaults
    link = sys.link.replace(constraint_limit_stiffness=jp.array([0.0] * 3))
    link = link.replace(constraint_stiffness=jp.array([100_000.0] * 3))
    link = link.replace(constraint_ang_damping=jp.array([0.0] * 3))
    link = link.replace(constraint_vel_damping=jp.array([200.0] * 3))
    sys = sys.replace(link=link)
    sys = sys.replace(ang_damping=0.0)
    sys = sys.tree_replace({'opt.timestep': 0.001})
    sys = sys.replace(solver_iterations=500)

    state = pipeline.init(sys, init_q, init_qd)
    # the qd calculation for pbd/spring doesn't match generalized, so we get xd
    # from generalized and plug it back into pbd
    # TODO: remove this xd override once kinematics.forward is fixed
    state_g = g_pipeline.init(sys, init_q, init_qd)
    off = state_g.x.pos - state_g.root_com
    xd = Transform.create(pos=off).vmap().do(state_g.cd)
    state = state.replace(xd=xd, xd_i=com.from_world(sys, state.x, xd)[1])

    j_spring_step = jax.jit(pipeline.step)
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
    x = state.x

    # compare against generalized step
    state = state_g
    j_g_step = jax.jit(g_pipeline.step)
    j_forward = jax.jit(kinematics.forward)
    for _ in range(1000):
      state = j_g_step(sys, state, jp.zeros(sys.act_size()))
    x_g, _ = j_forward(sys, state.q, state.qd)

    # trajectories should be close after 1 second of simulation
    self.assertLess(jp.linalg.norm(x_g.rot - x.rot), 1e-2)

  def test_prismatic_joint(self):
    # tests launching a mass along a slide joint in the [1,1,1] direction
    sys = test_utils.load_fixture('single_prismatic.xml')

    init_q = jp.array([0.9])
    init_qd = jp.array([1.0])

    # set sensible springy defaults
    link = sys.link.replace(constraint_limit_stiffness=jp.array([10_000.0]))
    link = link.replace(constraint_stiffness=jp.array([10_000.0]))
    link = link.replace(constraint_ang_damping=jp.array([0.0]))
    link = link.replace(constraint_vel_damping=jp.array([200.0]))
    sys = sys.replace(link=link)
    sys = sys.replace(ang_damping=0.0)
    sys = sys.tree_replace({'opt.timestep': 0.001})
    sys = sys.replace(solver_iterations=500)

    state = pipeline.init(sys, init_q, init_qd)
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
    x = state.x

    # compare against generalized step
    state = g_pipeline.init(sys, init_q, init_qd)
    j_g_step = jax.jit(g_pipeline.step)
    j_forward = jax.jit(kinematics.forward)
    for _ in range(1000):
      state = j_g_step(sys, state, jp.zeros(sys.qd_size()))
    x_g, _ = j_forward(sys, state.q, state.qd)

    # trajectories should be close after 1 second of simulation
    np.testing.assert_allclose(x_g.pos, x.pos, rtol=5e-3, atol=5e-3)

  def test_2d_sliding_joint(self):
    # tests launching a capsule at a wall with 2 sliding dofs
    sys = test_utils.load_fixture('double_prismatic.xml')
    sys = sys.replace(
        link=sys.link.replace(
            constraint_vel_damping=150 * jp.ones(sys.num_links())
        )
    )
    sys = sys.tree_replace({'opt.timestep': 0.001})

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(2.5)
    qd = qd.at[1].set(2.5)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    states = []
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
      states.append(state)
    x, xd = state.x, state.xd

    # capsule still constrained to the plane and not rotating
    self.assertAlmostEqual(x.pos[0, 2], 0.0, delta=1e-2)
    np.testing.assert_allclose(
        x.rot, jp.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-3
    )
    # capsule reflects off boundary and is traveling 2.5 m/s without rotating
    np.testing.assert_allclose(
        xd.vel[0], jp.array([-2.5, -2.5, 0.0]), atol=1e-3
    )
    np.testing.assert_allclose(xd.ang[0], jp.array([0.0, 0.0, 0.0]), atol=1e-7)

  def test_3d_sliding_joint(self):
    # tests launching a capsule at a wall with 3 sliding dofs
    sys = test_utils.load_fixture('triple_prismatic.xml')
    sys = sys.tree_replace({'opt.timestep': 0.001})

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(2.5)
    qd = qd.at[1].set(2.5)
    qd = qd.at[2].set(2.5)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    states = []
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
      states.append(state)
    x, xd = state.x, state.xd

    # capsule not rotating
    np.testing.assert_allclose(
        x.rot, jp.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-3
    )
    # capsule reflected off boundary and is traveling 2.5 m/s without rotating
    np.testing.assert_allclose(
        xd.vel[0], jp.array([-2.5, -2.5, -2.5]), atol=1e-3
    )
    np.testing.assert_allclose(xd.ang[0], jp.array([0.0, 0.0, 0.0]), atol=1e-7)

  def test_2d_prismaversal_joint(self):
    # tests a prismatic+universal 2dof joint sliding/rotating into its limits
    sys = test_utils.load_fixture('prismaversal_2dof_joint.xml')
    sys = sys.tree_replace({'opt.timestep': 0.001})

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(2.5)
    qd = qd.at[1].set(2.5)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    states = []
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
      states.append(state)

    # reflects off limits and is still traveling 2.5 m/s
    np.testing.assert_allclose(state.qd, jp.array([-2.5, -2.5]), atol=1e-3)
    np.testing.assert_allclose(
        state.xd.ang[0], jp.array([0.0, -2.5, 0.0]), atol=1e-4
    )

  def test_3d_prismaversal_joint(self):
    # tests a prismatic+spherical 3dof joint sliding/rotating into its limits
    sys = test_utils.load_fixture('prismaversal_3dof_joint.xml')
    sys = sys.tree_replace({'opt.timestep': 0.001})

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(2.5)
    qd = qd.at[1].set(2.5)
    qd = qd.at[2].set(2.5)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    states = []
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
      states.append(state)

    # reflects off limits and is still traveling 2.5 m/s
    np.testing.assert_allclose(
        state.qd, jp.array([-2.5, -2.5, -2.5]), atol=1e-3
    )
    np.testing.assert_allclose(
        state.xd.ang[0], jp.array([0.0, -2.5, 0.0]), atol=1e-4
    )

  def test_sliding_capsule(self):
    sys = test_utils.load_fixture('capsule.xml')
    sys = sys.tree_replace({'opt.timestep': 0.001})

    qd = jp.zeros(sys.qd_size())
    qd = qd.at[0].set(5.0)

    state = pipeline.init(sys, sys.init_q, qd)
    j_spring_step = jax.jit(pipeline.step)
    for _ in range(1000):
      state = j_spring_step(sys, state, jp.zeros(sys.act_size()))
    x, xd = state.x, state.xd

    # capsule slides to a stop
    self.assertAlmostEqual(x.pos[0, 2], 0.25, delta=1e-3)
    np.testing.assert_allclose(
        x.rot, jp.array([[1.0, 0.0, 0.0, 0.0]]), atol=1e-3
    )
    np.testing.assert_allclose(xd.vel, jp.zeros_like(xd.vel), atol=1e-3)
    np.testing.assert_allclose(xd.ang, jp.zeros_like(xd.ang), atol=1e-3)


if __name__ == '__main__':
  absltest.main()
