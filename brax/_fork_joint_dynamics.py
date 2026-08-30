# explore_bench fork of brax: joint-space dynamics the positional pipeline was
# missing.
"""Armature, passive joint springs/dampers, and implicit velocity damping.

`brax.positional` is a maximal-coordinate PBD engine, and three quantities that
MuJoCo carries per DOF had no path into it at all -- `dof_armature`,
`dof_damping` and `jnt_stiffness` are read by `brax/io/mjcf.py` into
`System.dof` and then used only by `brax.generalized`.  For the dynmanip 3D
scenes that is not a small omission:

* the UR5e/Panda/Spot joints carry `armature` up to 0.1 against link inertias
  as small as 1e-6, so the rotor inertia is up to five orders of magnitude
  larger than the link's own and is what actually sets the joint's response;
* the Robotiq finger joints carry the only damping they have in `dof_damping`.

The fourth problem is the integrator.  All four models are authored for
`mjINT_IMPLICITFAST`, which integrates every velocity-proportional force
implicitly.  brax integrates them explicitly, and an explicit Euler step is
unconditionally unstable once `c*dt/M > 2` for a damping coefficient `c`.
Measured on these models, Spot's `arm_f1x` position servo sits at
`kd*dt/M_ii = 98`; MuJoCo itself, forced onto explicit Euler, diverges to
|qd| = 2585 rad/s where implicitfast stays at 19.

`velocity_relaxation` applies MuJoCo's treatment in the one form that survives
a maximal-coordinate engine: a per-DOF (Jacobi/diagonal) implicit solve,
`qd_used = qd / (1 + dt * c_ii / m_ii)`.  This is exact for a decoupled DOF and
is unconditionally stable for any `m_ii <= M_ii`, which is why
`io/mjcf.py::_dof_inertia_bound` takes a MINIMUM of `diag(M)` over sampled
configurations rather than a single-pose value: an underestimate of the
inertia can only shrink the applied damping, never overshoot it.  Where the
explicit scheme was already stable (`c*dt/M` small) the relaxation factor is
~1 and the force is MuJoCo's unchanged.
"""

from brax import actuator
from brax import scan
import jax
from jax import numpy as jp


def velocity_relaxation(sys, qd: jax.Array) -> jax.Array:
  """Per-DOF implicit-Euler relaxation of every velocity-proportional force."""
  if sys.dof_damping_total is None or sys.dof_inertia is None:
    return qd
  ratio = sys.opt.timestep * sys.dof_damping_total / sys.dof_inertia
  return qd / (1.0 + ratio)


def passive(sys, q: jax.Array, qd_imp: jax.Array) -> jax.Array:
  """Joint spring and damper forces, in DOF coordinates.

  Mirrors `brax.generalized.dynamics._passive` so the two pipelines apply the
  same passive model; `qd_imp` is the relaxed velocity, so the damper is the
  implicit one.
  """

  def stiffness_fn(typ, q, dof):
    if typ in 'fb':
      return jp.zeros_like(dof.stiffness)
    return -q * dof.stiffness

  frc = scan.link_types(sys, stiffness_fn, 'qd', 'd', q, sys.dof)
  return frc - sys.dof.damping * qd_imp


def joint_force(sys, act: jax.Array, q: jax.Array, qd: jax.Array) -> jax.Array:
  """Actuator + passive joint force with implicit velocity damping."""
  qd_imp = velocity_relaxation(sys, qd)
  return actuator.to_tau(sys, act, q, qd_imp) + passive(sys, q, qd_imp)
