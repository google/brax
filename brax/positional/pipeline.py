# Copyright 2026 The Brax Authors.
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

"""Physics pipeline for fully articulated dynamics and collisiion."""

# pylint:disable=g-multiple-import
from typing import Optional
from brax import actuator
from brax import com
from brax import contact
from brax import _fork_joint_dynamics
from brax import fluid
from brax import kinematics
from brax.base import Motion, System, Transform
from brax.io import mjcf
from brax.positional import collisions
from brax.positional import integrator
from brax.positional import joints
from brax.positional.base import State
import jax
from jax import numpy as jp
import os
import numpy as np

# explore_bench fork: opt-in, A/B'd against the original (see _contact_sweeps).
_CONTACT_PRE = os.environ.get(
    'BRAX_FORK_CONTACT_PRE', '0').lower() not in ('0', '', 'false', 'no')
# BRAX_FORK_DEPTH_SIGN_LEGACY=1 restores the inverted depth refresh below,
# for A/B only.
_DEPTH_SIGN = -1.0 if os.environ.get(
    'BRAX_FORK_DEPTH_SIGN_LEGACY', '0').lower() not in (
        '0', '', 'false', 'no') else 1.0


def init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
    unused_act: Optional[jax.Array] = None,
    unused_ctrl: Optional[jax.Array] = None,
    debug: bool = False,
) -> State:
  """Initializes physics state.

  Args:
    sys: a brax system
    q: (q_size,) joint angle vector
    qd: (qd_size,) joint velocity vector
    debug: if True, adds contact to the state for debugging

  Returns:
    state: initial physics state
  """
  if sys.mj_model is not None:
    mjcf.validate_model(sys.mj_model)
  # position/velocity level terms
  x, xd = kinematics.forward(sys, q, qd)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  x_i, xd_i = com.from_world(sys, x, xd)
  c = contact.get(sys, x) if debug else None
  mass = sys.link.inertia.mass ** (1 - sys.spring_mass_scale)
  return State(q, qd, x, xd, c, x_i, xd_i, j, jd, a_p, a_c, mass)


def step(
    sys: System, state: State, act: jax.Array, debug: bool = False
) -> State:
  """Performs a single physics step using position-based dynamics.

  Resolves actuator forces, joints, and forces at acceleration level, and
  resolves collisions at velocity level with baumgarte stabilization.

  Args:
    sys: system defining the kinematic tree and other properties
    state: physics state prior to step
    act: (act_size,) actuator input vector
    debug: if True, adds contact to the state for debugging

  Returns:
    x: updated link transform in world frame
    xd: updated link motion in world frame
  """
  x_i_prev = state.x_i

  # calculate acceleration level updates
  # explore_bench fork: actuator force PLUS the passive joint spring/damper,
  # with every velocity-proportional term relaxed implicitly. brax previously
  # dropped dof_damping/jnt_stiffness here (only `generalized` read them) and
  # integrated the actuator's velocity feedback explicitly, which is
  # unconditionally unstable at the gains these models use.
  tau = _fork_joint_dynamics.joint_force(sys, act, state.q, state.qd)
  xdd_i = Motion.create(vel=sys.gravity)
  # get joint constraint forces
  xf_i = joints.acceleration_update(sys, state, tau)
  # explore_bench fork: site-transmission actuators apply a wrench at a frame
  # (rotor thrust), which to_tau cannot express -- see actuator.site_force.
  if sys.site_act_link is not None:
    xf_i += actuator.site_force(sys, act, state.x, state.x_i)
  if sys.enable_fluid:
    inertia = sys.link.inertia.i ** (1 - sys.spring_inertia_scale)
    xf_i += fluid.force(sys, state.x, state.xd, state.mass, inertia)
  xdd_i += Motion(
      ang=jax.vmap(lambda x, y: x @ y)(com.inv_inertia(sys, state.x), xf_i.ang),
      vel=jax.vmap(lambda x, y: x * y)(1 / state.mass, xf_i.vel),
  )

  # semi-implicit euler: apply acceleration update before resolving collisions
  x_i, xd_i = integrator.integrate_xdd(sys, state.x_i, state.xd_i, xdd_i)
  x, xd = com.to_world(sys, x_i, xd_i)
  state = state.replace(x=x, xd=xd, x_i=x_i, xd_i=xd_i)

  # perform position level joint updates
  # explore_bench fork: iterate the projection. One Jacobi sweep propagates a
  # constraint exactly one link along the tree, so a 6-link arm holding itself
  # against its own position servos never converged within a substep and
  # visibly sagged. `fori_loop` rather than a Python loop: the sweeps sit
  # inside the frame_skip scan, and unrolling them multiplied an already large
  # graph by the iteration count.
  # The servo drive rides INSIDE the sweep loop, re-derived at the current
  # configuration each time (see joints.drive_update). One application is too
  # weak on its own -- the correction is split between parent and child, and
  # rotating a link about its centre of mass is not rotating it about the
  # joint, so the joint projection immediately gives some of it back -- while
  # re-applying a STALE correction n times overshoots. Recomputing the angle
  # makes the sweep a contraction: n sweeps converge toward the target instead
  # of multiplying the first step by n.
  # The drive's Lagrange multiplier is carried ACROSS sweeps and reset each
  # substep -- that is what makes the result independent of the sweep count
  # (see joints.drive_update).
  n_drive = sum(1 for v in (sys.drive_act or ()) if v >= 0)
  lam0 = jp.zeros(sys.num_links()) if n_drive else None

  def _apply(st, x_i):
    x, _ = com.to_world(sys, x_i, st.xd_i)
    return st.replace(x=x, x_i=x_i)

  def _joint_sweep(carry, _):
    st, lam = carry
    # GAUSS-SEIDEL over the three constraint families, not Jacobi.
    #
    # The joint projection, the servo drives and the equality couplings each
    # compute a FULL correction. Evaluating all three against the same state
    # and summing them applies roughly three times the displacement any one of
    # them intended, and the error compounds with every sweep -- measured on
    # the arm pair, the sweep-count response was NON-MONOTONIC (2-4 sweeps
    # reached NaN at control step 1-2 while 6-16 survived to step 13), which
    # is a signature of over-relaxation rather than of slow convergence. The
    # runaway joints were the Robotiq linkage's `follower` and `spring_link`,
    # which are acted on by all three families at once.
    #
    # Updating the state between families costs one extra kinematics pass each
    # and makes every family see the corrections already applied.
    st = _apply(st, joints.position_update(sys, st))
    dq, lam = joints.drive_update(sys, st, act, lam)
    st = _apply(st, st.x_i + dq)
    if sys.eq_q1:
      from brax import kinematics as _kin
      _j, _jd, _, _ = _kin.world_to_joint(sys, st.x, st.xd)
      _q, _ = _kin.inverse(sys, _j, _jd)
      st = _apply(st, st.x_i + joints.equality_update(sys, st, _q))
    return (st, lam), None

  n_sweeps = max(1, int(sys.joint_solver_iterations))
  (state, _), _ = jax.lax.scan(_joint_sweep, (state, lam0), None,
                               length=n_sweeps)
  x_i, x = state.x_i, state.x

  # apply position level collision updates
  #
  # Contacts are DETECTED once and then PROJECTED inside the same sweep loop
  # the joints use. Resolving them once, after the joints, is what let a
  # gripper close through the object it is holding: the finger link's own
  # inertia is ~1e-6 kg m^2, so a single post-hoc contact impulse at a 20 mm
  # lever produced hundreds of rad/s (measured: up to 687 on the Robotiq
  # links), and brax's Panda jaw shut through a 26 mm ball to -0.0001 m where
  # MuJoCo holds it at 0.0163. Iterating lets the joint constraints carry the
  # contact load into the arm instead of the fingertip absorbing all of it.
  #
  # Detection is NOT repeated -- it is 49% of the substep. Within one 2 ms
  # substep the contact normal and the material points barely rotate, so the
  # manifold is held fixed and only the PENETRATION DEPTH is refreshed from
  # the current pose, which is the standard position-based contact solve.
  c = contact.get(sys, x)
  x_i, dlambda = _contact_sweeps(sys, state, x_i_prev, c, xd_i)
  xd_i_prev = xd_i

  xd_i = integrator.project_xd(sys, x_i, x_i_prev)
  x, xd = com.to_world(sys, x_i, xd_i)
  state = state.replace(x=x, xd=xd, x_i=x_i, xd_i=xd_i)

  # apply velocity level collision updates
  xdv_i = collisions.resolve_velocity(sys, state, xd_i_prev, c, dlambda)
  xd_i = integrator.integrate_xdv(sys, xd_i, xdv_i)

  x, xd = com.to_world(sys, x_i, xd_i)
  j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
  q, qd = kinematics.inverse(sys, j, jd)

  # explore_bench fork: the position servo's kv, as a per-DOF IMPLICIT
  # velocity relaxation on the STATE.
  #
  # `io/mjcf.py` folds the actuator's kv into `dof_damping_total`, but NOTHING
  # EVER APPLIED IT: `actuator.to_tau` is masked out for every driven actuator
  # (`drive_force_mask`), `passive()` reads `sys.dof.damping`, which is
  # MuJoCo's own dof_damping and is 0.0 on every UR5e joint, and
  # `_fork_joint_dynamics.velocity_relaxation` uses dof_damping_total only as a
  # ratio on the velocity handed to the force computation. So the drive was a
  # pure P servo: it holds a pose fine and its step response is bounded, but on
  # a RAMP it rings and accumulates lead -- measured on juggle-gripper's right
  # shoulder_lift, +0.306 rad PAST its own target by control step 8, at
  # 7.4 rad/s against MuJoCo's 1.24.
  #
  # It cannot go back on the FORCE path: brax divides a joint torque by the
  # LINK's own inertia, and kv = 400 against a 0.1 kg m^2 link at h = 2 ms is
  # far past the explicit stability limit (measured NaN at control step 1-3 at
  # 5-10% of the real kv). `qd <- qd / (1 + h kv / I)` is unconditionally
  # stable and is exactly MuJoCo's `implicitfast` treatment of the same term.
  #
  # `sys.dof.damping` is subtracted so the joint's own damping, which
  # `passive()` already applies, is not counted twice; the mask restricts this
  # to dofs whose actuator `to_tau` has been zeroed, so the non-masked tendon
  # actuators keep their existing force path.
  if sys.drive_act is not None and sys.dof_damping_total is not None:
    driven = jp.asarray(np.asarray(sys.drive_act) >= 0, qd.dtype)
    kv = jp.maximum(sys.dof_damping_total - sys.dof.damping, 0.0) * driven
    relax = 1.0 / (1.0 + sys.opt.timestep * kv
                   / jp.maximum(sys.dof_inertia, 1e-9))
    qd = qd * relax
    x, xd = kinematics.forward(sys, q, qd)
    j, jd, a_p, a_c = kinematics.world_to_joint(sys, x, xd)
    x_i, xd_i = com.from_world(sys, x, xd)

  c = contact.get(sys, x) if debug else None

  return State(q, qd, x, xd, c, x_i, xd_i, j, jd, a_p, a_c, state.mass)

def _contact_sweeps_vecform(sys, state, x_i_prev, c, xd_i):
  """Project a FIXED contact manifold repeatedly, refreshing depth each pass.

  `contact.get` is 49% of a substep, so re-detecting per sweep is not on the
  table. Instead the manifold (which links, which normal, which material
  points) is held fixed and `dist` is recomputed from the current pose:

      dist_now = dist_0 + n . (p_c_now - p_p_now)

  where `p_p`/`p_c` are the contact point carried rigidly by each of the two
  links. At detection they coincide, so the correction term starts at zero and
  grows exactly as the bodies separate -- which is what stops the second and
  later passes from re-applying a correction already made.
  """
  if c is None:
    return collisions.resolve_position(sys, state, x_i_prev, c)
  n_pass = max(1, int(sys.contact_solver_iterations))
  if n_pass == 1:
    return collisions.resolve_position(sys, state, x_i_prev, c)

  from brax import math
  x0 = state.x.concatenate(Transform.zero((1,)))
  # `link_idx` is a TUPLE of two (ncon,) arrays, so it stacks to (2, ncon);
  # transpose to (ncon, 2) so a contact is the leading axis. -1 indexes the
  # appended zero transform, which is how brax denotes the world body.
  link = jp.array(c.link_idx).T
  # SIGN.  `resolve_position` moves link_idx[0] along `-frame[0]` and
  # link_idx[1] along `+frame[0]`, so the pair SEPARATES as
  # `dot(frame[0], p1 - p0)` grows and the refreshed depth is
  #     dist_now = dist_0 + frame[0] . (p_c_now - p_p_now)
  # This read `-frame[0]`, which reports a separating pair as PENETRATING
  # DEEPER, so every pass after the first pushed harder than the last and the
  # total push-out came out (2^n_pass - 1) times the real penetration.
  # Invisible at one pass and exponential in the pass count -- which is what
  # made the pass count look like a convergence knob when it was a divergence
  # one.  Verified against a fresh narrowphase on the corrected pose
  # (`scratchpad/depth_sign.py`): on a contact row that stays live, dist0
  # -0.000846 refreshes to -0.000904 as coded and -0.000788 with this sign,
  # against a re-detected -0.000788.
  n_w = _DEPTH_SIGN * c.frame[:, 0]
  # the contact point's offset on each of the two links, in that link's frame
  q0 = x0.rot.take(link, axis=0)
  p0 = x0.pos.take(link, axis=0)

  def _to_local(qq, pp, cp):
    return jax.vmap(lambda q1, p1: math.rotate(cp - p1, math.quat_inv(q1)))(
        qq, pp)

  r_local = jax.vmap(_to_local)(q0, p0, c.pos)
  dist0 = c.dist

  def body(st):
    xw = st.x.concatenate(Transform.zero((1,)))
    qn = xw.rot.take(link, axis=0)
    pn = xw.pos.take(link, axis=0)
    def _to_world(qq, pp, rr):
      return jax.vmap(lambda q1, p1, r1: p1 + math.rotate(r1, q1))(qq, pp, rr)

    pw = jax.vmap(_to_world)(qn, pn, r_local)
    depth = dist0 + jax.vmap(jp.dot)(n_w, pw[:, 1] - pw[:, 0])
    x_i_new, dl = collisions.resolve_position(
        sys, st, x_i_prev, c.replace(dist=depth))
    xw2, _ = com.to_world(sys, x_i_new, xd_i)
    return st.replace(x=xw2, x_i=x_i_new), dl

  # a Python loop, not lax.scan: `n_pass` is a small static count and the
  # `dlambda` carry has no natural zero to initialise a scan with
  #
  # TRIED AND REVERTED: interleaving `joints.position_update` BETWEEN contact
  # passes, so the chain carries the contact load instead of the fingertip
  # absorbing it (which is what the comment in `step` claims happens, and does
  # not). It is a real gap -- `resolve_position` sizes the push-out with the
  # contacted link's OWN inverse mass, as though a few-gram Robotiq finger at
  # the tip of a nine-link arm were free -- but the interleave did NOT fix
  # juggle-gripper (still 4/4 non-finite on the demo gate) and costs an extra
  # kinematics pass per contact pass on the hottest path in the substep. The
  # real fix needs the ARTICULATED inverse mass at the contact point, not more
  # iterations of the wrong one.
  # The pose BEFORE ANY pass, not before the last one.
  #
  # `resolve_position` hands `resolve_velocity` the pre-correction pose so it
  # can subtract the velocity `integrator.project_xd` manufactures out of the
  # push-out (a positional correction of d metres becomes d/h m/s of REAL
  # velocity). Taking it from the last pass leaves the earlier passes'
  # push-out uncancelled: at n_pass=2 half the correction became velocity, at
  # 4 three quarters, which is exactly the shape of the measured pass-count
  # response (2 passes lost the ball, 8 went non-finite). Measured on the
  # grasp gate, held Panda jaw with a 26 mm ball: the ball left an 8 mm
  # penetration with a 19 mm GAP one control step later, i.e. ~2.7 m/s
  # manufactured by the correction that was supposed to be momentum-neutral.
  x_i_pre0 = state.x_i
  dlambda = None
  for _ in range(n_pass):
    state, dlambda = body(state)
  if _CONTACT_PRE and dlambda is not None:
    dlambda = (dlambda[0], x_i_pre0)
  return state.x_i, dlambda


def _contact_sweeps(sys, state, x_i_prev, c, xd_i):
  """`_contact_sweeps_vecform` with the depth refresh in LANE form.

  explore_bench fork, layout only.  The refresh gathers a link pose per
  contact and rotates a point by it, and written packed that is
  `(n_worlds, n_con, 2, 4)` / `(n_worlds, n_con, 2, 3)` per pass -- the same
  trailing-size-3 layout that makes `collisions.resolve_position` 46% of the
  substep's `bytes accessed` (see `brax/_fork_contact_lane.py`).  Component
  arrays are `(n_worlds, n_con)`.

  `BRAX_FORK_LANE_SWEEP=0` restores the packed form above; the arithmetic is
  identical either way and `.work/contact_equiv.py` checks it.
  """
  from brax import _fork_contact_lane as _clane
  if c is None or not _clane.ENABLE_SWEEP:
    return _contact_sweeps_vecform(sys, state, x_i_prev, c, xd_i)
  n_pass = max(1, int(sys.contact_solver_iterations))
  if n_pass == 1:
    return collisions.resolve_position(sys, state, x_i_prev, c)

  link = (jp.asarray(c.link_idx[0]), jp.asarray(c.link_idx[1]))
  # Same corrected depth-refresh sign as the vecform -- see `_DEPTH_SIGN`.
  # The inverted original made SEPARATING bodies report DEEPER penetration,
  # so the push-out came out (2^n_pass - 1)x the true depth. Both copies must
  # carry the same sign or `.work/contact_equiv.py` trips on the sweep rows.
  n_w = (_DEPTH_SIGN * c.frame[..., 0, 0],
         _DEPTH_SIGN * c.frame[..., 0, 1],
         _DEPTH_SIGN * c.frame[..., 0, 2])
  cpos = (c.pos[..., 0], c.pos[..., 1], c.pos[..., 2])
  r_local = _clane.contact_local(sys, state.x, link, cpos)
  dist0 = c.dist

  def body(st):
    depth = _clane.refresh_depth(sys, st.x, link, r_local, n_w, dist0)
    x_i_new, dl = collisions.resolve_position(
        sys, st, x_i_prev, c.replace(dist=depth))
    xw2, _ = com.to_world(sys, x_i_new, xd_i)
    return st.replace(x=xw2, x_i=x_i_new), dl

  dlambda = None
  for _ in range(n_pass):
    state, dlambda = body(state)
  return state.x_i, dlambda
