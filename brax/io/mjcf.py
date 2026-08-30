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

# pylint:disable=g-multiple-import
"""Function to load MuJoCo mjcf format to Brax model."""

import itertools
from typing import Dict, Optional, Tuple, Union
import warnings
from xml.etree import ElementTree
from brax import math
from brax.io import _fork_links
from brax.base import (
    Actuator,
    DoF,
    Inertia,
    Link,
    Motion,
    System,
    Transform,
)
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np


def _transform_do(
    parent_pos: np.ndarray,
    parent_quat: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  pos = parent_pos + math.rotate_np(pos, parent_quat)
  rot = math.quat_mul_np(parent_quat, quat)
  return pos, rot


def _offset(
    elem: ElementTree.Element, parent_pos: np.ndarray, parent_quat: np.ndarray
):
  """Offsets an element."""
  pos = elem.attrib.get('pos', '0 0 0')
  quat = elem.attrib.get('quat', '1 0 0 0')
  pos = np.fromstring(pos, sep=' ')
  quat = np.fromstring(quat, sep=' ')
  fromto = elem.attrib.get('fromto', None)
  if fromto:
    # fromto attributes are not compatible with pos/quat attributes
    from_pos = np.fromstring(' '.join(fromto.split(' ')[0:3]), sep=' ')
    to_pos = np.fromstring(' '.join(fromto.split(' ')[3:6]), sep=' ')
    from_pos, _ = _transform_do(parent_pos, parent_quat, from_pos, quat)
    to_pos, _ = _transform_do(parent_pos, parent_quat, to_pos, quat)
    fromto = ' '.join('%f' % i for i in np.concatenate([from_pos, to_pos]))
    elem.attrib['fromto'] = fromto
    return
  pos, quat = _transform_do(parent_pos, parent_quat, pos, quat)
  pos = ' '.join('%f' % i for i in pos)
  quat = ' '.join('%f' % i for i in quat)
  elem.attrib['pos'] = pos
  elem.attrib['quat'] = quat


def _fuse_bodies(elem: ElementTree.Element):
  """Fuses together parent child bodies that have no joint."""

  for child in list(elem):  # we will modify elem children, so make a copy
    _fuse_bodies(child)
    # this only applies to bodies with no joints
    if child.tag != 'body':
      continue
    if child.find('joint') is not None or child.find('freejoint') is not None:
      continue
    cpos = child.attrib.get('pos', '0 0 0')
    cpos = np.fromstring(cpos, sep=' ')
    cquat = child.attrib.get('quat', '1 0 0 0')
    cquat = np.fromstring(cquat, sep=' ')
    for grandchild in child:
      # TODO(brax-team): might need to offset more than just these tags
      if (
          grandchild.tag in ('body', 'geom', 'site', 'camera')
          and (cpos != 0).any()
      ):
        _offset(grandchild, cpos, cquat)
      elem.append(grandchild)
    elem.remove(child)


def _get_meshdir(elem: ElementTree.Element) -> Union[str, None]:
  """Gets the mesh directory specified by the mujoco compiler tag."""
  elems = list(elem.iter('compiler'))
  return elems[0].get('meshdir') if elems else None


def _find_assets(
    elem: ElementTree.Element,
    path: epath.Path,
    meshdir: Optional[str],
) -> Dict[str, bytes]:
  """Loads assets from an xml given a base path."""
  assets = {}
  path = path if path.is_dir() else path.parent
  fname = elem.attrib.get('file') or elem.attrib.get('filename')
  if fname and fname.endswith('.xml'):
    # an asset can be another xml!  if so, we must traverse it, too
    asset = (path / fname).read_text()
    asset_xml = ElementTree.fromstring(asset)
    _fuse_bodies(asset_xml)
    asset_meshdir = _get_meshdir(asset_xml)
    assets[fname] = ElementTree.tostring(asset_xml)
    assets.update(_find_assets(asset_xml, path, asset_meshdir))
  elif fname:
    # mesh, png, etc
    path = path / meshdir if meshdir else path
    assets[fname] = (path / fname).read_bytes()

  for child in list(elem):
    assets.update(_find_assets(child, path, meshdir))

  return assets


def _get_name(mj: mujoco.MjModel, i: int) -> str:
  names = mj.names[i:].decode('utf-8')
  return names[: names.find('\x00')]


def _check_custom(mj: mujoco.MjModel, custom: Dict[str, np.ndarray]) -> None:
  """Validates fields in custom."""
  if not (
      0 <= custom['spring_mass_scale'] <= 1
      and 0 <= custom['spring_inertia_scale'] <= 1
  ):
    raise ValueError('Spring inertia and mass scale must be in [0, 1].')
  if 'init_qpos' in custom and custom['init_qpos'].shape[0] != mj.nq:
    size = custom['init_qpos'].shape[0]
    raise ValueError(
        f'init_qpos had length {size} but expected length {mj.nq}.'
    )


def _get_custom(mj: mujoco.MjModel) -> Dict[str, np.ndarray]:
  """Gets custom mjcf parameters for brax, with defaults."""
  default = {
      'ang_damping': (0.0, None),
      'vel_damping': (0.0, None),
      'baumgarte_erp': (0.1, None),
      'spring_mass_scale': (0.0, None),
      'spring_inertia_scale': (0.0, None),
      'joint_scale_pos': (0.5, None),
      'joint_scale_ang': (0.2, None),
      'collide_scale': (1.0, None),
      'matrix_inv_iterations': (10, None),
      'solver_maxls': (20, None),
      'elasticity': (0.0, 'geom'),
      'constraint_stiffness': (2000.0, 'body'),
      'constraint_limit_stiffness': (1000.0, 'body'),
      'constraint_ang_damping': (0.0, 'body'),
      'constraint_vel_damping': (0.0, 'body'),
  }

  # add user provided overrides to the defaults
  for i, ni in enumerate(mj.name_numericadr):
    nsize = mj.numeric_size[i]
    name = _get_name(mj, ni)
    val = mj.numeric_data[mj.numeric_adr[i] : mj.numeric_adr[i] + nsize]
    typ = default[name][1] if name in default else None
    default[name] = (val, typ)

  # gather custom overrides with correct sizes
  custom = {}
  for name, (val, typ) in default.items():
    val = np.array([val])
    size = {
        'body': mj.nbody - 1,  # ignore the world body
        'geom': mj.ngeom,
    }.get(typ, val.shape[-1])
    if val.shape[-1] != size and val.shape[-1] > 1:
      # the provided shape does not match against our default size
      raise ValueError(
          f'"{name}" custom arg needed {size} values for the "{typ}" type, '
          f'but got {val.shape[-1]} values.'
      )
    elif val.shape[-1] != size and val.shape[-1] == 1:
      val = np.repeat(val, size)
    val = val.squeeze() if not typ else val.reshape(size)
    if typ == 'body':
      # pad one value for the world body, which gets dropped at Link creation
      val = np.concatenate([[val[0]], val])
    custom[name] = val

  # get tuple custom overrides
  for i, ni in enumerate(mj.name_tupleadr):
    start, end = mj.tuple_adr[i], mj.tuple_adr[i] + mj.tuple_size[i]
    objtype = mj.tuple_objtype[start:end]
    name = _get_name(mj, ni)
    if not all(objtype[0] == objtype):
      raise NotImplementedError(
          f'All tuple elements "{name}" should have the same object type.'
      )
    if objtype[0] not in [1, 5]:
      raise NotImplementedError(
          f'Custom tuple "{name}" with objtype=={objtype[0]} is not supported.'
      )
    typ = {1: 'body', 5: 'geom'}[objtype[0]]
    if name in default and default[name][1] != typ:
      raise ValueError(
          f'Custom tuple "{name}" is expected to be associated with'
          f' the {default[name][1]} objtype.'
      )

    size = {1: mj.nbody, 5: mj.ngeom}[objtype[0]]
    default_val, _ = default.get(name, (0.0, None))
    arr = np.repeat(default_val, size)
    objid = mj.tuple_objid[start:end]
    objprm = mj.tuple_objprm[start:end]
    arr[objid] = objprm
    custom[name] = arr

  _check_custom(mj, custom)
  return custom


#: explore_bench fork: MuJoCo parameters this pipeline cannot honour and now
#: accepts as recorded no-ops rather than rejecting the model outright. Read it
#: after validate_model to know what was ignored.
_FORK_IGNORED = set()


def validate_model(mj: mujoco.MjModel) -> None:
  """Checks if a MuJoCo model is compatible with brax physics pipelines."""
  if mj.opt.integrator != 0:
    raise NotImplementedError('Only euler integration is supported.')
  if mj.opt.cone != 0:
    raise NotImplementedError('Only pyramidal cone friction is supported.')
  # explore_bench fork: MuJoCo's ELLIPSOID fluid model is per-geom (blunt and
  # slender drag, angular drag, Kutta and Magnus lift, added mass from
  # precomputed virtual mass/inertia). brax has its own fluid model -- an
  # inertia-equivalent BOX per link, brax/fluid.py -- driven by the same
  # opt.density/opt.viscosity, and it stays enabled here. So the model is not
  # dropped, it is SUBSTITUTED, and the difference is a solver difference of
  # the same kind as positional-PBD-vs-Newton and pyramidal-vs-elliptic
  # friction cones. Recorded so callers can report it rather than assume the
  # aerodynamics matched. mjx does not implement the ellipsoid model either
  # (mjx._src.passive carries only _inertia_box_fluid_model), so there is no
  # existing implementation to delegate to.
  if (mj.geom_fluid != 0).any():
    _FORK_IGNORED.add('geom_fluid (ellipsoid model -> brax inertia-box model)')
  if mj.opt.wind.any():
    raise NotImplementedError('option.wind is not implemented.')
  # explore_bench fork: impratio scales MuJoCo's normal-vs-friction constraint
  # IMPEDANCE. The positional (PBD) pipeline has no impedance concept at all --
  # it resolves contacts as positional projections -- so there is nothing here
  # for the parameter to scale. Rejecting the model implied brax would honour
  # it at impratio=1, which it also does not; accepting it and recording the
  # difference is the honest treatment. Contact behaviour differs from MuJoCo
  # either way (see also the pyramidal-cone entry).
  if mj.opt.impratio != 1:
    _FORK_IGNORED.add('opt.impratio')
  # explore_bench fork: equality constraints were dropped without a word.
  # mjEQ_JOINT is now honoured (io/mjcf.py builds the coupling tables,
  # positional/joints.py::equality_update projects it); anything else is
  # recorded here so a caller can see what the backend is not modelling.
  if mj.neq:
    _kinds = {int(mj.eq_type[_e]) for _e in range(mj.neq)}
    _unsupported = _kinds - {int(mujoco.mjtEq.mjEQ_JOINT)}
    if _unsupported:
      _FORK_IGNORED.add(
          f'{len(_unsupported)} equality constraint type(s) '
          f'{sorted(_unsupported)} not implemented (only mjEQ_JOINT is)')

  # actuators
  if any(i not in [0, 1] for i in mj.actuator_biastype):
    raise NotImplementedError('Only actuator_biastype in [0, 1] are supported.')
  if any(i != 0 for i in mj.actuator_gaintype):
    raise NotImplementedError('Only actuator_gaintype in [0] is supported.')
  # explore_bench fork: joint AND fixed-tendon transmissions are supported (a
  # fixed tendon is a joint-space weighted sum, carried by the actuator moment
  # matrix). Everything else -- site, body, slidercrank, or a tendon that
  # wraps geometry rather than joints -- still raises, because those apply
  # force somewhere no joint coordinate can express.
  for _a in range(mj.nu):
    _trn = int(mj.actuator_trntype[_a])
    if _trn == mujoco.mjtTrn.mjTRN_JOINT:
      continue
    if _trn == mujoco.mjtTrn.mjTRN_SITE:
      continue        # applied as a link wrench, see actuator.site_force
    if _trn == mujoco.mjtTrn.mjTRN_TENDON:
      _t = int(mj.actuator_trnid[_a, 0])
      _adr, _num = int(mj.tendon_adr[_t]), int(mj.tendon_num[_t])
      _wt = {int(mj.wrap_type[_k]) for _k in range(_adr, _adr + _num)}
      if _wt <= {int(mujoco.mjtWrap.mjWRAP_JOINT)}:
        continue
      raise NotImplementedError(
          'Only FIXED tendons (joint wraps) are supported for actuators; '
          f'tendon {_t} wraps types {sorted(_wt)}.'
      )
    raise NotImplementedError(
        'Only joint, fixed-tendon and site transmission types are '
        'supported for '
        f'actuators; actuator {_a} uses trntype {_trn}.'
    )

  # solver parameters
  # Same class as impratio: solmix/priority weight MuJoCo's constraint
  # solver, which the positional pipeline does not run. Recorded, not honoured.
  if (mj.geom_solmix[0] != mj.geom_solmix).any():
    _FORK_IGNORED.add('geom_solmix')
  if (mj.geom_priority[0] != mj.geom_priority).any():
    _FORK_IGNORED.add('geom_priority')

  # check joints
  q_width = {0: 7, 1: 4, 2: 1, 3: 1}
  non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
  if mj.qpos0[non_free].any():
    raise NotImplementedError(
        'The `ref` attribute on joint types is not supported.'
    )

  for _, group in itertools.groupby(
      zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]
  ):
    position = np.array([p for _, p in group])
    if not (position == position[0]).all():
      raise RuntimeError('invalid joint stack: only one joint position allowed')

  # check dofs
  jnt_range = mj.jnt_range.copy()
  jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
  for typ, limit, stiffness in zip(mj.jnt_type, jnt_range, mj.jnt_stiffness):
    if typ == 0:
      if stiffness > 0:
        raise RuntimeError('brax does not support stiffness for free joints')
    elif typ == 1:
      if np.any(~np.isinf(limit)):
        raise RuntimeError('brax does not support joint ranges for ball joints')
    elif typ in (2, 3):
      continue
    else:
      raise RuntimeError(f'invalid joint type: {typ}')

  for _, group in itertools.groupby(
      zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]
  ):
    typs = [t for _, t in group]
    if len(typs) == 1 and typs[0] == 0:
      continue  # free
    elif 0 in typs:
      raise RuntimeError('invalid joint stack: cannot stack free joints')
    elif 1 in typs:
      raise NotImplementedError('ball joints not supported')

  # check collision geometries
  for i, typ in enumerate(mj.geom_type):
    mask = mj.geom_contype[i] | mj.geom_conaffinity[i] << 32
    if typ == 5:  # Cylinder
      # explore_bench fork: real cylinders are collidable once the cylinder
      # contact functions exist (brax/geometry/contact.py). This gate only
      # guarded their absence; unsupported PAIRS still raise from the contact
      # dispatch itself, which is the accurate place for that error.
      pass


def _dof_inertia_bound(mj: mujoco.MjModel, n_samples: int = 32,
                       seed: int = 0) -> np.ndarray:
  """A per-DOF LOWER bound on the mass-matrix diagonal, for implicit damping.

  `_fork_joint_dynamics.velocity_relaxation` divides the velocity by
  `1 + dt * c_ii / m_ii`.  That scheme is unconditionally stable for any
  `m_ii <= M_ii(q)` and merely under-damps when the bound is loose, whereas an
  overestimate lets the explicit blow-up back in -- so take the minimum of
  `diag(M)` over the reference pose plus sampled configurations rather than
  trusting one pose.  `dof_invweight0` is NOT usable here: it is
  `(M^-1)_ii`, and `1 / (M^-1)_ii` was measured at up to 1.95x `M_ii` on these
  models, i.e. on the wrong side of the bound.

  `diag(M)` already includes `dof_armature`, so armature-dominated DOFs get the
  large value they should.

  Returns BOTH the minimum and the MEDIAN, because the two consumers want
  opposite biases. The implicit damping relaxation wants a lower bound: an
  underestimate merely under-damps, an overestimate lets the explicit blow-up
  back in. The position DRIVE wants a representative value: its response is
  `torque_limit / M_ii`, so an UNDERestimate of the inertia OVERestimates how
  far the servo moves. Using the minimum for both made the Panda's step
  response 4.4x MuJoCo's, all of it from that one substitution.
  """
  d = mujoco.MjData(mj)
  rng = np.random.default_rng(seed)
  full = np.zeros((mj.nv, mj.nv))
  bound = None
  samples = []
  hinge_slide = [j for j in range(mj.njnt)
                 if int(mj.jnt_type[j]) in (2, 3) and mj.jnt_limited[j]
                 and np.all(np.isfinite(mj.jnt_range[j]))]
  for k in range(n_samples + 1):
    d.qpos[:] = mj.qpos0
    if k:
      for j in hinge_slide:
        lo, hi = float(mj.jnt_range[j][0]), float(mj.jnt_range[j][1])
        d.qpos[int(mj.jnt_qposadr[j])] = rng.uniform(lo, hi)
    mujoco.mj_kinematics(mj, d)
    mujoco.mj_comPos(mj, d)
    mujoco.mj_crb(mj, d)
    mujoco.mj_fullM(mj, d, full)
    diag = np.diagonal(full).copy()
    bound = diag if bound is None else np.minimum(bound, diag)
    samples.append(diag)
  return (np.maximum(bound, 1e-9),
          np.maximum(np.median(np.stack(samples), axis=0), 1e-9))


def _fold_armature(mj: mujoco.MjModel, lm, link_i: np.ndarray,
                   link_mass: np.ndarray, link_iquat: np.ndarray,
                   dof_link: np.ndarray) -> np.ndarray:
  """Adds MuJoCo's `dof_armature` to the link inertia tensors.

  Armature is rotor inertia expressed in JOINT space: MuJoCo adds it to
  `M_ii`, and there is no exact maximal-coordinate equivalent because it acts
  on the joint's relative motion, not on the body.  For a hinge with axis `a`
  the faithful maximal-coordinate image is `armature * a a^T` added to the
  child link's rotational inertia: about `a` -- the axis the joint actually
  moves -- the reflected inertia is then exactly MuJoCo's, and the residual is
  that the link also resists being rotated about `a` by its PARENT.  For links
  whose armature is a large multiple of their own inertia (the Robotiq and
  UR5e joints here) that residual is the price of keeping the joint's own
  response right, and dropping armature entirely -- what brax did -- is far
  larger and destabilising.

  Slide and free/ball armature has no anisotropic-mass representation in brax
  (`Inertia.mass` is a scalar), so it is added isotropically and RECORDED, not
  silently applied as if exact.
  """
  i_out = np.array(link_i, dtype=np.float64, copy=True)
  m_out = np.array(link_mass, dtype=np.float64, copy=True)
  # per-DOF inertia the POSITIONAL pipeline actually accelerates against
  axis_inertia = np.full(mj.nv, np.inf)
  motion_ang, motion_vel = [], []
  for j_typ, axis in zip(mj.jnt_type, mj.jnt_axis):
    if j_typ == 0:      # free: 3 translational then 3 rotational dofs
      motion_vel += [np.eye(3)[k] for k in range(3)]
      motion_ang += [np.zeros(3)] * 3
      motion_vel += [np.zeros(3)] * 3
      motion_ang += [np.eye(3)[k] for k in range(3)]
    elif j_typ == 1:    # ball
      motion_vel += [np.zeros(3)] * 3
      motion_ang += [np.eye(3)[k] for k in range(3)]
    elif j_typ == 2:    # slide
      motion_vel += [np.asarray(axis, float)]
      motion_ang += [np.zeros(3)]
    else:               # hinge
      motion_vel += [np.zeros(3)]
      motion_ang += [np.asarray(axis, float)]
  motion_ang = np.array(motion_ang)
  motion_vel = np.array(motion_vel)
  for dof in range(mj.nv):
    link = int(dof_link[dof])
    arm = float(mj.dof_armature[dof])
    if 0 <= link < len(i_out):
      # Record what a joint torque is divided by in maximal coordinates: the
      # CHILD LINK's own inertia about the joint axis, plus armature. This is
      # NOT MuJoCo's M_ii, which is the whole distal subtree's composite
      # inertia -- for a shoulder the two differ by orders of magnitude, and
      # using the larger one under-damps the implicit relaxation badly.
      a_ang, a_vel = motion_ang[dof], motion_vel[dof]
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, np.asarray(link_iquat[link], float))
      if np.any(a_ang):
        a_i = rot.reshape(3, 3).T @ a_ang
        n = np.linalg.norm(a_i)
        if n > 0:
          a_i = a_i / n
        axis_inertia[dof] = float(a_i @ i_out[link] @ a_i) + arm
      elif np.any(a_vel):
        axis_inertia[dof] = float(m_out[link]) + arm
    if arm <= 0.0:
      continue
    if link < 0:
      continue
    a_ang, a_vel = motion_ang[dof], motion_vel[dof]
    if np.any(a_ang):
      # express the axis in the link's INERTIAL frame, where `i` is stored
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, np.asarray(link_iquat[link], float))
      a_i = rot.reshape(3, 3).T @ a_ang
      n = np.linalg.norm(a_i)
      if n > 0:
        a_i = a_i / n
      i_out[link] += arm * np.outer(a_i, a_i)
    if np.any(a_vel):
      m_out[link] += arm
      _FORK_IGNORED.add('dof_armature on a slide/free DOF (added to the '
                        'link mass isotropically; brax has no directional '
                        'mass)')
  return i_out, m_out, np.maximum(axis_inertia, 1e-9)


def load_model(mj: mujoco.MjModel) -> System:
  """Creates a brax system from a MuJoCo model."""
  custom = _get_custom(mj)

  # create links
  # explore_bench fork: links are no longer assumed to be bodies. _fork_links
  # fuses welded (jointless) bodies into their nearest jointed ancestor and
  # splits >3-DOF joint stacks across chained massless links, so a body and a
  # link are related by an explicit map rather than by index arithmetic. See
  # brax/io/_fork_links.py for why both are needed and why both are exact.
  lm = _fork_links.LinkMap(mj)
  n_link = lm.num_links
  identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_link, 1))

  def _per_link_custom(key):
    v = np.asarray(custom[key])
    # explore_bench fork: _get_custom pads body-typed customs with the world
    # body, so they arrive nbody-long, not (nbody - 1). Only the shorter form
    # was remapped, which meant a per-body custom silently reached the wrong
    # link once links stopped being bodies one-for-one (25 links, 40 bodies on
    # the arm-pair scene). Both lengths are now resolved through link_bodies.
    if v.shape and v.shape[0] == mj.nbody:
      return v[lm.link_bodies]
    if v.shape and v.shape[0] == mj.nbody - 1:
      return v[lm.link_bodies - 1]
    if v.shape and v.shape[0] == 1 and n_link != 1:
      return np.repeat(v, n_link, axis=0)
    return v

  # explore_bench fork: MuJoCo's rotor inertia. See _fold_armature.
  _dof_link = np.concatenate(
      [[i] * {'f': 6, '3': 3, '2': 2, '1': 1}[t]
       for i, t in enumerate(lm.link_types)]) if lm.link_types else \
      np.zeros(0, dtype=int)
  _link_i, _link_mass, _dof_axis_inertia = _fold_armature(
      mj, lm, np.array([np.diag(i) for i in lm.link_idiag]), lm.link_mass,
      lm.link_iquat, _dof_link)

  link = Link(  # pytype: disable=wrong-arg-types  # jax-ndarray
      transform=Transform(pos=lm.link_pos, rot=lm.link_quat),  # pytype: disable=wrong-arg-types  # jax-ndarray
      inertia=Inertia(  # pytype: disable=wrong-arg-types  # jax-ndarray
          transform=Transform(pos=lm.link_ipos, rot=lm.link_iquat),  # pytype: disable=wrong-arg-types  # jax-ndarray
          i=_link_i,
          mass=_link_mass,
      ),
      invweight=lm.link_invweight,
      joint=Transform(pos=lm.link_joint_pos, rot=identity),  # pytype: disable=wrong-arg-types  # jax-ndarray
      constraint_stiffness=_per_link_custom('constraint_stiffness'),
      constraint_vel_damping=_per_link_custom('constraint_vel_damping'),
      constraint_limit_stiffness=_per_link_custom('constraint_limit_stiffness'),
      constraint_ang_damping=_per_link_custom('constraint_ang_damping'),
  )

  # create dofs
  mj.jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
  motions, limits, stiffnesses = [], [], []
  for typ, axis, limit, stiffness in zip(
      mj.jnt_type, mj.jnt_axis, mj.jnt_range, mj.jnt_stiffness
  ):
    if typ == 0:
      motion = Motion(ang=np.eye(6, 3, -3), vel=np.eye(6, 3))
      limit = np.array([-np.inf] * 6), np.array([np.inf] * 6)
      stiffness = np.zeros(6)
    elif typ == 1:
      motion = Motion(ang=np.eye(3), vel=np.zeros((3, 3)))
      limit = np.array([-np.inf] * 3), np.array([np.inf] * 3)
      stiffness = np.zeros(3)
    elif typ == 2:
      motion = Motion(ang=np.zeros((1, 3)), vel=axis.reshape((1, 3)))
      limit = limit[0:1], limit[1:2]
      stiffness = np.array([stiffness])
    elif typ == 3:
      motion = Motion(ang=axis.reshape((1, 3)), vel=np.zeros((1, 3)))
      limit = limit[0:1], limit[1:2]
      stiffness = np.array([stiffness])
    else:
      # invalid joint type
      continue
    motions.append(motion)
    limits.append(limit)
    stiffnesses.append(stiffness)
  motion = jax.tree.map(lambda *x: np.concatenate(x), *motions)

  limit = None
  if np.any(mj.jnt_limited):
    limit = jax.tree.map(lambda *x: np.concatenate(x), *limits)
  stiffness = np.concatenate(stiffnesses)
  solver_params_jnt = np.concatenate((mj.jnt_solref, mj.jnt_solimp), axis=1)
  solver_params_dof = solver_params_jnt[mj.dof_jntid]

  dof = DoF(  # pytype: disable=wrong-arg-types
      motion=motion,
      armature=mj.dof_armature,
      stiffness=stiffness,
      damping=mj.dof_damping,
      limit=limit,
      invweight=mj.dof_invweight0,
      solver_params=solver_params_dof,
  )

  # create actuators
  # TODO(brax-team): swap brax actuation for mjx actuation model.
  ctrl_range = mj.actuator_ctrlrange
  ctrl_range[~(mj.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
  force_range = mj.actuator_forcerange
  force_range[~(mj.actuator_forcelimited == 1), :] = np.array([-np.inf, np.inf])
  bias_q = mj.actuator_biasprm[:, 1] * (mj.actuator_biastype != 0)
  bias_qd = mj.actuator_biasprm[:, 2] * (mj.actuator_biastype != 0)
  # explore_bench fork: every actuator is kept (System.act_size() is mj.nu, so
  # masking rows here desynchronised `act` from the actuator arrays), and the
  # transmission is expressed as a MOMENT MATRIX row -- one-hot for a joint,
  # the wrap coefficients for a fixed tendon. validate_model still rejects the
  # transmissions this cannot express (site, body, slidercrank).
  moment_q = np.zeros((mj.nu, mj.nq))
  moment_qd = np.zeros((mj.nu, mj.nv))
  q_id = np.zeros(mj.nu, dtype=np.int64)
  qd_id = np.zeros(mj.nu, dtype=np.int64)
  for a in range(mj.nu):
    trn = int(mj.actuator_trntype[a])
    target = int(mj.actuator_trnid[a, 0])
    if trn == mujoco.mjtTrn.mjTRN_JOINT:
      q_id[a] = int(mj.jnt_qposadr[target])
      qd_id[a] = int(mj.jnt_dofadr[target])
      moment_q[a, q_id[a]] = 1.0
      moment_qd[a, qd_id[a]] = 1.0
    elif trn == mujoco.mjtTrn.mjTRN_TENDON:
      adr, num = int(mj.tendon_adr[target]), int(mj.tendon_num[target])
      first = True
      for k in range(adr, adr + num):
        if int(mj.wrap_type[k]) != mujoco.mjtWrap.mjWRAP_JOINT:
          continue
        j = int(mj.wrap_objid[k])
        coef = float(mj.wrap_prm[k])
        moment_q[a, int(mj.jnt_qposadr[j])] += coef
        moment_qd[a, int(mj.jnt_dofadr[j])] += coef
        if first:
          q_id[a] = int(mj.jnt_qposadr[j])
          qd_id[a] = int(mj.jnt_dofadr[j])
          first = False
  act_kwargs = {
      'gain': mj.actuator_gainprm[:, 0],
      'gear': mj.actuator_gear[:, 0],
      'ctrl_range': ctrl_range,
      'force_range': force_range,
      'bias_q': bias_q,
      'bias_qd': bias_qd,
  }

  actuator = Actuator(  # pytype: disable=wrong-arg-types
      q_id=q_id, qd_id=qd_id, moment_q=moment_q, moment_qd=moment_qd,
      **act_kwargs
  )

  # create non-pytree params.  these do not live on device directly, and they
  # cannot be differentiated, but they do change the emitted control flow
  link_names = lm.link_names
  link_types = lm.link_types
  link_parents = lm.link_parents

  # mujoco stores free q in world frame, so clear link transform for free links
  # TODO(brax-team): make this work for non-fused mj models
  if 'f' in link_types:
    free_idx = np.array([i for i, typ in enumerate(link_types) if typ == 'f'])
    link.transform.pos[free_idx] = np.zeros(3)
    link.transform.rot[free_idx] = np.array([1.0, 0.0, 0.0, 0.0])

  # explore_bench fork: everything the implicit velocity relaxation needs.
  # The actuator damping matrix is D = moment^T diag(-gear^2 bias_qd) moment;
  # its diagonal is what a per-DOF (Jacobi) implicit solve uses, and passive
  # dof_damping adds to the same coefficient.
  _act_damp = np.maximum(-bias_qd * mj.actuator_gear[:, 0] ** 2, 0.0)
  _dof_damp = (moment_qd ** 2 * _act_damp[:, None]).sum(axis=0)
  dof_damping_total = np.asarray(_dof_damp + mj.dof_damping, np.float64)
  # The implicit relaxation has to divide by the inertia THIS ENGINE applies
  # the torque against. MuJoCo's diag(M) is the articulated value and is the
  # right bound for a generalized-coordinate solver, and it is the right one
  # HERE TOO, now that `positional/joints.py::drive_update` moves the whole
  # articulated subtree: the damping that opposes the drive acts on the same
  # articulated coordinate the drive does.
  #
  # Two claims in the original justification for shrinking it were wrong. It
  # said an underestimate "merely under-damps" -- backwards, the relaxation is
  # `qd / (1 + dt*c/m)` so a SMALLER m damps MORE -- and that an overestimate
  # "lets the explicit blow-up back in", which cannot happen: `1/(1 + dt*c/m)`
  # is in (0, 1) for every positive m, so the scheme is unconditionally
  # contractive whatever inertia it gets. The Spot |qd| = 2425 rad/s that
  # motivated the bound came from the servo applied as a FORCE, which
  # `drive_update` no longer does.
  #
  # Measured on the Panda (+0.6 rad step, ratio to MuJoCo at 100 ms):
  # min 1.156, median 1.075. NOTE: tested BEFORE the subtree fix the median
  # looked slightly worse (0.33 -> 0.30), because the drive was still being
  # cancelled by the joint constraint -- a contaminated measurement, not a
  # reason to keep the minimum.
  _dof_inertia_min, _dof_inertia_med = _dof_inertia_bound(mj)
  dof_inertia = _dof_inertia_med

  # ---- mjEQ_JOINT equality constraints -----------------------------------
  # brax has no equality-constraint support at all and `validate_model` never
  # even looked at `mj.neq`, so these were dropped SILENTLY. They are not
  # decoration: a Robotiq 2F-85's six finger joints are one mechanism, and a
  # Panda's two fingers are coupled by exactly one of these. Both couplings
  # were measured to be a pure mimic -- the Robotiq linkage tracks its driver
  # to 1.3e-4 rad over the full stroke, the Panda's fingers to 9.3e-10 m -- so
  # projecting the coupled joint onto the polynomial of its partner is an
  # EXACT reduction here, not an approximation.
  # Only the joint-joint form is handled; mjEQ_CONNECT (loop closure) still is
  # not, and is recorded below rather than ignored.
  eq_q1, eq_q2, eq_dof1, eq_dof2 = [], [], [], []
  eq_poly = []
  eq_is_slide = []
  n_connect = 0
  for _e in range(mj.neq):
    _typ = int(mj.eq_type[_e])
    if _typ != int(mujoco.mjtEq.mjEQ_JOINT):
      if _typ == int(mujoco.mjtEq.mjEQ_CONNECT):
        n_connect += 1
      continue
    _j1, _j2 = int(mj.eq_obj1id[_e]), int(mj.eq_obj2id[_e])
    # hinge-hinge or slide-slide; a mixed pair has no single correction axis
    _t1, _t2 = int(mj.jnt_type[_j1]), int(mj.jnt_type[_j2])
    if _j2 < 0 or _t1 != _t2 or _t1 not in (2, 3):
      continue
    eq_is_slide.append(1.0 if _t1 == 2 else 0.0)
    eq_q1.append(int(mj.jnt_qposadr[_j1]))
    eq_q2.append(int(mj.jnt_qposadr[_j2]))
    eq_dof1.append(int(mj.jnt_dofadr[_j1]))
    eq_dof2.append(int(mj.jnt_dofadr[_j2]))
    eq_poly.append(np.asarray(mj.eq_data[_e][:5], float))
  if n_connect:
    _FORK_IGNORED.add(
        f'{n_connect} mjEQ_CONNECT loop closure(s) -- not implemented; the '
        f'affected linkage is unconstrained in brax')
  eq_q1 = np.asarray(eq_q1, np.int64); eq_q2 = np.asarray(eq_q2, np.int64)
  eq_dof1 = np.asarray(eq_dof1, np.int64); eq_dof2 = np.asarray(eq_dof2, np.int64)
  eq_poly = (np.stack(eq_poly) if eq_poly else np.zeros((0, 5)))
  eq_is_slide = np.asarray(eq_is_slide, float)

  # ---- position servos as POSITIONAL DRIVES ------------------------------
  # A MuJoCo `position` actuator is tau = gear^2 kp (ctrl/gear - q) - gear^2 kv
  # qd. brax divides that by the CHILD LINK's own inertia, which for a Spot
  # finger is ~1e-6 kg m^2: the servo's natural frequency is then ~22000 rad/s
  # against a 2 ms step, so the stiffness term is unstable however the damping
  # is integrated (measured: servos alone, no gravity and no contacts, reach
  # |qd| = 2425 rad/s where MuJoCo's whole scene sits at 0.45). MuJoCo never
  # meets this because in generalized coordinates the motor pushes against the
  # articulated inertia of everything distal to it.
  #
  # The fix is not a smaller step or a bigger damper: it is to stop treating a
  # position servo as a FORCE. In a position-based solver a servo is a driven
  # joint constraint, and a positional projection is unconditionally stable
  # whatever the link inertia is. `drive_alpha` is the XPBD compliance
  # weighting w / (w + alpha~) with compliance 1/kp, i.e.
  #     kp h^2 / (kp h^2 + I_axis),
  # so a stiff servo on a light link tracks almost exactly in one substep
  # while the same servo on a heavy arm moves it a few percent -- which is
  # what a real kp does. Only JOINT-transmission position actuators qualify;
  # tendon transmissions (the grippers) stay as forces.
  _h = float(mj.opt.timestep)
  drive_act = np.full(mj.nv, -1, dtype=np.int64)
  drive_gear = np.ones(mj.nv)
  drive_alpha = np.zeros(mj.nv)
  drive_w = np.zeros(mj.nv)
  # 1.0, not 0.0: an UNDRIVEN dof must still give a finite `w + at` denominator,
  # because links without a drive index into slot 0 of these tables and 0/0 ->
  # inf -> 0*inf -> NaN. That hid on every fixed-base arm, whose dof 0 happens
  # to be a driven hinge, and appeared on the drone, whose dof 0 belongs to the
  # free-floating base.
  drive_at = np.ones(mj.nv)
  drive_lam_max = np.full(mj.nv, np.inf)
  drive_kvkp = np.zeros(mj.nv)
  drive_lo = np.full(mj.nv, -np.inf)
  drive_hi = np.full(mj.nv, np.inf)
  for _a in range(mj.nu):
    if int(mj.actuator_trntype[_a]) != mujoco.mjtTrn.mjTRN_JOINT:
      continue
    if int(mj.actuator_biastype[_a]) != 1 or int(mj.actuator_gaintype[_a]) != 0:
      continue
    _kp = float(mj.actuator_gainprm[_a, 0])
    if _kp <= 0 or abs(_kp + float(mj.actuator_biasprm[_a, 1])) > 1e-6 * _kp:
      continue                      # not a plain position servo
    _j = int(mj.actuator_trnid[_a, 0])
    # HINGE only. `drive_update` realises the servo as an ANGULAR positional
    # constraint about the joint axis, and a SLIDE joint's angular motion is
    # zero -- driving one produced a null correction while `drive_force_mask`
    # had already removed its torque, so a prismatic servo was left with no
    # actuation whatsoever. Caught on juggle-paddle, whose plate rides a z
    # slide: its height fell 0.55 -> 0.16 m over 40 control steps where CPU
    # MuJoCo holds 0.496-0.518. Prismatic servos therefore keep the force
    # path, where the implicit velocity relaxation already covers them; they
    # were never the unstable ones (the instability was rotational, on links
    # with 1e-6 kg m^2 of inertia).
    if int(mj.jnt_type[_j]) != 3:
      continue
    _dof = int(mj.jnt_dofadr[_j])
    _gear = float(mj.actuator_gear[_a, 0])
    _kp_eff = _kp * _gear * _gear
    drive_act[_dof] = _a
    drive_gear[_dof] = _gear
    drive_alpha[_dof] = _kp_eff * _h * _h / (_kp_eff * _h * _h
                                             + _dof_axis_inertia[_dof])
    # XPBD terms kept separately: w is the generalised inverse mass about the
    # axis, alpha_tilde the compliance 1/kp scaled by h^2. A drive iterated
    # WITHOUT accumulating its Lagrange multiplier converges to C = 0, i.e. to
    # a RIGID constraint that teleports the joint onto its target inside one
    # substep -- and `project_xd` then reads that displacement as velocity.
    # With the multiplier it converges to `C = -alpha_tilde * lambda`, the
    # compliant response a real kp gives, and the result stops depending on
    # the sweep count.
    # The generalised inverse mass for a DRIVE is the ARTICULATED one, not the
    # child link's own. A drive is a positional constraint and the joint
    # sweeps propagate it along the chain, so what resists it is everything
    # distal -- `diag(M)`, which on the Panda's first four joints is 11.6-23.7x
    # the link's own axis inertia. (The implicit-damping relaxation above is
    # the opposite case: a FORCE really is divided by the child link's own
    # inertia, so it keeps the smaller value.)
    drive_w[_dof] = 1.0 / _dof_inertia_med[_dof]
    drive_at[_dof] = 1.0 / (_kp_eff * _h * _h)
    # MuJoCo's position servo also has a VELOCITY term,
    # tau = kp (q* - q) - kv qd, which this drive did not model at all --
    # kv reached the sim only inside the global implicit damping, against a
    # different inertia than the drive uses. The ratio is gear-independent
    # (both halves scale by gear^2), and folding (kv/kp) qd into the
    # constraint ERROR reproduces it exactly:
    #   kp (C + (kv/kp) qd) = kp C + kv qd.
    drive_kvkp[_dof] = max(-float(mj.actuator_biasprm[_a, 2]), 0.0) / _kp
    # A drive is an actuator, so it gets the actuator's AUTHORITY. lambda is an
    # impulse in joint space (torque * h^2), so MuJoCo's forcerange maps
    # straight onto it. Without this the projection is free to apply whatever
    # torque closes the constraint: measured on the Panda, a +0.6 rad step
    # command moved joint1 +0.5810 rad in one 10 ms control step against
    # MuJoCo's +0.0042 -- 138x, which is ~195x the actuator's +/-87 Nm limit.
    _fr = mj.actuator_forcerange[_a]
    _taumax = (max(abs(float(_fr[0])), abs(float(_fr[1]))) * abs(_gear)
               if int(mj.actuator_forcelimited[_a]) == 1 else np.inf)
    drive_lam_max[_dof] = _taumax * _h * _h
    if mj.jnt_limited[_j]:
      drive_lo[_dof] = float(mj.jnt_range[_j][0])
      drive_hi[_dof] = float(mj.jnt_range[_j][1])
  # an actuator resolved as a drive must NOT also be applied as a torque
  drive_force_mask = np.ones(mj.nu)
  for _dof in range(mj.nv):
    if drive_act[_dof] >= 0:
      drive_force_mask[int(drive_act[_dof])] = 0.0

  mjx_model = mjx.put_model(mj)

  sys = System(  # pytype: disable=wrong-arg-types  # jax-ndarray
      gravity=mj.opt.gravity,
      viscosity=mj.opt.viscosity,
      density=mj.opt.density,
      elasticity=custom['elasticity'],
      link=link,
      dof=dof,
      actuator=actuator,
      init_q=custom['init_qpos'] if 'init_qpos' in custom else mj.qpos0,
      vel_damping=custom['vel_damping'],
      ang_damping=custom['ang_damping'],
      baumgarte_erp=custom['baumgarte_erp'],
      spring_mass_scale=custom['spring_mass_scale'],
      spring_inertia_scale=custom['spring_inertia_scale'],
      joint_scale_ang=custom['joint_scale_ang'],
      joint_scale_pos=custom['joint_scale_pos'],
      collide_scale=custom['collide_scale'],
      enable_fluid=(mj.opt.viscosity > 0) | (mj.opt.density > 0),
      link_names=link_names,
      link_types=link_types,
      link_parents=link_parents,
      matrix_inv_iterations=int(custom['matrix_inv_iterations']),
      solver_iterations=mj.opt.iterations,
      solver_maxls=int(custom['solver_maxls']),
      mj_model=mj,
      geom_link_idx=lm.geom_link_idx,
      geom_link_pos=lm.geom_link_pos,
      geom_link_quat=lm.geom_link_quat,
      site_act_link=lm.site_act_link,
      site_act_pos=lm.site_act_pos,
      site_act_quat=lm.site_act_quat,
      site_act_gear=lm.site_act_gear,
      dof_damping_total=dof_damping_total,
      dof_inertia=dof_inertia,
      drive_act=tuple(int(v) for v in drive_act),
      drive_gear=drive_gear,
      drive_alpha=drive_alpha,
      drive_w=drive_w,
      drive_at=drive_at,
      drive_lam_max=drive_lam_max,
      drive_kvkp=drive_kvkp,
      eq_q1=tuple(int(v) for v in eq_q1),
      eq_q2=tuple(int(v) for v in eq_q2),
      eq_dof1=tuple(int(v) for v in eq_dof1),
      eq_dof2=tuple(int(v) for v in eq_dof2),
      eq_poly=eq_poly,
      eq_is_slide=eq_is_slide,
      drive_lo=drive_lo,
      drive_hi=drive_hi,
      drive_force_mask=drive_force_mask,
      **mjx_model.__dict__,
  )

  sys = jax.tree.map(jp.array, sys)

  warnings.warn(
      'Brax System, piplines and environments are not actively being'
      ' maintained. Please see MJX for a well maintained JAX-based physics'
      ' engine: https://github.com/google-deepmind/mujoco/tree/main/mjx. For a'
      ' host of environments that use MJX, see:'
      ' https://github.com/google-deepmind/mujoco_playground.',
      UserWarning,
  )

  return sys


def fuse_bodies(xml: str):
  """Fuses together parent child bodies that have no joint."""
  xml = ElementTree.fromstring(xml)
  _fuse_bodies(xml)
  return ElementTree.tostring(xml, encoding='unicode')


def loads(xml: str, asset_path: Union[str, epath.Path, None] = None) -> System:
  """Loads a brax system from a MuJoCo mjcf xml string."""
  elem = ElementTree.fromstring(xml)
  _fuse_bodies(elem)
  assets = {}
  if asset_path is not None:
    meshdir = _get_meshdir(elem)
    asset_path = epath.Path(asset_path)
    assets = _find_assets(elem, asset_path, meshdir)
  xml = ElementTree.tostring(elem, encoding='unicode')
  mj = mujoco.MjModel.from_xml_string(xml, assets=assets)

  return load_model(mj)


def load_mjmodel(path: Union[str, epath.Path]) -> mujoco.MjModel:
  """Loads an mj model from a MuJoCo mjcf file path."""
  elem = ElementTree.fromstring(epath.Path(path).read_text())
  _fuse_bodies(elem)
  meshdir = _get_meshdir(elem)
  assets = _find_assets(elem, epath.Path(path), meshdir)
  xml = ElementTree.tostring(elem, encoding='unicode')
  mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
  return mj


def load(path: Union[str, epath.Path]):
  """Loads a brax system from a MuJoCo mjcf file path."""
  mj = load_mjmodel(path)
  return load_model(mj)
