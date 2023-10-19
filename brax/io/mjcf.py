# Copyright 2023 The Brax Authors.
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
"""Function to load MuJoCo mjcf format to Brax system."""

import itertools
from typing import Dict, Optional, Tuple, Union
import warnings
from xml.etree import ElementTree

from brax import math
from brax.base import (
    Actuator,
    Box,
    Capsule,
    Cylinder,
    DoF,
    Inertia,
    Link,
    Mesh,
    Motion,
    Plane,
    Sphere,
    System,
    Transform,
)
from brax.geometry import mesh as geom_mesh
from etils import epath
import jax
from jax import numpy as jp
import mujoco
import numpy as np


def _transform_do(
    parent_pos: np.ndarray, parent_quat: np.ndarray, pos: np.ndarray,
    quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  pos = parent_pos + math.rotate_np(pos, parent_quat)
  rot = math.quat_mul_np(parent_quat, quat)
  return pos, rot


def _offset(
    elem: ElementTree.Element, parent_pos: np.ndarray, parent_quat: np.ndarray):
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
      # TODO: might need to offset more than just (body, geom)
      if grandchild.tag in ('body', 'geom') and (cpos != 0).any():
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


def _get_mesh(mj: mujoco.MjModel, i: int) -> Tuple[np.ndarray, np.ndarray]:
  """Gets mesh from mj at index i."""
  last = (i + 1) >= mj.nmesh
  face_start = mj.mesh_faceadr[i]
  face_end = mj.mesh_faceadr[i + 1] if not last else mj.mesh_face.shape[0]
  face = mj.mesh_face[face_start:face_end]

  vert_start = mj.mesh_vertadr[i]
  vert_end = mj.mesh_vertadr[i + 1] if not last else mj.mesh_vert.shape[0]
  vert = mj.mesh_vert[vert_start:vert_end]

  return vert, face


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
      'convex': (True, 'geom'),
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


def load_model(mj: mujoco.MjModel) -> System:
  """Creates a brax system from a MuJoCo model."""
  # do some validation up front
  if any(i not in [0, 1] for i in mj.actuator_biastype):
    raise NotImplementedError('Only actuator_biastype in [0, 1] are supported.')
  if any(i != 0 for i in mj.actuator_gaintype):
    raise NotImplementedError('Only actuator_gaintype in [0] is supported.')
  if mj.opt.integrator != 0:
    raise NotImplementedError('Only euler integration is supported.')
  if mj.opt.cone != 0:
    raise NotImplementedError('Only pyramidal cone friction is supported.')
  if not (mj.actuator_trntype == 0).all():
    raise NotImplementedError(
        'Only joint transmission types are supported for actuators.'
    )
  if (mj.geom_solmix[0] != mj.geom_solmix).any():
    raise NotImplementedError('geom_solmix parameter not supported.')
  if (mj.geom_priority[0] != mj.geom_priority).any():
    raise NotImplementedError('geom_priority parameter not supported.')
  q_width = {0: 7, 1: 4, 2: 1, 3: 1}
  non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
  if mj.qpos0[non_free].any():
    raise NotImplementedError(
        'The `ref` attribute on joint types is not supported.')
  if (mj.geom_fluid != 0).any():
    raise NotImplementedError('Ellipsoid fluid model not implemented.')
  if mj.opt.wind.any():
    raise NotImplementedError('option.wind is not implemented.')
  custom = _get_custom(mj)

  # create links
  joint_positions = [np.array([0.0, 0.0, 0.0])]
  for _, group in itertools.groupby(
      zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]
  ):
    position = np.array([p for _, p in group])
    if not (position == position[0]).all():
      raise RuntimeError('invalid joint stack: only one joint position allowed')
    joint_positions.append(position[0])
  joint_position = np.array(joint_positions)
  identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (mj.nbody, 1))
  link = Link(  # pytype: disable=wrong-arg-types  # jax-ndarray
      transform=Transform(pos=mj.body_pos, rot=mj.body_quat),  # pytype: disable=wrong-arg-types  # jax-ndarray
      inertia=Inertia(  # pytype: disable=wrong-arg-types  # jax-ndarray
          transform=Transform(pos=mj.body_ipos, rot=mj.body_iquat),  # pytype: disable=wrong-arg-types  # jax-ndarray
          i=np.array([np.diag(i) for i in mj.body_inertia]),
          mass=mj.body_mass,
      ),
      invweight=mj.body_invweight0[:, 0],
      joint=Transform(pos=joint_position, rot=identity),  # pytype: disable=wrong-arg-types  # jax-ndarray
      constraint_stiffness=custom['constraint_stiffness'],
      constraint_vel_damping=custom['constraint_vel_damping'],
      constraint_limit_stiffness=custom['constraint_limit_stiffness'],
      constraint_ang_damping=custom['constraint_ang_damping'],
  )
  # skip link 0 which is the world body in mujoco
  link = jax.tree_map(lambda x: x[1:], link)

  # create dofs
  mj.jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
  motions, limits, stiffnesses = [], [], []
  for typ, axis, limit, stiffness in zip(
      mj.jnt_type, mj.jnt_axis, mj.jnt_range, mj.jnt_stiffness
  ):
    if typ == 0:
      motion = Motion(ang=np.eye(6, 3, -3), vel=np.eye(6, 3))
      limit = np.array([-np.inf] * 6), np.array([np.inf] * 6)
      if stiffness > 0:
        raise RuntimeError('brax does not support stiffness for free joints')
      stiffness = np.zeros(6)
    elif typ == 1:
      motion = Motion(ang=np.eye(3), vel=np.zeros((3, 3)))
      if np.any(~np.isinf(limit)):
        raise RuntimeError('brax does not support joint ranges for ball joints')
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
      raise RuntimeError(f'invalid joint type: {typ}')
    motions.append(motion)
    limits.append(limit)
    stiffnesses.append(stiffness)
  motion = jax.tree_map(lambda *x: np.concatenate(x), *motions)

  limit = None
  if np.any(mj.jnt_limited):
    limit = jax.tree_map(lambda *x: np.concatenate(x), *limits)
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

  solver_params_geom = np.concatenate((mj.geom_solref, mj.geom_solimp), axis=1)
  # group geoms so that they can be stacked.  two geoms can be stacked if:
  # - they have the same type
  # - their fields have the same shape (e.g. Mesh verts might vary)
  # - they have the same mask
  key_fn = lambda g, m: (jax.tree_map(np.shape, g), m)
  geom_groups = {}
  for i, typ in enumerate(mj.geom_type):
    rgba = mj.geom_rgba[i]
    if (mj.geom_rgba[i] == [0.5, 0.5, 0.5, 1.0]).all():
      # convert the default mjcf color to brax default color
      rgba = np.array([0.4, 0.33, 0.26, 1.0])
    kwargs = {
        'link_idx': mj.geom_bodyid[i] - 1 if mj.geom_bodyid[i] > 0 else None,
        'transform': Transform(pos=mj.geom_pos[i], rot=mj.geom_quat[i]),
        'friction': mj.geom_friction[i, 0],
        'elasticity': custom['elasticity'][i],
        'solver_params': solver_params_geom[i],
        'rgba': rgba,
    }
    mask = mj.geom_contype[i] | mj.geom_conaffinity[i] << 32
    if typ == 0:  # Plane
      geom = Plane(**kwargs)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    elif typ == 5:  # Cylinder
      radius, halflength = mj.geom_size[i, 0:2]
      if halflength > 0.001 and mask > 0:
        # TODO: support cylinders with depth.
        raise NotImplementedError(
            'Cylinders of half-length>0.001 are not supported for collision.')
      geom = Cylinder(radius=radius, length=halflength * 2, **kwargs)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    elif typ == 2:  # Sphere
      geom = Sphere(radius=mj.geom_size[i, 0], **kwargs)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    elif typ == 3:  # Capsule
      radius, halflength = mj.geom_size[i, 0:2]
      geom = Capsule(radius=radius, length=halflength * 2, **kwargs)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    elif typ == 6:  # Box
      geom = Box(halfsize=mj.geom_size[i, :], **kwargs)
      geom_groups.setdefault(key_fn(geom, 0), []).append(geom)  # visual only
      if custom['convex'][i]:
        geom = geom_mesh.convex_hull(geom)
      else:
        geom = geom_mesh.box_tri(geom)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    elif typ == 7:  # Mesh
      vert, face = _get_mesh(mj, mj.geom_dataid[i])
      geom = Mesh(vert=vert, face=face, **kwargs)  # pytype: disable=wrong-arg-types
      if custom['convex'][i]:
        geom_groups.setdefault(key_fn(geom, 0), []).append(geom)  # visual only
        geom = geom_mesh.convex_hull(geom)
      geom_groups.setdefault(key_fn(geom, mask), []).append(geom)
    else:
      warnings.warn(f'unrecognized collider, geom_type: {typ}')
      continue

  geoms = [
      jax.tree_map(lambda *x: np.stack(x), *g) for g in geom_groups.values()
  ]
  geom_masks = [m for _, m in geom_groups.keys()]

  # create actuators
  ctrl_range = mj.actuator_ctrlrange
  ctrl_range[~(mj.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
  force_range = mj.actuator_forcerange
  force_range[~(mj.actuator_forcelimited == 1), :] = np.array([-np.inf, np.inf])
  q_id = np.array([mj.jnt_qposadr[i] for i in mj.actuator_trnid[:, 0]])
  qd_id = np.array([mj.jnt_dofadr[i] for i in mj.actuator_trnid[:, 0]])
  bias_q = mj.actuator_biasprm[:, 1] * (mj.actuator_biastype != 0)
  bias_qd = mj.actuator_biasprm[:, 2] * (mj.actuator_biastype != 0)

  # TODO: might be nice to add actuator names for debugging
  actuator = Actuator(  # pytype: disable=wrong-arg-types
      q_id=q_id,
      qd_id=qd_id,
      gain=mj.actuator_gainprm[:, 0],
      gear=mj.actuator_gear[:, 0],
      ctrl_range=ctrl_range,
      force_range=force_range,
      bias_q=bias_q,
      bias_qd=bias_qd,
  )

  # create non-pytree params.  these do not live on device directly, and they
  # cannot be differentiated, but they do change the emitted control flow
  link_names = [_get_name(mj, i) for i in mj.name_bodyadr[1:]]
  # convert stacked joints to 1, 2, or 3
  link_types = ''
  for _, group in itertools.groupby(
      zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]
  ):
    typs = [t for _, t in group]
    if len(typs) == 1 and typs[0] == 0:  # free
      typ = 'f'
    elif 0 in typs:
      raise RuntimeError('invalid joint stack: cannot stack free joints')
    elif 1 in typs:
      raise NotImplementedError('ball joints not supported')
    else:
      typ = str(len(typs))
    link_types += typ
  link_parents = tuple(mj.body_parentid - 1)[1:]

  # mujoco stores free q in world frame, so clear link transform for free links
  if 'f' in link_types:
    free_idx = np.array([i for i, typ in enumerate(link_types) if typ == 'f'])
    link.transform.pos[free_idx] = np.zeros(3)
    link.transform.rot[free_idx] = np.array([1.0, 0.0, 0.0, 0.0])

  sys = System(  # pytype: disable=wrong-arg-types  # jax-ndarray
      dt=mj.opt.timestep,
      gravity=mj.opt.gravity,
      viscosity=mj.opt.viscosity,
      density=mj.opt.density,
      link=link,
      dof=dof,
      geoms=geoms,
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
      geom_masks=geom_masks,
      link_names=link_names,
      link_types=link_types,
      link_parents=link_parents,
      matrix_inv_iterations=int(custom['matrix_inv_iterations']),
      solver_iterations=mj.opt.iterations,
      solver_maxls=int(custom['solver_maxls']),
  )

  sys = jax.tree_map(jp.array, sys)

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


def load_mjmodel(path: Union[str, epath.Path]):
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
