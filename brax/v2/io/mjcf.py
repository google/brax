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

# pylint:disable=g-multiple-import
"""Function to load MuJoCo mjcf format to Brax system."""

import itertools
from typing import Dict, List, Tuple, TypeVar, Union
import warnings
from xml.etree import ElementTree

from brax.v2.base import (
    Actuator,
    Box,
    Capsule,
    Convex,
    DoF,
    Geometry,
    Inertia,
    Link,
    Mesh,
    Motion,
    Plane,
    Sphere,
    System,
    Transform,
)
from brax.v2.geometry import mesh as geom_mesh
from etils import epath
from jax import numpy as jp
from jax.tree_util import tree_map
import mujoco
import numpy as np


Geom = TypeVar('Geom', bound=Geometry)


# map from mujoco geom_type to brax geometry string
_GEOM_TYPE_CLS = {0: Plane, 2: Sphere, 3: Capsule, 6: Box, 7: Mesh}

# map from mujoco joint type to brax joint type string
_JOINT_TYPE_STR = {
    0: 'f',  # free
    1: 'b',  # ball
    2: 'p',  # prismatic
    3: 'r',  # revolute
}

# map from mujoco bias type to brax actuator type string
_ACT_TYPE_STR = {
    0: 'm',  # motor
    1: 'p',  # position
}

_COLLIDABLES = [
    # ((Geometry, is_static), (Geometry, is_static))
    ((Sphere, False), (Plane, True)),
    ((Sphere, False), (Sphere, False)),
    ((Sphere, False), (Capsule, False)),
    ((Sphere, False), (Box, False)),
    ((Sphere, False), (Mesh, False)),
    ((Capsule, False), (Plane, True)),
    ((Capsule, False), (Capsule, False)),
    ((Capsule, False), (Box, False)),
    ((Capsule, False), (Mesh, False)),
    ((Box, False), (Plane, True)),
    ((Box, False), (Box, False)),
    ((Box, False), (Mesh, False)),
    ((Mesh, False), (Plane, True)),
    ((Mesh, False), (Mesh, False)),
]


def _fuse_bodies(elem: ElementTree.Element):
  """Fuses together parent child bodies that have no joint."""

  for child in list(elem):  # we will modify elem children, so make a copy
    if child.tag == 'body' and 'joint' not in [e.tag for e in child]:
      cpos = child.attrib.get('pos', '0 0 0')
      cpos = np.fromstring(cpos, sep=' ')
      for grandchild in child:
        # TODO: might need to offset more than just body, geom
        if grandchild.tag in ('body', 'geom') and (cpos != 0).any():
          gcpos = grandchild.attrib.get('pos', '0 0 0')
          gcpos = np.fromstring(gcpos, sep=' ') + cpos
          gcpos = ' '.join('%f' % i for i in gcpos)
          grandchild.attrib['pos'] = gcpos
        elem.append(grandchild)
      elem.remove(child)
    _fuse_bodies(child)


def _get_meshdir(elem: ElementTree.Element) -> Union[str, None]:
  """Gets the mesh directory specified by the mujoco compiler tag."""
  elem = elem.find('./mujoco/compiler')
  return elem.get('meshdir') if elem is not None else None


def _find_assets(
    elem: ElementTree.Element,
    path: Union[str, epath.Path],
    meshdir: Union[str, None] = None,
) -> Dict[str, bytes]:
  """Loads assets from an xml given a base path."""
  assets = {}
  path = epath.Path(path)
  meshdir = meshdir or _get_meshdir(elem)
  fname = elem.attrib.get('file') or elem.attrib.get('filename')
  if fname:
    dirname = path if path.is_dir() else path.parent
    assets[fname] = (dirname / (meshdir or '') / fname).read_bytes()

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


def _get_custom(mj: mujoco.MjModel) -> Dict[str, np.ndarray]:
  """Gets custom mjcf parameters for brax, with defaults."""
  default = {
      'vel_damping': (0.0, None),
      'ang_damping': (0.0, None),
      'baumgarte_erp': (0.1, None),
      'elasticity': (0.0, 'geom'),
      'constraint_stiffness': (2000.0, 'body'),
      'constraint_damping': (150.0, 'body'),
      'constraint_limit_stiffness': (1000.0, 'body'),
      'constraint_ang_damping': (0.0, 'body'),
  }

  # get numeric default overrides
  for i, ni in enumerate(mj.name_numericadr):
    nsize = mj.numeric_size[i]
    name = _get_name(mj, ni)
    val = mj.numeric_data[mj.numeric_adr[i] : mj.numeric_adr[i] + nsize]
    typ = default[name][1] if name in default else None
    default[name] = (val, typ)

  custom = {}
  for name, (val, typ) in default.items():
    size = {'body': mj.nbody, 'geom': mj.ngeom}.get(typ)
    custom[name] = np.repeat(val, size) if size else np.array(val).squeeze()

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

  return custom


def _contact_geoms(geom_a: Geom, geom_b: Geom) -> Tuple[Geom, Geom]:
  """Converts geometries for contact functions."""
  if isinstance(geom_a, Box) and isinstance(geom_b, Box):
    geom_a = geom_mesh.box_hull(geom_a)
    geom_b = geom_mesh.box_hull(geom_b)
  elif isinstance(geom_a, Box) and isinstance(geom_b, Mesh):
    geom_a = geom_mesh.box_hull(geom_a)
    geom_b = geom_mesh.convex_hull(geom_b)
  elif isinstance(geom_a, Mesh) and isinstance(geom_b, Box):
    geom_a = geom_mesh.convex_hull(geom_a)
    geom_b = geom_mesh.box_hull(geom_b)
  elif isinstance(geom_a, Mesh) and isinstance(geom_b, Mesh):
    geom_a = geom_mesh.convex_hull(geom_a)
    geom_b = geom_mesh.convex_hull(geom_b)
  elif isinstance(geom_a, Box):
    geom_a = geom_mesh.box_tri(geom_a)
  elif isinstance(geom_b, Box):
    geom_b = geom_mesh.box_tri(geom_b)

  # pad face vertices so that we can broadcast between geom_a and geom_b faces
  if isinstance(geom_a, Convex) and isinstance(geom_b, Convex):
    sa = geom_a.face.shape[-1]
    sb = geom_b.face.shape[-1]
    if sa < sb:
      face = np.pad(geom_a.face, ((0, 0), (0, sb - sa)), 'edge')
      geom_a = geom_a.replace(face=face)
    elif sb < sa:
      face = np.pad(geom_b.face, ((0, 0), (0, sa - sb)), 'edge')
      geom_b = geom_b.replace(face=face)

  return geom_a, geom_b


def _contacts_from_geoms(
    mj: mujoco.MjModel, geoms: List[Geom]
) -> List[Tuple[Geom, Geom]]:
  """Gets a list of contact geom pairs."""
  collidables = []
  for key_a, key_b in _COLLIDABLES:
    if mj.opt.collision == 1:  # only check predefined pairs in mj.pair_*
      geoms_ab = []
      for geom_id_a, geom_id_b in zip(mj.pair_geom1, mj.pair_geom2):
        geom_a, geom_b = geoms[geom_id_a], geoms[geom_id_b]
        static_a, static_b = geom_a.link_idx is None, geom_b.link_idx is None
        cls_a, cls_b = type(geom_a), type(geom_b)
        if (cls_a, static_a) == key_a and (cls_b, static_b) == key_b:
          geoms_ab.append((geom_a, geom_b))
        elif (cls_a, static_a) == key_b and (cls_b, static_b) == key_a:
          geoms_ab.append((geom_b, geom_a))
    elif key_a == key_b:  # types match, avoid double counting (a, b), (b, a)
      geoms_a = [g for g in geoms if (type(g), g.link_idx is None) == key_a]
      geoms_ab = list(itertools.combinations(geoms_a, 2))
    else:  # types don't match, take every permutation
      geoms_a = [g for g in geoms if (type(g), g.link_idx is None) == key_a]
      geoms_b = [g for g in geoms if (type(g), g.link_idx is None) == key_b]
      geoms_ab = list(itertools.product(geoms_a, geoms_b))
    if not geoms_ab:
      continue
    # filter out self-collisions
    geoms_ab = [(a, b) for a, b in geoms_ab if a.link_idx != b.link_idx]
    # convert the geometries so that they can be used for contact functions
    geoms_ab = [_contact_geoms(a, b) for a, b in geoms_ab]
    collidables.append(geoms_ab)

  # meshes with different shapes cannot be stacked, so we group meshes by vert
  # and face shape
  def key_fn(x):
    def get_key(x):
      if isinstance(x, Convex):
        return (x.vert.shape, x.face.shape, x.unique_edge.shape)
      if isinstance(x, Mesh):
        return (x.vert.shape, x.face.shape)
      return -1

    return get_key(x[0]), get_key(x[1])

  contacts = []
  for geoms_ab in collidables:
    geoms_ab = sorted(geoms_ab, key=key_fn)
    for _, g in itertools.groupby(geoms_ab, key=key_fn):
      geom_a, geom_b = tree_map(lambda *x: np.stack(x), *g)
      contacts.append((geom_a, geom_b))

  return contacts


def load_model(mj: mujoco.MjModel) -> System:
  """Creates a brax system from a MuJoCo model."""
  # do some validation up front
  if any(i not in [0, 1] for i in mj.actuator_biastype):
    raise NotImplementedError('Only actuator_biastype in [0, 1] are supported.')
  if mj.opt.integrator != 0:
    raise NotImplementedError('Only euler integration is supported.')
  if mj.opt.cone != 0:
    raise NotImplementedError('Only pyramidal cone friction is supported.')
  if not (mj.actuator_trntype == 0).all():
    raise NotImplementedError(
        'Only joint transmission types are supported for actuators.'
    )

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
  link = Link(
      transform=Transform(pos=mj.body_pos, rot=mj.body_quat),
      inertia=Inertia(
          transform=Transform(pos=mj.body_ipos, rot=mj.body_iquat),
          i=np.array([np.diag(i) for i in mj.body_inertia]),
          mass=mj.body_mass,
      ),
      invweight=mj.body_invweight0[:, 0],
      joint=Transform(pos=joint_position, rot=identity),
      constraint_stiffness=custom['constraint_stiffness'],
      constraint_damping=custom['constraint_damping'],
      constraint_limit_stiffness=custom['constraint_limit_stiffness'],
      constraint_ang_damping=custom['constraint_ang_damping'],
  )
  # skip link 0 which is the world body in mujoco
  link = tree_map(lambda x: x[1:], link)

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
  motion = tree_map(lambda *x: np.concatenate(x), *motions)

  limit = None
  if np.any(mj.jnt_limited):
    limit = tree_map(lambda *x: np.concatenate(x), *limits)
  stiffness = np.concatenate(stiffnesses)

  dof = DoF(
      motion=motion,
      armature=mj.dof_armature,
      stiffness=stiffness,
      damping=mj.dof_damping,
      limit=limit,
      invweight=mj.dof_invweight0,
  )

  # create geoms
  geoms = []
  for i, typ in enumerate(mj.geom_type):
    if typ not in _GEOM_TYPE_CLS:
      warnings.warn(f'unrecognized collider, geom_type: {typ}')
      continue

    kwargs = {
        'link_idx': mj.geom_bodyid[i] - 1 if mj.geom_bodyid[i] > 0 else None,
        'transform': Transform(pos=mj.geom_pos[i], rot=mj.geom_quat[i]),
        'friction': mj.geom_friction[i, 0],
        'elasticity': custom['elasticity'][i],
    }

    geom_cls = _GEOM_TYPE_CLS[typ]
    if geom_cls is Plane:
      geom = Plane(**kwargs)
    elif geom_cls is Sphere:
      geom = Sphere(radius=mj.geom_size[i, 0], **kwargs)
    elif geom_cls is Capsule:
      geom = Capsule(
          radius=mj.geom_size[i, 0], length=mj.geom_size[i, 1] * 2, **kwargs
      )
    elif geom_cls is Box:
      geom = Box(halfsize=mj.geom_size[i, :], **kwargs)
    elif geom_cls is Mesh:
      vert, face = _get_mesh(mj, mj.geom_dataid[i])
      geom = Mesh(vert=vert, face=face, **kwargs)
    geoms.append(geom)

  contacts = _contacts_from_geoms(mj, geoms)

  # create actuators
  ctrl_range = mj.actuator_ctrlrange
  ctrl_range[~(mj.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
  actuator = Actuator(
      gear=mj.actuator_gear[:, 0],
      ctrl_range=ctrl_range,
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

  # create non-pytree params for actuators.
  actuator_types = ''.join([_ACT_TYPE_STR[bt] for bt in mj.actuator_biastype])
  actuator_link_id = [mj.jnt_bodyid[i] - 1 for i in mj.actuator_trnid[:, 0]]
  unsupported_act_links = set(link_types[i] for i in actuator_link_id) - {
      '1',
      '2',
      '3',
  }
  if unsupported_act_links:
    raise NotImplementedError(
        f'Link types {unsupported_act_links} are not supported for actuators.'
    )
  actuator_qid = [mj.jnt_qposadr[i] for i in mj.actuator_trnid[:, 0]]
  actuator_qdid = [mj.jnt_dofadr[i] for i in mj.actuator_trnid[:, 0]]

  # mujoco stores free q in world frame, so clear link transform for free links
  if 'f' in link_types:
    free_idx = np.array([i for i, typ in enumerate(link_types) if typ == 'f'])
    link.transform.pos[free_idx] = np.zeros(3)
    link.transform.rot[free_idx] = np.array([1.0, 0.0, 0.0, 0.0])

  sys = System(
      dt=mj.opt.timestep,
      gravity=mj.opt.gravity,
      link=link,
      dof=dof,
      geoms=geoms,
      contacts=contacts,
      actuator=actuator,
      init_q=custom['init_qpos'] if 'init_qpos' in custom else mj.qpos0,
      vel_damping=custom['vel_damping'],
      ang_damping=custom['ang_damping'],
      baumgarte_erp=custom['baumgarte_erp'],
      link_names=link_names,
      link_types=link_types,
      link_parents=link_parents,
      actuator_types=actuator_types,
      actuator_link_id=actuator_link_id,
      actuator_qid=actuator_qid,
      actuator_qdid=actuator_qdid,
      solver_iterations=mj.opt.iterations,
  )

  sys = tree_map(jp.array, sys)

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
  assets = {} if asset_path is None else _find_assets(elem, asset_path)
  xml = ElementTree.tostring(elem, encoding='unicode')
  mj = mujoco.MjModel.from_xml_string(xml, assets=assets)

  return load_model(mj)


def load(path: Union[str, epath.Path]):
  """Loads a brax system from a MuJoCo mjcf file path."""
  elem = ElementTree.fromstring(epath.Path(path).read_text())
  _fuse_bodies(elem)
  assets = _find_assets(elem, path)
  xml = ElementTree.tostring(elem, encoding='unicode')
  mj = mujoco.MjModel.from_xml_string(xml, assets=assets)

  return load_model(mj)
