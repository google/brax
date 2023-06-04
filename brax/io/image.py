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

"""Exports a system config and state as an image."""

import io
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence

import brax
from brax import base
from brax import math
import jax
from jax import numpy as jp
import numpy as onp
from PIL import Image
from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, UpAxis, create_capsule, create_cube, transpose_for_display
import trimesh


def grid(grid_size, color):
  grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
  grid[:, :] = onp.array(color) / 255.0
  grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
  # to reverse texture along y direction
  grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
  return jp.asarray(grid)


_GROUND = grid(100, [200, 200, 200])

class Obj(NamedTuple):
  """An object to be rendered in the scene.

  Assume the system is unchanged throughout the rendering.

  col is accessed from the batched geoms `sys.geoms`, representing one geom.
  """
  instance: Instance
  """An instance to be rendered in the scene, defined by jaxrenderer."""
  link_idx: int
  """col.link_idx if col.link_idx is not None else -1"""
  off: jp.ndarray
  """col.transform.rot"""
  rot: jp.ndarray
  """col.transform.rot"""

def _build_objects(sys: brax.System) -> List[Obj]:
  """Converts a brax System to a list of Obj."""
  objs: List[Obj] = []

  def take_i(obj, i):
    return jax.tree_map(lambda x: jp.take(x, i, axis=0), obj)

  link_names: List[str]
  link_names = [n or f'link {i}' for i, n in enumerate(sys.link_names)]
  link_names += ['world']
  link_geoms: Dict[str, List[Any]] = {}
  for batch in sys.geoms:
    num_geoms = len(batch.friction)
    for i in range(num_geoms):
      link_idx = -1 if batch.link_idx is None else batch.link_idx[i]
      link_geoms.setdefault(link_names[link_idx], []).append(take_i(batch, i))

  for _, geom in link_geoms.items():
    for col in geom:
      tex = col.rgba[:3].reshape((1, 1, 3))
      # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
      specular_map = jax.lax.full(tex.shape[:2], 2.0)

      if isinstance(col, base.Capsule):
        half_height = col.length / 2
        model = create_capsule(
          radius=col.radius,
          half_height=half_height,
          up_axis=UpAxis.Z,
          diffuse_map=tex,
          specular_map=specular_map,
        )
      elif isinstance(col, base.Box):
        model = create_cube(
          half_extents=col.halfsize,
          diffuse_map=tex,
          texture_scaling=jp.array(16.),
          specular_map=specular_map,
        )
      elif isinstance(col, base.Sphere):
        model = create_capsule(
          radius=col.radius,
          half_height=jp.array(0.),
          up_axis=UpAxis.Z,
          diffuse_map=tex,
          specular_map=specular_map,
        )
      elif isinstance(col, base.Plane):
        tex = _GROUND
        model = create_cube(
          half_extents=jp.array([1000.0, 1000.0, 0.0001]),
          diffuse_map=tex,
          texture_scaling=jp.array(8192.),
          specular_map=specular_map,
        )
      elif isinstance(col, base.Convex):
        # convex objects are not visual
        continue
      elif isinstance(col, base.Mesh):
        tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
        model = RendererMesh.create(
            verts=tm.vertices,
            norms=tm.vertex_normals,
            uvs=jp.zeros((tm.vertices.shape[0], 2), dtype=int),
            faces=tm.faces,
            diffuse_map=tex,
        )
      else:
        raise RuntimeError(f'unrecognized collider: {type(col)}')

      i: int = col.link_idx if col.link_idx is not None else -1
      instance = Instance(model=model)
      off = col.transform.pos
      rot = col.transform.rot
      obj = Obj(instance=instance, link_idx=i, off=off, rot=rot)

      objs.append(obj)

  return objs

def _with_state(objs: Iterable[Obj], x: brax.Transform) -> List[Instance]:
  """x must has at least 1 element. This can be ensured by calling
    `x.concatenate(base.Transform.zero((1,)))`. x is `state.x`.

    This function does not modify any inputs, rather, it produces a new list of
    `Instance`s.
  """
  if (len(x.pos.shape), len(x.rot.shape)) != (2, 2):
    raise RuntimeError('unexpected shape in state')

  instances: List[Instance] = []
  for obj in objs:
    i = obj.link_idx
    pos = x.pos[i] + math.rotate(obj.off, x.rot[i])
    rot = math.quat_mul(x.rot[i], obj.rot)
    instance = obj.instance
    instance = instance.replace_with_position(pos)
    instance = instance.replace_with_orientation(rot)
    instances.append(instance)

  return instances


def _eye(sys: brax.System, state: brax.State) -> List[float]:
  """Determines the camera location for a Brax system."""
  xj = state.x.vmap().do(sys.link.joint)
  dist = onp.concatenate(xj.pos[None, ...] - xj.pos[:, None, ...])
  dist = onp.linalg.norm(dist, axis=1).max()
  off = [2 * dist, -2 * dist, dist]
  return list(state.x.pos[0, :] + onp.array(off))


def _up(unused_sys: brax.System) -> List[float]:
  """Determines the up orientation of the camera."""
  return [0., 0., 1.]


def get_target(state: brax.State) -> jp.ndarray:
  """Gets target of camera."""
  return jp.array([state.x.pos[0, 0], state.x.pos[0, 1], 0])

def get_camera(
    sys: brax.System, state: brax.State, width: int, height: int, ssaa: int = 2
) -> Camera:
  """Gets camera object."""
  eye, up = _eye(sys, state), _up(sys)
  hfov = 58.0
  vfov = hfov * height / width
  target = get_target(state)
  camera = Camera(
      viewWidth=width * ssaa,
      viewHeight=height * ssaa,
      position=eye,
      target=target,
      up=up,
      hfov=hfov,
      vfov=vfov)
  return camera


@jax.default_matmul_precision("float32")
def render_instances(
  instances: Sequence[Instance],
  width: int,
  height: int,
  camera: Camera,
  light: Optional[Light] = None,
  shadow: Optional[Shadow] = None,
  camera_target: Optional[jp.ndarray] = None,
  enable_shadow: bool = True,
) -> jp.ndarray:
  """Renders an RGB array of sequence of instances.

  Rendered result is not transposed with `transpose_for_display`; it is in
  floating numbers in [0, 1], not `uint8` in [0, 255].
  """
  if light is None:
    direction = jp.array([0.57735, -0.57735, 0.57735])
    light = Light(
        direction=direction,
        ambient=0.8,
        diffuse=0.8,
        specular=0.6,
    )
  if shadow is None and enable_shadow:
    assert camera_target is not None, 'camera_target is None'
    shadow = Shadow(centre=camera_target)
  elif not enable_shadow:
    shadow = None

  img = Renderer.get_camera_image(
    objects=instances,
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow,
  )
  arr = jax.lax.clamp(0., img, 1.)

  return arr

@jax.default_matmul_precision("float32")
def render_array(sys: brax.System,
                 state: brax.State,
                 width: int,
                 height: int,
                 light: Optional[Light] = None,
                 camera: Optional[Camera] = None,
                 ssaa: int = 2,
                 shadow: Optional[Shadow] = None,
                 enable_shadow: bool = True) -> onp.ndarray:
  """Renders an RGB array of a brax system and QP."""
  objs = _build_objects(sys)
  instances = _with_state(objs, state.x.concatenate(base.Transform.zero((1,))))

  if camera is None:
    camera = get_camera(sys, state, width, height, ssaa)

  img = render_instances(
    instances=instances,
    width=width * ssaa,
    height=height * ssaa,
    camera=camera,
    light=light,
    shadow=shadow,
    camera_target=get_target(state),
    enable_shadow=enable_shadow,
  )
  arr = transpose_for_display((img * 255).astype(jp.uint8))
  if ssaa > 1:
    arr = onp.asarray(Image.fromarray(onp.asarray(arr)).resize((width, height)))
  else:
    arr = onp.asarray(arr)
  return arr


def render(sys: brax.System,
           states: List[brax.State],
           width: int,
           height: int,
           light: Optional[Light] = None,
           cameras: Optional[List[Camera]] = None,
           ssaa: int = 2,
           fmt='png',
           shadow: Optional[Shadow] = None,
           enable_shadow: bool = True) -> bytes:
  """Returns an image of a brax system and QP."""
  if not states:
    raise RuntimeError('must have at least one qp')
  if cameras is None:
    cameras = [get_camera(sys, state, width, height, ssaa) for state in states]

  objs = _build_objects(sys)
  _render = jax.jit(
    render_instances,
    static_argnames=("width", "height", "enable_shadow"),
  )
  frames: List[Image.Image] = []
  for state, camera in zip(states, cameras):
    x = state.x.concatenate(base.Transform.zero((1,)))
    instances = _with_state(objs, x)
    target = state.x.pos[0, :]
    img = _render(
      instances=instances,
      width=width * ssaa,
      height=height * ssaa,
      camera=camera,
      light=light,
      shadow=shadow,
      camera_target=get_target(state),
      enable_shadow=enable_shadow,
    )
    arr = transpose_for_display((img * 255).astype(jp.uint8))
    frame = Image.fromarray(onp.asarray(arr))
    if ssaa > 1:
      frame = frame.resize((width, height))

    frames.append(frame)

  f = io.BytesIO()
  if len(frames) == 1:
    frames[0].save(f, format=fmt)
  else:
    frames[0].save(
        f,
        format=fmt,
        append_images=frames[1:],
        save_all=True,
        duration=sys.dt * 1000,
        loop=0)
  return f.getvalue()
