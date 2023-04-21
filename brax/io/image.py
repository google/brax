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
from typing import List, Optional, Tuple

import brax
from brax import base
from brax import math
import jax
from jax import numpy as jp
import numpy as onp
from PIL import Image
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from pytinyrenderer import TinySceneRenderer as Renderer
import trimesh


class TextureRGB888:

  def __init__(self, pixels):
    self.pixels = pixels
    self.width = int(onp.sqrt(len(pixels) / 3))
    self.height = int(onp.sqrt(len(pixels) / 3))


class Grid(TextureRGB888):

  def __init__(self, grid_size, color):
    grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.int32)
    grid[:, :] = onp.array(color)
    grid[0] = onp.zeros((grid_size, 3), dtype=onp.int32)
    grid[:, 0] = onp.zeros((grid_size, 3), dtype=onp.int32)
    super().__init__(list(grid.ravel()))


_BASIC = TextureRGB888([133, 118, 102])
_TARGET = TextureRGB888([255, 34, 34])
_GROUND = Grid(100, [200, 200, 200])


def _scene(sys: brax.System, state: brax.State) -> Tuple[Renderer, List[int]]:
  """Converts a brax System and state to a pytinyrenderer scene and instances."""
  scene = Renderer()
  instances = []

  def take_i(obj, i):
    return jax.tree_map(lambda x: jp.take(x, i, axis=0), obj)

  link_names = [n or f'link {i}' for i, n in enumerate(sys.link_names)]
  link_names += ['world']
  link_geoms = {}
  for batch in sys.geoms:
    num_geoms = len(batch.friction)
    for i in range(num_geoms):
      link_idx = -1 if batch.link_idx is None else batch.link_idx[i]
      link_geoms.setdefault(link_names[link_idx], []).append(take_i(batch, i))

  for _, geom in link_geoms.items():
    for col in geom:
      tex = TextureRGB888((col.rgba[:3] * 255).astype('uint32'))
      if isinstance(col, base.Capsule):
        half_height = col.length / 2
        model = scene.create_capsule(col.radius, half_height, 2,
                                     tex.pixels, tex.width, tex.height)
      elif isinstance(col, base.Box):
        model = scene.create_cube(col.halfsize, tex.pixels, tex.width,
                                  tex.height, 16.)
      elif isinstance(col, base.Sphere):
        model = scene.create_capsule(
            col.radius, 0, 2, tex.pixels, tex.width, tex.height
        )
      elif isinstance(col, base.Plane):
        tex = _GROUND
        model = scene.create_cube([1000.0, 1000.0, 0.0001], tex.pixels,
                                  tex.width, tex.height, 8192)
      elif isinstance(col, base.Convex):
        # convex objects are not visual
        continue
      elif isinstance(col, base.Mesh):
        tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
        vert_norm = tm.vertex_normals
        model = scene.create_mesh(
            col.vert.reshape((-1)).tolist(),
            vert_norm.reshape((-1)).tolist(),
            [0] * col.vert.shape[0] * 2,
            col.face.reshape((-1)).tolist(),
            tex.pixels,
            tex.width,
            tex.height,
            1.0,
        )
      else:
        raise RuntimeError(f'unrecognized collider: {type(col)}')

      i = col.link_idx if col.link_idx is not None else -1
      x = state.x.concatenate(base.Transform.zero((1,)))
      instance = scene.create_object_instance(model)
      off = col.transform.pos
      pos = onp.array(x.pos[i]) + math.rotate(off, x.rot[i])
      rot = col.transform.rot
      rot = math.quat_mul(x.rot[i], rot)
      scene.set_object_position(instance, list(pos))
      scene.set_object_orientation(instance, [rot[1], rot[2], rot[3], rot[0]])
      instances.append(instance)

  return scene, instances


def _eye(sys: brax.System, state: brax.State) -> List[float]:
  """Determines the camera location for a Brax system."""
  xj = state.x.vmap().do(sys.link.joint)
  dist = onp.concatenate(xj.pos[None, ...] - xj.pos[:, None, ...])
  dist = onp.linalg.norm(dist, axis=1).max()
  off = [2 * dist, -2 * dist, dist]
  return list(state.x.pos[0, :] + onp.array(off))


def _up(unused_sys: brax.System) -> List[float]:
  """Determines the up orientation of the camera."""
  return [0, 0, 1]


def get_camera(
    sys: brax.System, state: brax.State, width: int, height: int, ssaa: int = 2
) -> Camera:
  """Gets camera object."""
  eye, up = _eye(sys, state), _up(sys)
  hfov = 58.0
  vfov = hfov * height / width
  target = [state.x.pos[0, 0], state.x.pos[0, 1], 0]
  camera = Camera(
      viewWidth=width * ssaa,
      viewHeight=height * ssaa,
      position=eye,
      target=target,
      up=up,
      hfov=hfov,
      vfov=vfov)
  return camera


def render_array(sys: brax.System,
                 state: brax.State,
                 width: int,
                 height: int,
                 light: Optional[Light] = None,
                 camera: Optional[Camera] = None,
                 ssaa: int = 2) -> onp.ndarray:
  """Renders an RGB array of a brax system and QP."""
  if (len(state.x.pos.shape), len(state.x.rot.shape)) != (2, 2):
    raise RuntimeError('unexpected shape in state')
  scene, instances = _scene(sys, state)
  target = state.x.pos[0, :]
  if light is None:
    direction = [0.57735, -0.57735, 0.57735]
    light = Light(
        direction=direction,
        ambient=0.8,
        diffuse=0.8,
        specular=0.6,
        shadowmap_center=target)
  if camera is None:
    eye, up = _eye(sys, state), _up(sys)
    hfov = 58.0
    vfov = hfov * height / width
    camera = Camera(
        viewWidth=width * ssaa,
        viewHeight=height * ssaa,
        position=eye,
        target=target,
        up=up,
        hfov=hfov,
        vfov=vfov)
  img = scene.get_camera_image(instances, light, camera).rgb
  arr = onp.reshape(
      onp.array(img, dtype=onp.uint8),
      (camera.view_height, camera.view_width, -1))
  if ssaa > 1:
    arr = onp.asarray(Image.fromarray(arr).resize((width, height)))
  return arr


def render(sys: brax.System,
           states: List[brax.State],
           width: int,
           height: int,
           light: Optional[Light] = None,
           cameras: Optional[List[Camera]] = None,
           ssaa: int = 2,
           fmt='png') -> bytes:
  """Returns an image of a brax system and QP."""
  if not states:
    raise RuntimeError('must have at least one qp')
  if cameras is None:
    cameras = [None] * len(states)

  frames = [
      Image.fromarray(
          render_array(sys, state, width, height, light, camera, ssaa))
      for state, camera in zip(states, cameras)
  ]
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
