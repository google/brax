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
from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ShadowParameters as Shadow
from renderer import Renderer, Scene, UpAxis, transpose_for_display
import trimesh


def grid(grid_size, color):
  grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
  grid[:, :] = onp.array(color) / 255.0
  grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
  # to reverse texture along y direction
  grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
  return jp.asarray(grid)


_GROUND = grid(100, [200, 200, 200])


def _scene(sys: brax.System, state: brax.State) -> Tuple[Scene, List[int]]:
  """Converts a brax System and state to a jaxrenderer scene and instances."""
  scene = Scene()
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
      tex = col.rgba[:3].reshape((1, 1, 3))
      if isinstance(col, base.Capsule):
        half_height = col.length / 2
        scene, model = scene.add_capsule(
          radius=col.radius,
          half_height=half_height,
          up_axis=UpAxis.Z,
          diffuse_map=tex,
        )
      elif isinstance(col, base.Box):
        scene, model = scene.add_cube(
          half_extents=col.halfsize,
          diffuse_map=tex,
          texture_scaling=16.,
        )
      elif isinstance(col, base.Sphere):
        scene, model = scene.add_capsule(
          radius=col.radius,
          half_height=0.,
          up_axis=UpAxis.Z,
          diffuse_map=tex,
        )
      elif isinstance(col, base.Plane):
        tex = _GROUND
        scene, model = scene.add_cube(
          half_extents=[1000.0, 1000.0, 0.0001],
          diffuse_map=tex,
          texture_scaling=8192.,
        )
      elif isinstance(col, base.Convex):
        # convex objects are not visual
        continue
      elif isinstance(col, base.Mesh):
        tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
        mesh = RendererMesh.create(
            verts=tm.vertices,
            norms=tm.vertex_normals,
            uvs=jp.zeros((tm.vertices.shape[0], 2), dtype=int),
            faces=tm.faces,
            diffuse_map=tex,
        )
        scene, model = scene.add_model(mesh)
      else:
        raise RuntimeError(f'unrecognized collider: {type(col)}')

      i = col.link_idx if col.link_idx is not None else -1
      x = state.x.concatenate(base.Transform.zero((1,)))
      scene, instance = scene.add_object_instance(model)
      off = col.transform.pos
      pos = x.pos[i] + math.rotate(off, x.rot[i])
      rot = col.transform.rot
      rot = math.quat_mul(x.rot[i], rot)
      scene = scene.set_object_position(instance, pos)
      scene = scene.set_object_orientation(instance, rot)
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
  return [0., 0., 1.]


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
                 ssaa: int = 2,
                 shadow: Optional[Shadow] = None,
                 enable_shadow: bool = True) -> onp.ndarray:
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
    )
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
  if shadow is None and enable_shadow:
    shadow = Shadow(centre=camera.target)
  objects = [scene.objects[inst] for inst in instances]
  img = Renderer.get_camera_image(
    objects=objects,
    light=light,
    camera=camera,
    width=camera.viewWidth,
    height=camera.viewHeight,
    shadow_param=shadow,
  )
  arr = transpose_for_display(jax.lax.clamp(0., img * 255, 255.).astype(jp.uint8))
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
