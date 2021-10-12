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

"""Exports a system config and state as an image."""

import io
from typing import List, Optional, Tuple

import brax
import numpy as onp
from PIL import Image
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from pytinyrenderer import TinySceneRenderer as Renderer


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


_BASIC = TextureRGB888([102, 85, 68])
_TARGET = TextureRGB888([255, 34, 34])
_GROUND = Grid(100, [200, 200, 200])


def _qmult(u, v):
  """Multiplies two quaternions.

  Args:
    u: jnp.ndarray (4) (w,x,y,z)
    v: jnp.ndarray (4) (w,x,y,z)

  Returns:
    A quaternion u*v.
  """
  return onp.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def _euler_to_quat(v):
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = onp.cos(onp.array([v.x, v.y, v.z]) * onp.pi / 360)
  s1, s2, s3 = onp.sin(onp.array([v.x, v.y, v.z]) * onp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return onp.array([w, x, y, z])


def _rotate(vec: onp.ndarray, quat: onp.ndarray):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: jnp.ndarray (3)
    quat: jnp.ndarray (4) (w,x,y,z)

  Returns:
    A jnp.ndarry(3) containing vec rotated by quat.
  """

  u = quat[1:]
  s = quat[0]
  return 2 * (onp.dot(u, vec) *
              u) + (s * s - onp.dot(u, u)) * vec + 2 * s * onp.cross(u, vec)


def _scene(sys: brax.System, qp: brax.QP) -> Tuple[Renderer, List[int]]:
  """Converts a brax System and qp to a pytinyrenderer scene and instances."""
  scene = Renderer()
  instances = []
  for i, body in enumerate(sys.config.bodies):
    tex = _TARGET if body.name.lower() == 'target' else _BASIC
    for col in body.colliders:
      if col.WhichOneof('type') == 'capsule':
        half_height = col.capsule.length / 2 - col.capsule.radius
        model = scene.create_capsule(col.capsule.radius, half_height, 2,
                                     tex.pixels, tex.width, tex.height)
      elif col.WhichOneof('type') == 'box':
        hs = col.box.halfsize
        model = scene.create_cube([hs.x, hs.y, hs.z], _BASIC.pixels,
                                  tex.width, tex.height, 16.)
      elif col.WhichOneof('type') == 'sphere':
        model = scene.create_capsule(col.sphere.radius, 0, 2, tex.pixels,
                                     tex.width, _BASIC.height)
      elif col.WhichOneof('type') == 'plane':
        tex = _GROUND
        model = scene.create_cube([1000.0, 1000.0, 0.0001], tex.pixels,
                                  tex.width, tex.height, 8192)
      else:
        unrecognized_type = col.WhichOneof('type')
        raise RuntimeError(f'unrecognized collider: {unrecognized_type}')

      instance = scene.create_object_instance(model)
      off = onp.array([col.position.x, col.position.y, col.position.z])
      pos = onp.array(qp.pos[i]) + _rotate(off, qp.rot[i])
      rot = _qmult(qp.rot[i], _euler_to_quat(col.rotation))
      scene.set_object_position(instance, list(pos))
      scene.set_object_orientation(instance, [rot[1], rot[2], rot[3], rot[0]])
      instances.append(instance)

  return scene, instances


def _eye(sys: brax.System, qp: brax.QP) -> List[float]:
  """Determines the camera location for a Brax system."""
  d = {}
  for joint in sys.config.joints:
    if joint.parent not in d:
      d[joint.parent] = []
    po, co = joint.parent_offset, joint.child_offset
    off = onp.array([po.x, po.y, po.z]) - onp.array([co.x, co.y, co.z])
    d[joint.parent].append((joint.child, onp.linalg.norm(off)))

  def max_dist(parent):
    ret = 0
    for child, dist in d.get(parent, []):
      dist += max_dist(child)
      if dist > ret:
        ret = dist
    return ret

  # TODO: improve on this rough approximation of the bounding box
  dist = max([max_dist(p) for p in d]) * 3
  off = [dist * .5, -dist, dist * .5]

  if sys.config.frozen.position.x:
    off = [dist, 0, 0]
  elif sys.config.frozen.position.y:
    off = [0, -dist, 0]
  elif sys.config.frozen.position.z:
    off = [0, 0, dist * 2]

  return list(qp.pos[0] + onp.array(off))


def _up(sys: brax.System) -> List[float]:
  """Determines the up orientation of the camera."""
  if sys.config.frozen.position.z:
    return [0, 1, 0]
  else:
    return [0, 0, 1]


def render_array(sys: brax.System, qp: brax.QP, width: int, height: int,
                 light: Optional[Light] = None, camera: Optional[Camera] = None,
                 ssaa: int = 2) -> onp.ndarray:
  """Renders an RGB array of a brax system and QP."""
  if (len(qp.pos.shape), len(qp.rot.shape)) != (2, 2):
    raise RuntimeError('unexpected shape in qp')

  scene, instances = _scene(sys, qp)
  target = [qp.pos[0, 0], qp.pos[0, 1], 0]
  if light is None:
    direction = [0.57735, -0.57735, 0.57735]
    light = Light(direction=direction, ambient=0.8, diffuse=0.8, specular=0.6,
                  shadowmap_center=target)
  if camera is None:
    eye, up = _eye(sys, qp), _up(sys)
    camera = Camera(viewWidth=width * ssaa, viewHeight=height * ssaa,
                    position=eye, target=target, up=up)
  img = scene.get_camera_image(instances, light, camera).rgb
  arr = onp.reshape(
      onp.array(img, dtype=onp.uint8),
      (camera.view_height, camera.view_width, -1))
  if ssaa > 1:
    arr = onp.asarray(Image.fromarray(arr).resize((width, height)))
  return arr


def render(sys: brax.System, qps: List[brax.QP], width: int, height: int,
           light: Optional[Light] = None,
           cameras: Optional[List[Camera]] = None,
           ssaa: int = 2, fmt='png') -> bytes:
  """Returns an image of a brax system and QP."""
  if not qps:
    raise RuntimeError('must have at least one qp')
  if cameras is None:
    cameras = [None] * len(qps)

  frames = [
      Image.fromarray(
          render_array(sys, qp, width, height, light, camera, ssaa))
      for qp, camera in zip(qps, cameras)
  ]
  f = io.BytesIO()
  if len(frames) == 1:
    frames[0].save(f, format=fmt)
  else:
    frames[0].save(f, format=fmt, append_images=frames[1:], save_all=True,
                   duration=sys.config.dt * 1000, loop=0)
  return f.getvalue()

