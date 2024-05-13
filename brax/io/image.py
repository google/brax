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

"""Exports a system config and state as an image."""

import io
from typing import List, Optional, Sequence, Union

import brax
from brax import base
import mujoco
import numpy as np
from PIL import Image


def render_array(
    sys: brax.System,
    trajectory: Union[List[base.State], base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
) -> Union[Sequence[np.ndarray], np.ndarray]:
  """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
  renderer = mujoco.Renderer(sys.mj_model, height=height, width=width)
  camera = camera or -1

  def get_image(state: base.State):
    d = mujoco.MjData(sys.mj_model)
    d.qpos, d.qvel = state.q, state.qd
    mujoco.mj_forward(sys.mj_model, d)
    renderer.update_scene(d, camera=camera)
    return renderer.render()

  if isinstance(trajectory, list):
    return [get_image(s) for s in trajectory]

  return get_image(trajectory)


def render(
    sys: brax.System,
    trajectory: List[base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    fmt: str = 'png',
) -> bytes:
  """Returns an image of a brax System."""
  if not trajectory:
    raise RuntimeError('must have at least one state')

  frames = render_array(sys, trajectory, height, width, camera)
  frames = [Image.fromarray(image) for image in frames]

  f = io.BytesIO()
  if len(frames) == 1:
    frames[0].save(f, format=fmt)
  else:
    frames[0].save(
        f,
        format=fmt,
        append_images=frames[1:],
        save_all=True,
        duration=sys.opt.timestep * 1000,
        loop=0,
    )

  return f.getvalue()
