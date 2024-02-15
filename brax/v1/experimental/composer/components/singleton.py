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

"""Single primitive object, e.g. block, capsule, sphere."""
from brax.v1.experimental.composer import component_editor

ROOT = 'object'


def get_specs(size: float = 0.25,
              collider_type: str = 'sphere',
              no_obs: bool = False):
  """Get system config."""
  if collider_type == 'sphere':
    assert isinstance(size, (float, int)), size
    collider = dict(sphere=dict(radius=size))
    scale = size
  elif collider_type == 'capsule':
    assert len(size) == 2, size
    collider = dict(capsule=dict(radius=size[0], length=size[1]))
    scale = size[0]
  else:
    raise NotImplementedError(collider_type)
  config_json = dict(bodies=[
      dict(
          name=ROOT,
          colliders=[collider],
          inertia=dict(x=1.0, y=1.0, z=1.0),
          mass=1.0 * scale**3)
  ])
  message_str = component_editor.json2message_str(config_json)
  return dict(
      message_str=message_str,
      collides=(ROOT,),
      root=ROOT,
      term_fn=None,
      observers=('qp',) if not no_obs else ())
