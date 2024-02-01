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

"""Print the Center of Mass of a mesh file."""

from collections.abc import Sequence

from absl import app
from absl import flags
from brax.v1.io import file
from trimesh.exchange.load import load_mesh

_PATH = flags.DEFINE_string('path', None, 'Path to mesh file.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with file.File(_PATH.value, 'rb') as f:
    mesh = load_mesh(f, file_type=_PATH.value)
  if not mesh.is_watertight:
    raise AssertionError('Mesh must be watertight.')
  print(
      f'{{ x: {mesh.center_mass[0]} y: {mesh.center_mass[1]}  z: {mesh.center_mass[2]} }}'
  )


if __name__ == '__main__':
  app.run(main)
