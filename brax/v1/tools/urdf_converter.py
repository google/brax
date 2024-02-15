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

"""Command line tool for converting URDF models to Brax."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from brax.v1.io import file
from brax.v1.physics import config_pb2
from brax.v1.tools import urdf

from google.protobuf import text_format

FLAGS = flags.FLAGS

flags.DEFINE_string('xml_model_path', None,
                    'Path of the Mujoco XML model to import.')
flags.DEFINE_string('config_path', None, 'Path of the output config.')
flags.DEFINE_bool('add_collision_pairs', False,
                  'Adds the collision pairs to the config.')
# System parameters. See brax/physics/config.proto for more information.
flags.DEFINE_float('angular_damping', -0.05,
                   'Angular velocity damping applied to each body.')
flags.DEFINE_float(
    'baumgarte_erp', 0.1,
    'How aggressively interpenetrating bodies should push away each another.')
flags.DEFINE_float('dt', 0.02, 'Time to simulate each step, in seconds.')
flags.DEFINE_float('friction', 0.6,
                   'How much surfaces in contact resist translation.')
flags.DEFINE_integer('substeps', 4,
                     'Substeps to perform to maintain numerical stability.')

flags.DEFINE_bool('add_floor', True,
                  'Whether or not to add a floor to the scene.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Read the Mujoco model.
  filename = FLAGS.xml_model_path
  with file.File(filename) as f:
    logging.info('Loading urdf model from %s', filename)
    xml_string = f.read()

  # Convert the model.
  m = urdf.UrdfConverter(
      xml_string, add_collision_pairs=FLAGS.add_collision_pairs)
  config = m.config

  # Add the default options.
  config.angular_damping = FLAGS.angular_damping
  config.baumgarte_erp = FLAGS.baumgarte_erp
  config.dt = FLAGS.dt
  config.friction = FLAGS.friction
  config.substeps = FLAGS.substeps

  if FLAGS.add_floor:
    floor = config.bodies.add()
    floor.name = 'floor'
    floor.frozen.all = True
    floor.colliders.add(plane=config_pb2.Collider.Plane())
    floor.mass = 1
    floor.inertia.MergeFrom(config_pb2.Vector3(x=1, y=1, z=1))

  # Save the config.
  if FLAGS.config_path:
    text_proto = text_format.MessageToString(config)
    with file.File(FLAGS.config_path, mode='w+') as f:
      f.write(text_proto)


if __name__ == '__main__':
  app.run(main)
