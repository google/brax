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

"""App for brax visualization."""

from wsgiref import simple_server
from wsgiref import validate

from absl import app
from absl import flags
from brax.v2.io import html
from etils import epath
import flask
from flask import jsonify
from flask import send_from_directory
import flask_cors


PORT = flags.DEFINE_integer(
    name='port', default=8080, help='Port to run server on'
)
DEBUG = flags.DEFINE_boolean(
    name='debug', default=False, help='Debug the server.'
)

flask_app = flask.Flask(__name__)
flask_cors.CORS(flask_app)


@flask_app.route('/', methods=['GET'])
def index():
  return jsonify(success=True)


@flask_app.route('/favicon.ico')
def favicon():
  """Serves the brax favicon."""
  path = epath.Path(flask_app.root_path)
  return send_from_directory(str(path), 'favicon.ico')


@flask_app.route('/js/<path:path>', methods=['GET'])
def js(path):
  """Serves files from the js/ directory."""
  path = epath.Path(flask_app.root_path) / 'js' / path
  response = flask.Response(path.read_text(), mimetype='text/javascript')
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response


def _render_file(path: epath.Path) -> str:
  system = path.read_text()
  return html.render_from_json(
      system, height='100vh', colab=False, base_url='/js/viewer.js'
  )


@flask_app.route('/file/<path:path>', methods=['GET'])
def from_file(path):
  """Renders a json encoded brax system from a local file path."""
  return _render_file(epath.Path(path))


def main(_):
  if DEBUG.value:
    flask_app.run(
        host='localhost', port=PORT.value, use_reloader=True, debug=True
    )
    return

  server = simple_server.make_server(
      'localhost', PORT.value, validate.validator(flask_app)
  )
  server.serve_forever()


if __name__ == '__main__':
  app.run(main)
