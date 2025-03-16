# Copyright 2025 The Brax Authors.
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

"""Event handler for Rscope."""

from watchdog.events import FileSystemEventHandler

from brax.training.rscope.rollout import append_unroll


class MjUnrollHandler(FileSystemEventHandler):
  """Handles new .mj_unroll files appearing in the base directory."""

  def on_created(self, event):
    if not event.is_directory and event.src_path.endswith(".mj_unroll"):
      append_unroll(event.src_path)
