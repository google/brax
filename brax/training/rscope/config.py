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

"""Rscope configuration."""

from pathlib import Path
import sys

from absl import flags

# Parse command-line flags.
flags.DEFINE_string(
    "logdir", "/tmp/rscope/active_run", "Path to the rscope directory."
)
flags.FLAGS(sys.argv)

# Global paths.
BASE_PATH = Path(flags.FLAGS.logdir)
META_PATH = BASE_PATH / "rscope_meta.pkl"
