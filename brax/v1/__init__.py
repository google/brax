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

"""Import top-level classes and functions here for encapsulation/clarity."""

__version__ = '0.1.2'

import warnings

from brax.v1.physics.base import Info
from brax.v1.physics.base import QP
from brax.v1.physics.config_pb2 import Config
from brax.v1.physics.system import System

warnings.warn(
    'brax.v1 is deprecated and will be removed in a future release.',
    DeprecationWarning)
