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

"""Observation descriptions."""
from typing import Union, Tuple, Any

OBS_INDICES = dict(
    vel=dict(
        ant=(13, 14),
        humanoid=(22, 23),
        halfcheetah=(11,),
        hopper=(8,),
        walker2d=(11,),
        uni_ant=(('body_vel:torso_ant1', 0), ('body_vel:torso_ant1', 1)),
        bi_ant=(('body_vel:torso_ant1', 0), ('body_vel:torso_ant2', 0)),
    ),)


def register_indices(env_name: str, indices_type: str,
                     indices: Tuple[Union[int, Any]]):
  """Register indices."""
  global OBS_INDICES
  if indices_type not in OBS_INDICES:
    OBS_INDICES[indices_type] = {}
  OBS_INDICES[indices_type][env_name] = indices


def get_indices(env_name: str, indices_type: str):
  """Get indices."""
  return OBS_INDICES.get(indices_type, {}).get(env_name, None)
