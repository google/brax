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

"""BIG-Gym tasks."""
from typing import Dict, Any


def Run(component: str, component_params: Dict[str, Any]):
  return dict(
      components=dict(
          agent1=dict(
              component=component,
              component_params=component_params,
              pos=(0, 0, 0),
              reward_fns=dict(
                  goal=dict(
                      reward_type='root_goal',
                      sdcomp='vel',
                      indices=(0, 1),
                      offset=5,
                      target_goal=(4, 0))),
          ),),
      global_options=dict(dt=0.2, substeps=16),
  )


TASKS = dict(run=Run,)
