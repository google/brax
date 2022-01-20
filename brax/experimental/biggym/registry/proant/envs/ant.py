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

"""Ant tasks."""
from brax.experimental.biggym import registry


def Run(num_legs: int = 4):
  return dict(
      components=dict(
          agent1=dict(
              component=registry.get_comp_name('proant', 'ant'),
              component_params=dict(num_legs=num_legs),
              pos=(0, 0, 0),
              reward_fns=dict(
                  goal=dict(
                      reward_type='root_goal',
                      sdcomp='vel',
                      indices=(0, 1),
                      offset=5,
                      target_goal=(4, 0))),
          ),),
      global_options=dict(dt=0.02, substeps=16),
  )
