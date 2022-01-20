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

"""BIG-Gym tasks."""
from typing import Dict, Any, Tuple
from brax.experimental.composer import reward_functions
from brax.experimental.composer.envs import ma_descs
from brax.experimental.composer.observers import SimObserver as so


def get_task_env_name(task_name: str, comp_name: str):
  return f'{task_name}__{comp_name}'


def get_match_env_name(task_name: str, comp1: str, comp2):
  return f'match_{task_name}__{comp1}__{comp2}'


def race(component: str, pos: Tuple[float] = (0, 0, 0), **component_params):
  return dict(
      components=dict(
          agent1=dict(
              component=component,
              component_params=component_params,
              pos=pos,
              reward_fns=dict(
                  run=dict(
                      reward_type=reward_functions.state_reward,
                      obs=lambda x: so('body', 'vel', x['root'], indices=(0,)),
                      scale=-1,
                  )),
          ),),
      global_options=dict(dt=0.02, substeps=16),
  )


def race_ma(component: str,
            opponent: str = 'ant',
            opponent_params: Dict[str, Any] = None,
            **component_params):
  """Two agents racing."""
  opponent_params = opponent_params or {}
  agent1_config = race(component, pos=(0, 1.5, 0), **component_params)
  agent2_config = race(opponent, pos=(0, -1.5, 0), **opponent_params)
  agent1_config['components']['agent2'] = agent2_config['components']['agent1']
  agent1_config['agent_groups'] = {
      k: dict(reward_agents=(k,)) for k in ['agent1', 'agent2']
  }
  return agent1_config


def onigokko_prey(component: str,
                  opponent: str = 'ant',
                  opponent_params: Dict[str, Any] = None,
                  **component_params):
  """Agent needs to escape from another agent."""
  opponent_params = opponent_params or {}
  return ma_descs.create_desc(
      main_agent=component,
      main_agent_params=component_params,
      other_agent=opponent,
      other_agent_params=opponent_params,
      task='chase')


def onigokko_predator(component: str,
                      opponent: str = 'ant',
                      opponent_params: Dict[str, Any] = None,
                      **component_params):
  """Agent needs to catch another agent."""
  opponent_params = opponent_params or {}
  return ma_descs.create_desc(
      main_agent=opponent,
      main_agent_params=opponent_params,
      other_agent=component,
      other_agent_params=component_params,
      task='chase')


def sumo(component: str,
         opponent: str = 'ant',
         opponent_params: Dict[str, Any] = None,
         **component_params):
  """Two agents sumoing."""
  opponent_params = opponent_params or {}
  return ma_descs.create_desc(
      main_agent=component,
      main_agent_params=component_params,
      other_agent=opponent,
      other_agent_params=opponent_params,
      task='sumo')


TASKS = dict(
    race=race,
    race_ma=race_ma,
    sumo=sumo,
    onigokko_predator=onigokko_predator,
    onigokko_prey=onigokko_prey,
)

SYMMETRIC_MA_TASKS = ('race_ma', 'sumo')
ASYMMETRIC_MA_TASKS = ('onigokko_prey', 'onigokko_predator')
