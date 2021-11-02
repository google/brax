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

"""multi-agent ants environments."""
import functools
import itertools
from typing import Any, Dict, Sequence
from brax.experimental.composer import component_editor
from brax.experimental.composer import reward_functions
from brax.experimental.composer.composer_utils import merge_desc
from brax.experimental.composer.observers import SimObserver as so
import numpy as np

MAX_DIST = 20
MIN_DIST = 0.5


def get_n_agents_desc(agents: Sequence[str],
                      agents_params: Sequence[str] = None,
                      init_r: float = 2):
  """Get n agents."""
  angles = np.linspace(0, 2 * np.pi, len(agents) + 1)
  agents_params = agents_params or ([None] * len(agents))
  components = {}
  edges = {}
  for i, (angle, agent,
          agent_params) in enumerate(zip(angles[:-1], agents, agents_params)):
    pos = (np.cos(angle) * init_r, np.sin(angle) * init_r, 0)
    components[f'agent{i}'] = dict(component=agent, pos=pos)
    if agent_params:
      components[f'agent{i}'].update(dict(component_params=agent_params))
  for k1, k2 in itertools.combinations(list(components), 2):
    if k1 == k2:
      continue
    k1, k2 = sorted([k1, k2])  # ensure the name is always sorted in order
    edge_name = component_editor.concat_comps(k1, k2)
    edges[edge_name] = dict(
        extra_observers=[dict(observer_type='root_vec', indices=(0, 1))])
  return dict(components=components, edges=edges)


def add_follow(env_desc: Dict[str, Any], leader_vel: float = 3.0):
  """Add follow task."""
  agent_groups = {}
  components = {}
  edges = {}
  agents = sorted(env_desc['components'])
  leader, followers = agents[0], agents[1:]
  # leader aims to run at a specific velocity
  components[leader] = dict(
      reward_fns=dict(
          goal=dict(
              reward_type='root_goal',
              sdcomp='vel',
              indices=(0, 1),
              offset=leader_vel + 2,
              target_goal=(leader_vel, 0))))
  agent_groups[leader] = dict(reward_agents=(leader,))
  # follower follows
  for agent in followers:
    edge_name = component_editor.concat_comps(agent, leader)
    edges[edge_name] = dict(
        reward_fns=dict(
            dist=dict(
                reward_type='root_dist',
                max_dist=MAX_DIST,
                offset=MAX_DIST + 1)))
    agent_groups[agent] = dict(reward_names=(('dist', agent, leader),))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, components=components, edges=edges))
  return env_desc


def add_chase(env_desc: Dict[str, Any]):
  """Add chase task."""
  agent_groups = {}
  components = {}
  edges = {}
  agents = sorted(env_desc['components'])
  prey, predators = agents[0], agents[1:]
  prey_rewards = ()
  run_reward = dict(
      reward_type=reward_functions.norm_reward,
      obs=lambda x: so('body', 'vel', x['root'], indices=(0, 1)),
      scale=-1)
  for agent in predators:
    edge_name = component_editor.concat_comps(agent, prey)
    edges[edge_name] = dict(
        reward_fns=dict(
            # predators aim to chase the prey
            chase=dict(
                reward_type='root_dist',
                offset=MAX_DIST + 1,
                min_dist=MIN_DIST,
                done_bonus=1000 * MAX_DIST),
            # prey aims to run away from all predators
            escape=dict(
                reward_type='root_dist',
                scale=-1,
                max_dist=MAX_DIST,
                done_bonus=1000 * MAX_DIST,
            ),
        ))
    prey_rewards += (('escape', agent, prey),)
    # add velocity bonus for each agent
    components[agent] = dict(reward_fns=dict(run=run_reward))
    agent_groups[agent] = dict(
        reward_names=(('chase', agent, prey), ('run', agent)))
  # add velocity bonus for each agent
  components[prey] = dict(reward_fns=dict(run=run_reward))
  agent_groups[prey] = dict(reward_names=prey_rewards + (('run', prey),))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc


def create_desc(main_agent: str = 'ant',
                other_agent: str = 'ant',
                main_agent_params: Dict[str, Any] = None,
                other_agent_params: Dict[str, Any] = None,
                num_agents: int = 2,
                task: str = 'follow',
                init_r: float = 2.,
                **kwargs):
  """Creat env_desc."""
  if main_agent_params or other_agent_params:
    agents_params = [main_agent_params] + [other_agent_params] * (
        num_agents - 1)
  else:
    agents_params = None
  env_desc = get_n_agents_desc(
      agents=[main_agent] + [other_agent] * (num_agents - 1),
      agents_params=agents_params,
      init_r=init_r)

  return dict(
      follow=add_follow, chase=add_chase)[task](
          env_desc=env_desc, **kwargs)


ENV_DESCS = dict(
    follow=functools.partial(create_desc, task='follow'),
    chase=functools.partial(create_desc, task='chase'),
)
