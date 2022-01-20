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

"""multi-agent environments."""
import functools
import itertools
from typing import Any, Dict, Sequence
from brax import jumpy as jp
from brax import math
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
                reward_type='root_dist', max_dist=MAX_DIST, offset=MAX_DIST +
                1)))
    agent_groups[agent] = dict(reward_names=(('dist', agent, leader),))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, components=components, edges=edges))
  return env_desc


def get_run_reward(scale: float = 1.0):
  return dict(
      reward_type=reward_functions.norm_reward,
      obs=lambda x: so('body', 'vel', x['root'], indices=(0, 1)),
      scale=-scale)


def add_chase(env_desc: Dict[str, Any]):
  """Add chase task."""
  agents = sorted(env_desc['components'])
  agent_groups = {agent: {'reward_names': ()} for agent in agents}
  components = {agent: {'reward_fns': {}} for agent in agents}
  edges = {}
  prey, predators = agents[0], agents[1:]
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
    agent_groups[prey]['reward_names'] += (('escape', agent, prey),)
    agent_groups[agent]['reward_names'] += (('chase', agent, prey),)
  for agent in agents:
    # add velocity bonus for each agent
    components[agent]['reward_fns'].update(dict(run=get_run_reward()))
    agent_groups[agent]['reward_names'] += (('run', agent),)
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc


def get_ring_components(name: str = 'ring',
                        num_segments: int = 4,
                        radius: float = 3.0,
                        thickness: float = None,
                        offset: Sequence[float] = None):
  """Draw a ring with capsules."""
  offset = offset or [0, 0, 0]
  offset = jp.array(offset)
  thickness = thickness or radius / 40.
  components = {}
  angles = np.linspace(0, np.pi * 2, num_segments + 1)
  for i, angle in enumerate(angles[:-1]):
    k = f'{name}{i}'
    ring_length = radius * np.tan(np.pi / num_segments)
    components[k] = dict(
        component='singleton',
        component_params=dict(
            size=[thickness, ring_length * 2],
            collider_type='capsule',
            no_obs=True),
        pos=offset + jp.array(
            (radius * np.cos(angle), radius * np.sin(angle), -ring_length)),
        quat=math.euler_to_quat(jp.array([90, angle / jp.pi * 180, 0])),
        quat_origin=(0, 0, ring_length),
        frozen=True,
        collide=False)
  return components


def add_sumo(
    env_desc: Dict[str, Any],
    centering_scale: float = 1.,
    control_scale: float = 0.1,
    draw_scale: float = 0.,
    knocking_scale: float = 1.,
    opp_scale: float = 1.,
    ring_size: float = 3.,
    win_bonus: float = 1.,
):
  """Add a sumo task."""
  agents = sorted(env_desc['components'])
  agent_groups = {agent: {'reward_names': ()} for agent in agents}
  components = {agent: {'reward_fns': {}} for agent in agents}
  edges = {}
  yokozuna, komusubis = agents[0], agents[1:]
  for agent in komusubis:
    edge_name = component_editor.concat_comps(agent, yokozuna)
    edges[edge_name] = dict(
        reward_fns=dict(
            # komusubis wants to push out yokozuna
            komu_win_bonus=dict(
                reward_type=reward_functions.exp_norm_reward,
                obs=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                max_dist=ring_size,
                done_bonus=win_bonus,
                scale=-knocking_scale,
            ),
            komu_lose_penalty=dict(
                reward_type=reward_functions.exp_norm_reward,
                obs=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                max_dist=ring_size,
                done_bonus=-win_bonus,
                scale=centering_scale,
            ),
            # yokozuna wants to push out komusubis
            yoko_win_bonus=dict(
                reward_type=reward_functions.exp_norm_reward,
                obs=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                max_dist=ring_size,
                done_bonus=win_bonus,
                scale=-knocking_scale,
            ),
            # each agent aims to be close to the center
            yoko_lose_penalty=dict(
                reward_type=reward_functions.exp_norm_reward,
                obs=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                max_dist=ring_size,
                done_bonus=-win_bonus,
                scale=centering_scale,
            ),
            # move to opponent's direction
            komu_move_to_yoko=dict(
                reward_type=reward_functions.direction_reward,
                vel0=lambda x, y: so('body', 'vel', x['root'], indices=(0, 1)),
                vel1=lambda x, y: so('body', 'vel', y['root'], indices=(0, 1)),
                pos0=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                pos1=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                scale=opp_scale,
            ),
            yoko_move_to_komu=dict(
                reward_type=reward_functions.direction_reward,
                vel0=lambda x, y: so('body', 'vel', y['root'], indices=(0, 1)),
                vel1=lambda x, y: so('body', 'vel', x['root'], indices=(0, 1)),
                pos0=lambda x, y: so('body', 'pos', y['root'], indices=(0, 1)),
                pos1=lambda x, y: so('body', 'pos', x['root'], indices=(0, 1)),
                scale=opp_scale,
            ),
        ))
    agent_groups[agent]['reward_names'] += (('komu_win_bonus', agent, yokozuna),
                                            ('komu_lose_penalty', agent,
                                             yokozuna), ('komu_move_to_yoko',
                                                         agent, yokozuna))
    agent_groups[yokozuna]['reward_names'] += (('yoko_win_bonus', agent,
                                                yokozuna), ('yoko_lose_penalty',
                                                            agent, yokozuna),
                                               ('yoko_move_to_komu', yokozuna,
                                                agent))
  for agent in agents:
    components[agent]['reward_fns'].update(
        dict(
            control_penalty=dict(
                reward_type=reward_functions.control_reward,
                scale=control_scale,
            ),
            draw_penalty=dict(
                reward_type=reward_functions.constant_reward,
                value=-draw_scale,
            ),
        ))
    agent_groups[agent]['reward_names'] += (('control_penalty', agent),
                                            ('draw_penalty', agent))
  # add sumo ring
  components.update(get_ring_components(radius=ring_size, num_segments=20))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc


def add_squidgame(env_desc: Dict[str, Any],
                  ring_size: float = 3.0,
                  run_scale: float = 0):
  """Add a simplified squid game task."""
  # TODO: finish reward functions
  agents = sorted(env_desc['components'])
  agent_groups = {agent: {'reward_names': ()} for agent in agents}
  components = {agent: {'reward_fns': {}} for agent in agents}
  edges = {}
  defender, attackers = agents[0], agents[1:]
  for agent in attackers:
    edge_name = component_editor.concat_comps(agent, defender)
    edges[edge_name] = dict(
        reward_fns=dict(
            # defenders aim to chase the attackers
            chase=dict(reward_type='root_dist', offset=2 * ring_size + 0.5),))
    agent_groups[defender]['reward_names'] += (('chase', agent, defender),)
  for agent in agents:
    if run_scale > 0:
      # add velocity bonus for each agent
      components[agent] = dict(
          reward_fns=dict(run=get_run_reward(scale=run_scale)))
      agent_groups[agent]['reward_names'] += (('run', agent),)
  # add rings
  components.update(
      get_ring_components(
          name='square',
          offset=(ring_size, 0, 0),
          radius=ring_size,
          thickness=ring_size / 40.,
          num_segments=4))
  components.update(
      get_ring_components(
          name='defender_circle',
          offset=(ring_size * 2, 0, 0),
          radius=ring_size / 5,
          thickness=ring_size / 40.,
          num_segments=10))
  components.update(
      get_ring_components(
          name='triangle',
          offset=(-ring_size / np.sqrt(3), 0, 0),
          radius=ring_size / np.sqrt(3),
          thickness=ring_size / 40.,
          num_segments=3))
  components.update(
      get_ring_components(
          name='attacker_circle',
          offset=(-ring_size * np.sqrt(3), 0, 0),
          radius=ring_size / 5,
          thickness=ring_size / 40.,
          num_segments=10))
  merge_desc(
      env_desc,
      dict(agent_groups=agent_groups, edges=edges, components=components))
  return env_desc


TASK_MAP = dict(
    follow=add_follow, chase=add_chase, sumo=add_sumo, squidgame=add_squidgame)


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

  return TASK_MAP[task](env_desc=env_desc, **kwargs)


ENV_DESCS = {k: functools.partial(create_desc, task=k) for k in TASK_MAP}
