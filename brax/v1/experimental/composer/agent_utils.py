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

"""Multi-agent RL utilities.

ComposerEnv() is inherently object-oriented (observations, actions,
rewards are all dictionaries), and therefore it is
straight-forward to enable multi-agent environments.

composer_env.metadata.agent_groups specifies full information for the
multi-agent task. E.g. envs/ma_descs.py.py.

E.g.
   agent_groups=dict(
                agent1=dict(reward_names=(('dist', 'agent1', 'agent2'),)),
                agent2=dict(reward_agents=('agent2',)),
            )
   means:
     - agent1 uses 'dist' reward function between 'agent1' and 'agent2'
     - agent2 uses all rewards of 'agent2'
     - by defaults, each agent uses its own action_space
       e.g. equivalent to agent1=(..., action_agents=('agent1',), ...)

agent_groups currently defines which rewards/actions belong to which agent.
observation is the same among all agents (TODO: add optionality).
"""

from collections import OrderedDict as odict
from typing import Dict, Tuple, Any, List
from brax.v1.experimental.composer import component_editor as ce
from jax import numpy as jnp


def set_agent_groups(metadata: Any, action_shapes: Dict[str, Any],
                     observer_shapes: Dict[str, Any]):
  """Set metadata.agent_groups and return additional infos.

  Args:
    metadata: a Metadata object of ComposerEnv()
    action_shapes: an OrderedDict of sys action shape info (data_utils.py)
    observer_shapes: an OrderedDict of observation shape info (data_utils.py)

  Returns:
    group_action_shapes: an OrderedDict of agent-based action shape info
  """
  del observer_shapes
  if not metadata.agent_groups:
    return {}
  group_action_shapes = odict()
  for _, (k, v) in enumerate(sorted(metadata.agent_groups.items())):
    set_names_info(v, sorted(metadata.reward_fns.keys()), 'reward')
    set_names_info(v, action_shapes.keys(), 'action', default_agents=(k,))
    group_action_shapes[k] = get_action_shape(v, action_shapes)
  return group_action_shapes


def process_agent_rewards(metadata: Any,
                          reward_tuple_dict: Dict[str, Tuple[jnp.ndarray]]):
  """Process reward etc.

  based on metadata.agent_groups.

  Args:
    metadata: Metadata object in ComposerEnv()
    reward_tuple_dict: a dict of scalars (reward, score, done)

  Returns:
    reward: a jnp.ndarray of size [num_agents]
    score: a jnp.ndarray of size [num_agents]
    done: a jnp.ndarray of size [num_agents]
  """
  num_agents = len(metadata.agent_groups)
  reward, score, done = jnp.zeros((3,) + (num_agents,))
  all_reward_names = ()
  for i, (_, v) in enumerate(sorted(metadata.agent_groups.items())):
    reward_names = v.get('reward_names', ())
    for reward_name in reward_names:
      assert reward_name in reward_tuple_dict, (
          f'{reward_name} not in {reward_tuple_dict.keys()}')
      r, s, d = reward_tuple_dict[reward_name]
      reward = reward.at[i].add(r)
      score = score.at[i].add(s)
      done = done.at[i].set(jnp.logical_or(done[i], d))
    all_reward_names += reward_names
  assert set(all_reward_names) == set(reward_tuple_dict.keys()), (
      f'{set(all_reward_names)} != {set(reward_tuple_dict.keys())}')
  done = jnp.any(done, axis=-1)  # ensure done is a scalar
  return reward, score, done


def get_action_shape(v: Dict[str, Tuple[Any]], action_shapes: Dict[str, Any]):
  """Set action_indices."""
  names = v.get('action_names')
  indices = ()
  for name in names:
    s = action_shapes[name]
    indices_ = s.get('indices', list(range(s['start'], s['end'])))
    indices += tuple(indices_)
  return dict(size=len(indices), shape=(len(indices),), indices=indices)


def set_names_info(
    v: Dict[str, Tuple[Any]],
    all_names: List[str],
    var: str = 'reward',
    default_agents: Tuple[str] = (),
):
  """Set names based on '{var}_(names|agents)'."""
  names = v.get(f'{var}_names', ())
  assert all(isinstance(v, tuple)
             for v in names), f'{names} must be a Sequence of Tuples'
  names = tuple(ce.concat_name(*v) for v in names)
  agents = v.get(f'{var}_agents', default_agents)
  for agent in agents:
    agent_args = (agent,) if isinstance(agent, str) else agent
    assert isinstance(agent_args, (tuple, list)), agent_args
    names += tuple(k for k in all_names if ce.match_name(k, *agent_args))
  names = tuple(dict.fromkeys(names))  # remove duplicates/keep order
  v[f'{var}_names'] = names
