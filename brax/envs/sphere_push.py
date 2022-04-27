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

"""Trains an actuated sphere to roll in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import env


class SpherePush(env.Env):
  """Trains an actuated sphere to push a ball in the +x direction."""

  def __init__(self, legacy_spring=False, **kwargs):
    config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
    super().__init__(config=config, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    ball_init_x, ball_init_y = 2., 0. # planar starting position of pushable ball
    qp = brax.QP(
    # position of **each body** in 3d (z is up, right-hand coordinates) -- 2 bodies, ground and ball 
    pos = jp.array([[0., 0., .5],                   # p1
                    [0., 0., .5],                   # roll
                    [0., 0., .5],                   # pitch
                    [ball_init_x, ball_init_y, .5], # ball
                    [0., 0., 0.]]),                 # ground 
    # velocity of each body in 3d (both at rest)
    vel = jp.array([[0., 0., 0.],       
                    [0., 0., 0.],       
                    [0., 0., 0.],       
                    [0., 0., 0.],
                    [0., 0., 0.]]),     
    # rotation about center of body, as a quaternion (w, x, y, z)
    rot = jp.array([[1., 0., 0., 0.], 
                    [1., 0., 0., 0.], 
                    [1., 0., 0., 0.], 
                    [1., 0., 0., 0.],
                    [1., 0., 0., 0.]]), 
    # angular velocity about center of body in 3d
    ang = jp.array([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]])
)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'ball_forward_reward': zero,
        # 'ball_dist_reward': zero,
        'towards_ball_reward': zero,
        # 'near_ball_cost': zero,
        'reward_ctrl_cost': zero,
        'reward_contact_cost': zero,
        'reward_survive': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)
    
    
    # push ball forward - big reward
    x_ball_before = state.qp.pos[3, 0]
    x_ball_after = qp.pos[3, 0]
    ball_forward_reward = (x_ball_after - x_ball_before) / self.sys.config.dt
    ball_forward_reward *= 30
    
    # # ball distance travelled from starting position
    # ball_dist_reward = qp.pos[3,0] - 2.0
    
    # move p1 towards ball - small reward
    x_dist_before = abs(state.qp.pos[0, 0] - state.qp.pos[3, 0])
    x_dist_after = abs(qp.pos[0, 0] - qp.pos[3, 0])
    y_dist_before = abs(state.qp.pos[0, 1] - state.qp.pos[3, 1])
    y_dist_after = abs(qp.pos[0, 1] - qp.pos[3, 1])
    dist_before = abs((x_dist_before**2 + y_dist_before**2)**0.5)
    dist_after = abs((x_dist_after**2 + y_dist_after**2)**0.5)
    towards_ball_reward = (dist_before - dist_after) / self.sys.config.dt
    
    # # have p1 be near to ball - small reward
    # x_dist = abs(qp.pos[0, 0] - qp.pos[3, 0])
    # y_dist = abs(qp.pos[0, 1] - qp.pos[3, 1])
    # dist = abs((x_dist**2 + y_dist**2)**0.5)
    # near_ball_cost = dist
    
    ctrl_cost = .5 * jp.sum(jp.square(action)) # dependent on torque
    
    # not sure what this is - set to zero for now
    contact_cost = jp.float32(0) # (0.5 * 1e-3 * jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    survive_reward = jp.float32(1)
    
    reward = ball_forward_reward 
    # reward += ball_dist_reward 
    reward += towards_ball_reward
    # reward -= near_ball_cost
    reward -= ctrl_cost
    reward += survive_reward - contact_cost

    # termination - these shouldn't matter for our ball
    done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
    done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
    state.metrics.update(
        ball_forward_reward=ball_forward_reward,
        # ball_dist_reward=ball_dist_reward,
        towards_ball_reward=towards_ball_reward,
        # near_ball_cost=near_ball_cost,
        reward_ctrl_cost=ctrl_cost,
        reward_contact_cost=contact_cost,
        reward_survive=survive_reward)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observes body position and velocities."""
    # # some pre-processing to pull joint angles and velocities
    # (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # # qpos:
    # # Z of the torso (1,)
    # # orientation of the torso as quaternion (4,)
    # # joint angles (8,)
    # qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

    # # qvel:
    # # velocity of the torso (3,)
    # # angular velocity of the torso (3,)
    # # joint angle velocities (8,)
    # qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # # external contact forces:
    # # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
    # # ignores
    # cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
    # # flatten bottom dimension
    # cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
    
    # obs = jp.concatenate(qpos + qvel + cfrc)
    
    # Trying something - observe everything?
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
    pos, rot, vel, ang = [qp.pos.flatten(), joint_angle], [qp.rot.flatten()], [qp.vel.flatten()], [qp.ang.flatten(), joint_vel]
    obs = jp.concatenate(pos + rot + vel + ang)

    return obs


_SYSTEM_CONFIG = """

bodies {
  name: "p1"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}
bodies {
  name: "p1_roll"
  mass: 0.01
}
bodies {
  name: "p1_pitch"
  mass: 0.01
}

bodies {
  name: "ball"
  colliders {
    capsule {
      radius: 0.5
      length: 1.0
    }
  }
  mass: 1.0
}

bodies {
  name: "ground"
  colliders {
    plane {
    }
  }
  frozen {
    all: true
  }
}
joints {
  name: "joint1"
  parent: "p1_roll"
  child: "p1"
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
joints {
  name: "joint2"
  parent: "p1_pitch"
  child: "p1"
  rotation {
    z: -90.0
  }
  angle_limit {
    min: -180.0
    max: 180.0
  }
}
actuators {
  name: "torque1"
  joint: "joint1"
  strength: 100.0
  torque {
  }
}
actuators {
  name: "torque2"
  joint: "joint2"
  strength: 100.0
  torque {
  }
}
elasticity: 1.0
friction: 6.0
gravity {
  z: -9.8
}
angular_damping: -0.05
dt: 0.05
substeps: 20
dynamics_mode: "pbd"

"""