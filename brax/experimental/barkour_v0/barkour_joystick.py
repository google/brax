# Copyright 2023 DeepMind Technologies Limited.
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

"""Barkour v0 brax environment for flat-terrain joystick policy training."""

from brax import actuator
from brax import base
from brax import envs
from brax import math
import brax.envs.base as env
from brax.io import mjcf
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict


def get_config():
  """Returns reward config."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            # The coefficients for all reward terms used for training. All
            # physical quantities are in SI units, if no otherwise specified,
            # i.e. joint positions are in rad, positions are measured in meters,
            # torques in Nm, and time in seconds, and forces in Newtons.
            scales=config_dict.ConfigDict(
                dict(
                    # Tracking rewards are computed using exp(-delta^2/sigma)
                    # sigma can be a hyperparameters to tune.
                    # Track the base x-y velocity (no z-velocity tracking.)
                    tracking_lin_vel=1.5,
                    # Track the angular velocity along z-axis, i.e. yaw rate.
                    tracking_ang_vel=0.8,
                    # Below are regularization terms, we roughly divide the
                    # terms to base state regularizations, joint
                    # regularizations, and other behavior regularizations.
                    # Penalize the base velocity in z direction, L2 penalty.
                    lin_vel_z=-2.0,
                    # Penalize the base roll and pitch rate. L2 penalty.
                    ang_vel_xy=-0.05,
                    # Penalize non-zero roll and pitch angles. L2 penalty.
                    orientation=-5.0,
                    # L2 regularization of joint torques, |tau|^2.
                    torques=-0.0002,
                    # Penalize the change in the action and encourage smooth
                    # actions. L2 regularization |action - last_action|^2
                    action_rate=-0.2,
                    # Encourage long swing steps.  However, it does not
                    # encourage high clearances.
                    feet_air_time=0.1,
                    # Encourage no motion at zero command, L2 regularization
                    # |q - q_default|^2.
                    stand_still=-0.5,
                    # Early termination penalty.
                    termination=-1.0,
                    # Penalizing foot slipping on the ground.
                    foot_slip=-0.1,
                )
            ),
            # Tracking reward = exp(-error^2/sigma).
            tracking_sigma=0.25,
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(rewards=get_default_rewards_config(),))

  return default_config


class Barkourv0(env.PipelineEnv):
  """Environment for training the barkour quadruped joystick policy.

  This environment demonstrates training the joystick policy as seen in
  https://arxiv.org/abs/2305.14654. A command [x, y, w] is chosen upon reset,
  where x/y are the target linear velocities and w is that target angular
  velocity in the +z direction. The `tracking_lin_vel` and
  `tracking_ang_vel` rewards make sure the robot follows these joystick
  commands. The other rewards regularize the policy such that walking gaits
  are smooth and show better transfer to real.
  """

  def __init__(
      self,
      obs_noise: float = 0.05,
      kick_vel=0.05,
      action_scale=0.3,
      backend: str = 'generalized',
      debug=False,
      **kwargs,
  ):
    if backend != 'generalized':
      raise NotImplementedError(
          'Only generalized backend is supported for barkour.'
      )

    n_frames = 10
    path = epath.Path(epath.resource_path('brax')) / (
        'experimental/barkour_v0/assets/barkour_v0_brax.xml'
    )
    sys = mjcf.load(path)

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
    kwargs['debug'] = kwargs.get('debug', debug)
    super().__init__(sys=sys, backend=backend, **kwargs)

    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._default_ap_pose = sys.init_q[7:19]
    self.reward_config = get_config()
    self.torso_idx = self.sys.link_names.index('chassis')
    self.lowers = self._default_ap_pose - jp.array([0.2, 0.8, 0.8] * 4)
    self.uppers = self._default_ap_pose + jp.array([0.2, 0.8, 0.8] * 4)

  def sample_command(self, rng: jax.Array) -> jax.Array:
    lin_vel_x = [-0.6, 1.5]  # min max [m/s]
    lin_vel_y = [-0.8, 0.8]  # min max [m/s]
    ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(
        key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
    )
    new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

    return new_cmd

  def reset(self, rng: jax.Array) -> env.State:
    rng, key = jax.random.split(rng)

    q = self.sys.init_q
    qd = jp.zeros(self.sys.qd_size())
    new_cmd = self.sample_command(key)
    pipeline_state = self.pipeline_init(q, qd)

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'last_contact_buffer': jp.zeros((20, 4), dtype=bool),
        'command': new_cmd,
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'obs_history': jp.zeros(15 * 31),
        'rew_tuple': {
            'tracking_lin_vel': 0.0,
            'tracking_ang_vel': 0.0,
            'lin_vel_z': 0.0,
            'ang_vel_xy': 0.0,
            'orientation': 0.0,
            'torque': 0.0,
            'action_rate': 0.0,
            'stand_still': 0.0,
            'feet_air_time': 0.0,
            'foot_slip': 0.0,
        },
        'step': 0,
        'kick': jp.array([0.0, 0.0]),
    }

    obs = self._get_obs(pipeline_state, state_info)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rew_tuple']:
      metrics[k] = state_info['rew_tuple'][k]
    state = env.State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
    return state

  def step(self, state: env.State, action: jax.Array) -> env.State:
    rng, rng_noise, cmd_rng, kick_noise_2 = jax.random.split(
        state.info['rng'], 4
    )

    # kick
    push_interval = 100
    kick_theta = jax.random.uniform(kick_noise_2, minval=0.0, maxval=2 * jp.pi)
    state.info['kick'] = jp.where(
        state.info['step'] > push_interval,
        jp.array([jp.cos(kick_theta), jp.sin(kick_theta)]),
        jp.array([0.0, 0.0]),
    )
    state = state.tree_replace(
        {
            'pipeline_state.qd': state.pipeline_state.qd.at[:2].set(
                state.info['kick'] * self._kick_vel
                + state.pipeline_state.qd[:2]  # pytype: disable=attribute-error
            )
        }
    )

    # physics step
    cur_action = jp.array(action)
    action = action[:12] * self._action_scale
    motor_targets = jp.clip(
        action + self._default_ap_pose, self.lowers, self.uppers
    )
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)

    # observation data
    obs = self._get_obs(pipeline_state, state.info)
    obs_noise = self._obs_noise * jax.random.uniform(
        rng_noise, obs.shape, minval=-1, maxval=1)
    q, qd = pipeline_state.q, pipeline_state.qd
    joint_angles = q[7:]
    joint_vel = qd[6:]
    default_angles = self._default_ap_pose
    torque = actuator.to_tau(self.sys, action, q, qd)

    # foot contact data
    foot_contact = 0.017 - self._get_feet_pos_vel(pipeline_state)[0][:, 2]
    contact = foot_contact > -1e-3  # a mm or less off the floor
    contact_filt_mm = jp.logical_or(contact, state.info['last_contact'])
    contact_filt_cm = jp.logical_or(
        foot_contact > -1e-2, state.info['last_contact']
    )
    first_contact = (state.info['feet_air_time'] > 0) * (contact_filt_mm)
    state.info['feet_air_time'] += self.dt

    # reward
    qp = pipeline_state
    rew_tuple = {
        'tracking_lin_vel': (
            self._reward_tracking_lin_vel(state.info['command'], qp)
            * self.reward_config.rewards.scales.tracking_lin_vel
        ),
        'tracking_ang_vel': (
            self._reward_tracking_ang_vel(state.info['command'], qp)
            * self.reward_config.rewards.scales.tracking_ang_vel
        ),
        'lin_vel_z': (
            self._reward_lin_vel_z(qp)
            * self.reward_config.rewards.scales.lin_vel_z
        ),
        'ang_vel_xy': (
            self._reward_ang_vel_xy(qp)
            * self.reward_config.rewards.scales.ang_vel_xy
        ),
        'orientation': (
            self._reward_orientation(qp)
            * self.reward_config.rewards.scales.orientation
        ),
        'torque': (
            self._reward_torques(torque)
            * self.reward_config.rewards.scales.torques
        ),
        'action_rate': (
            self._reward_action_rate(cur_action, state.info['last_act'])
            * self.reward_config.rewards.scales.action_rate
        ),
        'stand_still': (
            self._reward_stand_still(
                state.info['command'], joint_angles, default_angles
            )
            * self.reward_config.rewards.scales.stand_still
        ),
        'feet_air_time': (
            self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            )
            * self.reward_config.rewards.scales.feet_air_time
        ),
        'foot_slip': (
            self._reward_foot_slip(qp, contact_filt_cm)
            * self.reward_config.rewards.scales.foot_slip
        ),
    }
    rew = (
        rew_tuple['tracking_lin_vel']
        + rew_tuple['tracking_ang_vel']
        + rew_tuple['orientation']
        + rew_tuple['ang_vel_xy']
        + rew_tuple['lin_vel_z']
        + rew_tuple['torque']
        + rew_tuple['action_rate']
        + rew_tuple['stand_still']
        + rew_tuple['feet_air_time']
        + rew_tuple['foot_slip']
    )

    reward = jp.clip(rew * self.dt, 0.0, 10000.0)

    # state management
    state.info['last_act'] = cur_action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt_mm
    state.info['last_contact'] = contact
    state.info['last_contact_buffer'] = jp.roll(
        state.info['last_contact_buffer'], 1, axis=0
    )
    state.info['last_contact_buffer'] = (
        state.info['last_contact_buffer'].at[0].set(contact)
    )
    state.info['rew_tuple'] = rew_tuple
    state.info['step'] += 1
    state.info.update(rng=rng)

    # resetting logic if joint limits are reached or robot is falling
    done = 0.0
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.where(jp.dot(math.rotate(up, qp.x.rot[0]), up) < 0, 1.0, done)
    done = jp.where(jp.logical_or(
        jp.any(joint_angles < .98 * self.lowers),
        jp.any(joint_angles > .98 * self.uppers)), 1.0, done)
    done = jp.where(qp.x.pos[self.torso_idx, 2] < 0.18, 1.0, done)

    # termination reward
    reward += jp.where(
        (done == 1.0) & (state.info['step'] < 500),
        self.reward_config.rewards.scales.termination,
        0.0,
    )

    # when done, reset step counter and sample new command
    state.info['command'] = jp.where(
        (done == 1.0) & (state.info['step'] > 500),
        self.sample_command(cmd_rng), state.info['command'])
    state.info['step'] = jp.where(
        (done == 1.0) | (state.info['step'] > 500), 0, state.info['step']
    )

    # log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(qp.x.pos[self.torso_idx])[1]
    for k in state.info['rew_tuple'].keys():  # pytype: disable=attribute-error
      state.metrics[k] = state.info['rew_tuple'][k]

    state = state.replace(
        pipeline_state=pipeline_state, obs=obs + obs_noise, reward=reward,
        done=done)
    return state

  def _get_obs(self, pipeline_state, state_info) -> jax.Array:
    # Get observations:
    # yaw_rate,  projected_gravity, command,  motor_angles, last_action
    x, xd = pipeline_state.x, pipeline_state.xd

    inv_base_orientation = math.quat_inv(x.rot[0])
    local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)
    cmd = state_info['command']

    obs_list = []
    # yaw rate
    obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
    # projected gravity
    obs_list.append(
        math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
    # command
    obs_list.append(cmd * jp.array([2.0, 2.0, 0.25]))
    # motor angles
    angles = pipeline_state.q[7:19]
    obs_list.append(angles - self._default_ap_pose)
    # last action
    obs_list.append(state_info['last_act'])

    obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

    # stack observations through time
    single_obs_size = len(obs)
    state_info['obs_history'] = jp.roll(
        state_info['obs_history'], single_obs_size
    )
    state_info['obs_history'] = jp.array(
        state_info['obs_history']).at[:single_obs_size].set(obs)
    return state_info['obs_history']

  # ------------ reward functions----------------
  def _reward_lin_vel_z(self, qp):
    # Penalize z axis base linear velocity
    return jp.square(qp.xd.vel[0, 2])

  def _reward_ang_vel_xy(self, qp):
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(qp.xd.ang[0, :2]))

  def _reward_orientation(self, qp):
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, qp.x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_orientation2(self, qp):
    # Reward flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, qp.x.rot[0])
    return jp.sum(rot_up[2])

  def _reward_torques(self, torques):
    # Penalize torques
    # return jp.sum(jp.square(torques))
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(self, act, last_act):
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(self, commands, qp):
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(qp.xd.vel[0], math.quat_inv(qp.x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(
        -lin_vel_error / self.reward_config.rewards.tracking_sigma
    )
    return lin_vel_reward

  def _reward_tracking_ang_vel(self, commands, qp):
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(qp.xd.ang[0], math.quat_inv(qp.x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error/self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(self, air_time, first_contact, commands):
    # Reward air time.
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= (
        math.normalize(commands[:2])[1] > 0.05
    )  # no reward for zero command
    return rew_air_time

  def _reward_stand_still(
      self,
      commands,
      joint_angles,
      default_angles,
  ):
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - default_angles)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _get_feet_pos_vel(self, qp):
    foot_indices = jp.array([3, 6, 9, 12])
    offset = jp.array([
        [-0.191284, -0.0191638, 0.013],
        [-0.191284, -0.0191638, -0.013],
        [-0.191284, -0.0191638, 0.013],
        [-0.191284, -0.0191638, -0.013],
    ])
    offset = base.Transform.create(pos=offset)
    pos = qp.x.take(foot_indices).vmap().do(offset).pos
    vel = offset.vmap().do(qp.xd.take(foot_indices)).vel
    return pos, vel

  def _reward_foot_slip(self, qp, contact_filt):
    # Get feet velocities
    _, foot_world_vel = self._get_feet_pos_vel(qp)
    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(
        jp.square(foot_world_vel[:, :2]) * contact_filt.reshape((-1, 1))
    )


envs.register_environment('barkour_v0_joystick', Barkourv0)


def domain_randomize(
    sys,
    rng,
    gain_min=-5,
    gain_max=5,
):
  """Randomizes the system in training for sim2real transfer."""

  @jax.vmap
  def rand(rng):
    """Generates random values."""
    # friction
    shape = [g.friction.shape[0] for g in sys.geoms]
    friction = []
    for s in shape:
      rng, key = jax.random.split(rng)
      friction.append(jax.random.uniform(key, (s,), minval=0.8, maxval=1.4))
    # actuator
    rng, key1 = jax.random.split(rng, 2)
    gain_range = (gain_min, gain_max)
    gain_noise = jax.random.uniform(
        key1, (1,), minval=gain_range[0], maxval=gain_range[1]
    )
    gain = sys.actuator.gain + gain_noise
    # center of mass
    _, key = jax.random.split(rng)
    com = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
    ltps = sys.link.inertia.transform.pos.at[0, :3].set(
        sys.link.inertia.transform.pos[0, :3] + com
    )
    return friction, gain, ltps

  friction, gain, link_inertia_transform_pos = rand(rng)

  in_axes = jax.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'actuator.gain': 0,
      'link.inertia.transform.pos': 0,
      'geoms.friction': 0,
  })

  sys = sys.tree_replace({
      'actuator.gain': gain,
      'link.inertia.transform.pos': link_inertia_transform_pos,
      'geoms.friction': friction,
  })

  return sys, in_axes
