"""
SafePointGoalWeighted Environment

A point navigation environment derived from `SafePointGoal` that uses
distance-weighted hazard cost (Safety-Gymnasium style):

  cost_hazards = sum_i( hazard_cost_weight * max(0, hazard_size - dist_i) )

All other mechanics (observations, reward shaping, placement) are identical to
`SafePointGoal`.
"""

from typing import Dict
import os
from ml_collections import config_dict
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State

# Construct absolute path to point_hazard_goal.xml
_current_dir = os.path.dirname(os.path.abspath(__file__))


def get_xml_path_for_hazards(num_hazards: int) -> str:
    """Get the appropriate XML file path for the given number of hazards."""
    available_configs = {
        3: 'point_hazard_goal_mocap.xml',      # Original 3 hazards
        4: 'point_hazard_goal_mocap_4.xml',     # 4 hazards
        8: 'point_hazard_goal_mocap_8.xml',     # 8 hazards
    }

    if num_hazards in available_configs:
        xml_filename = available_configs[num_hazards]
    else:
        # Find closest available count
        available_counts = sorted(available_configs.keys())
        closest_count = min(available_counts, key=lambda x: abs(x - num_hazards))
        xml_filename = available_configs[closest_count]
        print(f"Warning: No exact XML config for {num_hazards} hazards, using {closest_count} hazards instead")

    return os.path.join(_current_dir, 'assets', xml_filename)


def default_config(num_hazards: int = 8) -> config_dict.ConfigDict:
    """Returns the default config for SafePointGoalWeighted environment."""
    config = config_dict.create(
        # Environment settings
        num_hazards=num_hazards,
        hazard_size=0.3,
        hazard_cost_weight=1.0,  # NEW: distance-weighted hazard cost scale

        # Goal settings
        goal_size=0.7,
        reward_goal=10.0,
        reward_distance=3.0,

        # Control settings
        ctrl_cost_weight=0.001,

        # Lidar settings (simplified)
        lidar_num_bins=16,
        lidar_max_dist=3.0,

        # Physics settings
        terminate_when_unhealthy=True,
        healthy_z_range=(0.05, 0.3),
        reset_noise_scale=0.005,
        max_velocity=5.0,

        # Placement constraints (Safety Gymnasium style)
        placement_extents=(-2.0, -2.0, 2.0, 2.0),
        agent_keepout=0.4,
        goal_keepout=0.4,
        hazard_keepout=0.4,
        placement_margin=0.1,
        max_placement_attempts=100,
        max_layout_attempts=1000,

        # Debug settings
        debug=False,
    )
    return config


def safe_norm(x, axis=None, keepdims=False, eps=1e-8):
    """Safely compute the norm with a small epsilon to avoid NaN."""
    return jp.sqrt(jp.sum(jp.square(x), axis=axis, keepdims=keepdims) + eps)


class SafePointGoalWeighted(PipelineEnv):
    """
    Safe Point Goal Navigation Environment with distance-weighted hazard cost.

    Identical to `SafePointGoal` except the safety cost.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        num_hazards: int = 8,
        **kwargs,
    ):
        if config is None:
            config = default_config(num_hazards)

        overrides = kwargs.pop('config_overrides', None)
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                config[key] = value

        self._debug = config.debug

        # Load the appropriate MuJoCo model based on hazard count
        xml_path = get_xml_path_for_hazards(num_hazards)
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 4

        # Body ids
        self._agent_body = 1
        self._goal_mocap_id = 0

        xml_filename = xml_path.split('/')[-1]
        if '8.xml' in xml_filename:
            available_hazards_in_xml = 8
        elif '4.xml' in xml_filename:
            available_hazards_in_xml = 4
        elif 'mocap.xml' in xml_filename:
            available_hazards_in_xml = 3
        else:
            available_hazards_in_xml = 8

        self._num_hazards = min(num_hazards, available_hazards_in_xml)
        self._hazard_mocap_ids = list(range(1, self._num_hazards + 1))

        if self._num_hazards < num_hazards:
            print(f"Warning: Requested {num_hazards} hazards but XML only has {available_hazards_in_xml}. Using {self._num_hazards} hazards.")

        # Sensor lookup
        self._sensor_info = {}
        required_sensors = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
        sensor_found_flags = {name: False for name in required_sensors}
        if mj_model.nsensor > 0:
            for i in range(mj_model.nsensor):
                name = mj_model.sensor(i).name
                if name in required_sensors:
                    start_adr = mj_model.sensor_adr[i]
                    dim = mj_model.sensor_dim[i]
                    self._sensor_info[name] = (start_adr, dim)
                    sensor_found_flags[name] = True
        else:
            print("Warning: Model has no sensors defined (mj_model.nsensor = 0).")

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 4
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        # Store configuration
        self._config = config
        self._hazard_size = config.hazard_size
        self._goal_size = config.goal_size
        self._reward_goal = config.reward_goal
        self._reward_distance = config.reward_distance
        self._ctrl_cost_weight = config.ctrl_cost_weight
        self._terminate_when_unhealthy = config.terminate_when_unhealthy
        self._healthy_z_range = config.healthy_z_range
        self._reset_noise_scale = config.reset_noise_scale
        self._max_velocity = config.max_velocity
        self._hazard_cost_weight = config.hazard_cost_weight

        # Lidar configuration
        self._lidar_num_bins = config.lidar_num_bins
        self._lidar_max_dist = config.lidar_max_dist

        # Placement constraints
        self._placement_extents = config.placement_extents
        self._agent_keepout = config.agent_keepout
        self._goal_keepout = config.goal_keepout
        self._hazard_keepout = config.hazard_keepout
        self._placement_margin = config.placement_margin
        self._max_placement_attempts = config.max_placement_attempts
        self._max_layout_attempts = config.max_layout_attempts

        if self._debug:
            print(f"SafePointGoalWeighted initialized with {num_hazards} hazards")

    def reset(self, rng: jp.ndarray) -> State:
        """Reset with constrained placement using JAX control flow."""
        rng, rng1, rng2, rng_layout = jax.random.split(rng, 4)

        # Randomize initial position with small noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        # Ensure valid quaternion
        qpos = jax.lax.cond(
            qpos.shape[0] > 6,
            lambda qp: qp.at[3:7].set(qp[3:7] / (safe_norm(qp[3:7]) + 1e-8)),
            lambda qp: qp,
            qpos,
        )

        data = self.pipeline_init(qpos, qvel)
        agent_pos = data.xpos[self._agent_body]

        # Helpers for placement
        def _sample_candidate_positions(rkey: jp.ndarray, num_candidates: int, keepout: float) -> jp.ndarray:
            min_x, min_y, max_x, max_y = self._placement_extents
            min_x = min_x + keepout
            min_y = min_y + keepout
            max_x = max_x - keepout
            max_y = max_y - keepout
            rkey, sk1, sk2 = jax.random.split(rkey, 3)
            xs = jax.random.uniform(sk1, (num_candidates,), minval=min_x, maxval=max_x)
            ys = jax.random.uniform(sk2, (num_candidates,), minval=min_y, maxval=max_y)
            zs = jp.full((num_candidates,), 0.09)
            return jp.stack([xs, ys, zs], axis=1)

        def _choose_valid_position(rkey: jp.ndarray,
                                   existing_xy_all: jp.ndarray,
                                   existing_keepouts_all: jp.ndarray,
                                   active_count: jp.ndarray,
                                   keepout: float,
                                   num_candidates: int):
            candidates = _sample_candidate_positions(rkey, num_candidates, keepout)
            cand_xy = candidates[:, :2]

            diffs = cand_xy[:, None, :] - existing_xy_all[None, :, :]
            dists = jp.sqrt(jp.sum(jp.square(diffs), axis=-1) + 1e-8)
            req = existing_keepouts_all[None, :] + keepout + self._placement_margin

            M = existing_xy_all.shape[0]
            idxs = jp.arange(M)[None, :]
            active_mask = idxs < active_count
            per_pos_valid = jp.logical_or(jp.logical_not(active_mask), dists >= req)
            all_valid = jp.all(per_pos_valid, axis=1)

            any_valid = jp.any(all_valid)
            idx = jp.argmax(all_valid.astype(jp.int32))
            chosen = candidates[idx]
            chosen = jax.lax.cond(any_valid, lambda x: x, lambda x: candidates[0], chosen)
            rkey, _ = jax.random.split(rkey)
            return chosen, rkey

        # Build layout: goal then hazards using lax.scan
        num_candidates = self._max_placement_attempts

        max_entries = 2 + self._num_hazards
        positions_xy = jp.zeros((max_entries, 2))
        keepouts = jp.zeros((max_entries,))
        positions_xy = positions_xy.at[0].set(agent_pos[:2])
        keepouts = keepouts.at[0].set(self._agent_keepout)
        count = jp.array(1, dtype=jp.int32)

        # Place goal
        goal_pos, rng_layout = _choose_valid_position(
            rng_layout,
            positions_xy,
            keepouts,
            count,
            self._goal_keepout,
            num_candidates,
        )
        positions_xy = positions_xy.at[count].set(goal_pos[:2])
        keepouts = keepouts.at[count].set(self._goal_keepout)
        count = count + 1

        # Scan over hazards
        def scan_fn(carry, _):
            rkey, pos_xy, ko, cnt = carry
            pos, rkey = _choose_valid_position(
                rkey,
                pos_xy,
                ko,
                cnt,
                self._hazard_keepout,
                num_candidates,
            )
            pos_xy = pos_xy.at[cnt].set(pos[:2])
            ko = ko.at[cnt].set(self._hazard_keepout)
            cnt = cnt + 1
            return (rkey, pos_xy, ko, cnt), pos

        (rng_layout, positions_xy, keepouts, count), hazards = jax.lax.scan(
            scan_fn,
            (rng_layout, positions_xy, keepouts, count),
            xs=jp.arange(self._num_hazards),
        )

        hazard_positions = hazards

        # Set mocaps
        data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(goal_pos))
        hazard_ids = jp.array(self._hazard_mocap_ids, dtype=jp.int32)

        def set_hazard(data_in, i, pos):
            mocap_id = hazard_ids[i]
            return data_in.replace(mocap_pos=data_in.mocap_pos.at[mocap_id].set(pos))

        def body_fn(d, idx):
            return set_hazard(d, idx, hazard_positions[idx])

        data = jax.lax.fori_loop(0, self._num_hazards, lambda i, d: body_fn(d, i), data)

        agent_pos = data.xpos[self._agent_body]
        initial_dist_goal = safe_norm(agent_pos[:2] - goal_pos[:2])

        info = {
            "goal_pos": goal_pos,
            "hazard_positions": hazard_positions,
            "step_count": 0,
            "last_dist_goal": initial_dist_goal,
            "goals_reached_count": 0,
            "goals_per_episode": 0,
            "cost": 0.0,
        }

        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = self._get_metrics(data, reward, 0.0, initial_dist_goal, initial_dist_goal, 0.0, 0.0, 0.0, 0.0)

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        last_dist_goal = state.info['last_dist_goal']

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Get positions
        agent_pos = data.xpos[self._agent_body]
        goal_pos = state.info['goal_pos']
        hazard_positions = state.info['hazard_positions']

        # Distances and reward components
        dist_goal = safe_norm(agent_pos[:2] - goal_pos[:2])
        dist_reward = (last_dist_goal - dist_goal) * self._reward_distance

        goal_achieved = dist_goal <= self._goal_size
        goal_reward = jp.where(goal_achieved, self._reward_goal, 0.0)

        ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight

        # Distance-weighted safety cost
        cost = self._calculate_safety_cost(agent_pos, hazard_positions)

        reward = dist_reward + goal_reward - ctrl_cost

        # Health check
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.logical_and(
            agent_pos[2] >= min_z,
            agent_pos[2] <= max_z
        ).astype(jp.float32)

        done = jp.logical_or(
            (1.0 - is_healthy) * self._terminate_when_unhealthy,
            jp.any(jp.isnan(agent_pos))
        )

        # Update goal if achieved (simple random reposition with min distance)
        rng_goal = jax.random.PRNGKey(state.info['step_count'])
        new_goal_pos = jax.random.uniform(
            rng_goal,
            (3,),
            minval=jp.array([-2.0, -2.0, 0.09]),
            maxval=jp.array([2.0, 2.0, 0.09]),
        )
        new_dist_to_agent = safe_norm(new_goal_pos[:2] - agent_pos[:2])
        new_goal_pos = jp.where(
            new_dist_to_agent < 1.0,
            agent_pos[:2] + (new_goal_pos[:2] - agent_pos[:2]) / (new_dist_to_agent + 1e-8) * 1.2,
            new_goal_pos[:2]
        )
        new_goal_pos = jp.array([new_goal_pos[0], new_goal_pos[1], 0.09])

        updated_goal_pos = jp.where(goal_achieved, new_goal_pos, goal_pos)
        updated_goals_reached = jp.where(goal_achieved,
                                         state.info['goals_reached_count'] + 1,
                                         state.info['goals_reached_count'])
        updated_goals_per_episode = jp.where(goal_achieved,
                                            state.info['goals_per_episode'] + 1,
                                            state.info['goals_per_episode'])
        goals_per_step = jp.where(goal_achieved, 1.0, 0.0)  # Binary indicator for this step

        if self._goal_mocap_id >= 0:
            data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(updated_goal_pos))

        new_last_dist_goal = jp.where(
            goal_achieved,
            safe_norm(agent_pos[:2] - updated_goal_pos[:2]),
            dist_goal
        )

        obs = self._get_obs(data)
        metrics = self._get_metrics(data, reward, cost, dist_goal, last_dist_goal, ctrl_cost, updated_goals_reached, updated_goals_per_episode, goals_per_step)

        new_info = state.info.copy()
        new_info.update({
            "goal_pos": updated_goal_pos,
            "step_count": state.info['step_count'] + 1,
            "last_dist_goal": new_last_dist_goal,
            "goals_reached_count": updated_goals_reached,
            "goals_per_episode": updated_goals_per_episode,
            "cost": cost,
        })

        return State(data, obs, reward, done.astype(jp.float32), metrics, new_info)

    def _sample_position_in_extents(self, rng_key: jp.ndarray, keepout: float = 0.0) -> jp.ndarray:
        min_x, min_y, max_x, max_y = self._placement_extents
        min_x = min_x + keepout
        min_y = min_y + keepout
        max_x = max_x - keepout
        max_y = max_y - keepout
        pos_x = jax.random.uniform(rng_key, minval=min_x, maxval=max_x)
        rng_key, subkey = jax.random.split(rng_key)
        pos_y = jax.random.uniform(subkey, minval=min_y, maxval=max_y)
        return jp.array([pos_x, pos_y, 0.09])

    def _calculate_safety_cost(self, agent_pos: jp.ndarray, hazard_positions: jp.ndarray) -> jp.ndarray:
        """Distance-weighted hazard cost (Safety-Gymnasium style).

        Sum over hazards of: hazard_cost_weight * max(0, hazard_size - distance).
        """
        if hazard_positions.shape[0] == 0:
            return jp.array(0.0)

        agent_pos_xy = agent_pos[:2]
        hazard_positions_xy = hazard_positions[:, :2]
        distances = jp.sqrt(jp.sum(jp.square(agent_pos_xy - hazard_positions_xy), axis=1) + 1e-8)

        per_hazard = self._hazard_cost_weight * jp.maximum(0.0, self._hazard_size - distances)
        return jp.sum(per_hazard)

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Creates an observation with separate lidars for goals and hazards.
        Structure identical to `SafePointGoal`.
        """
        agent_pos = data.xpos[self._agent_body]
        goal_pos = data.mocap_pos[self._goal_mocap_id]

        # 1. Sensors
        sensor_data = data.sensordata
        default_val = jp.zeros(3, dtype=sensor_data.dtype)

        accel_adr, accel_dim = self._sensor_info.get('accelerometer', (0, 0))
        accelerometer = jax.lax.dynamic_slice(sensor_data, (accel_adr,), (accel_dim,))
        accelerometer = jp.where(accel_dim == 3, accelerometer, default_val)

        velo_adr, velo_dim = self._sensor_info.get('velocimeter', (0, 0))
        velocimeter = jax.lax.dynamic_slice(sensor_data, (velo_adr,), (velo_dim,))
        velocimeter = jp.where(velo_dim == 3, velocimeter, default_val)

        gyro_adr, gyro_dim = self._sensor_info.get('gyro', (0, 0))
        gyro = jax.lax.dynamic_slice(sensor_data, (gyro_adr,), (gyro_dim,))
        gyro = jp.where(gyro_dim == 3, gyro, default_val)

        mag_adr, mag_dim = self._sensor_info.get('magnetometer', (0, 0))
        magnetometer = jax.lax.dynamic_slice(sensor_data, (mag_adr,), (mag_dim,))
        magnetometer = jp.where(mag_dim == 3, magnetometer, default_val)

        # 2. Agent-centric goal vector
        rel_goal_pos_3d_world = goal_pos - agent_pos
        agent_z_angle = data.qpos[2]
        cos_a = jp.cos(agent_z_angle)
        sin_a = jp.sin(agent_z_angle)
        world_dx_goal = rel_goal_pos_3d_world[0]
        world_dy_goal = rel_goal_pos_3d_world[1]
        agent_centric_dx_goal = world_dx_goal * cos_a + world_dy_goal * sin_a
        agent_centric_dy_goal = -world_dx_goal * sin_a + world_dy_goal * cos_a
        agent_centric_rel_goal_xy = jp.array([agent_centric_dx_goal, agent_centric_dy_goal])
        goal_comp = agent_centric_rel_goal_xy / (safe_norm(agent_centric_rel_goal_xy) + 1e-8)

        # 3. Lidars (goal and hazards), identical logic to SafePointGoal
        _lidar_num_bins = self._lidar_num_bins
        _lidar_max_dist = self._lidar_max_dist
        _lidar_alias = True

        goal_lidar_obs = jp.zeros(_lidar_num_bins)
        hazard_lidar_obs = jp.zeros(_lidar_num_bins)

        dx_goal = agent_centric_dx_goal
        dy_goal = agent_centric_dy_goal
        dist_goal = safe_norm(jp.array([dx_goal, dy_goal]))
        angle_goal = jp.arctan2(dy_goal, dx_goal)
        angle_goal = (angle_goal + 2 * jp.pi) % (2 * jp.pi)
        bin_size = (2 * jp.pi) / _lidar_num_bins
        bin_idx_float_goal = angle_goal / bin_size
        bin_idx_goal = jp.floor(bin_idx_float_goal)
        bin_idx_goal = jp.minimum(bin_idx_goal, _lidar_num_bins - 1).astype(int)
        sensor_val_goal = jp.maximum(0.0, _lidar_max_dist - dist_goal) / _lidar_max_dist
        sensor_val_goal = jp.where(dist_goal > _lidar_max_dist, 0.0, sensor_val_goal)
        goal_lidar_obs = goal_lidar_obs.at[bin_idx_goal].set(jp.maximum(goal_lidar_obs[bin_idx_goal], sensor_val_goal))
        if _lidar_alias:
            alias_factor_goal = bin_idx_float_goal - bin_idx_goal
            bin_plus_idx_goal = (bin_idx_goal + 1) % _lidar_num_bins
            goal_lidar_obs = goal_lidar_obs.at[bin_plus_idx_goal].set(
                jp.maximum(goal_lidar_obs[bin_plus_idx_goal], alias_factor_goal * sensor_val_goal)
            )
            bin_minus_idx_goal = (bin_idx_goal - 1 + _lidar_num_bins) % _lidar_num_bins
            goal_lidar_obs = goal_lidar_obs.at[bin_minus_idx_goal].set(
                jp.maximum(goal_lidar_obs[bin_minus_idx_goal], (1.0 - alias_factor_goal) * sensor_val_goal)
            )

        def process_hazard_lidar(carry, hazard_mocap_id):
            hazard_lidar, agent_pos, agent_z_angle, cos_a, sin_a = carry
            hazard_pos_3d = jp.where(
                hazard_mocap_id >= 0,
                data.mocap_pos[hazard_mocap_id],
                jp.array([0.0, 0.0, 0.0])
            )
            rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos
            world_dx_hazard = rel_hazard_pos_3d_world[0]
            world_dy_hazard = rel_hazard_pos_3d_world[1]
            agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
            agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a
            dist_hazard = safe_norm(jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard]))
            angle_hazard = jp.arctan2(agent_centric_dy_hazard, agent_centric_dx_hazard)
            angle_hazard = (angle_hazard + 2 * jp.pi) % (2 * jp.pi)
            bin_idx_float_hazard = angle_hazard / bin_size
            bin_idx_hazard = jp.floor(bin_idx_float_hazard)
            bin_idx_hazard = jp.minimum(bin_idx_hazard, _lidar_num_bins - 1).astype(int)
            sensor_val_hazard = jp.maximum(0.0, _lidar_max_dist - dist_hazard) / _lidar_max_dist
            sensor_val_hazard = jp.where(dist_hazard > _lidar_max_dist, 0.0, sensor_val_hazard)
            sensor_val_hazard = jp.where(hazard_mocap_id >= 0, sensor_val_hazard, 0.0)
            hazard_lidar = hazard_lidar.at[bin_idx_hazard].set(
                jp.maximum(hazard_lidar[bin_idx_hazard], sensor_val_hazard)
            )
            if _lidar_alias:
                alias_factor_hazard = bin_idx_float_hazard - bin_idx_hazard
                bin_plus_idx_hazard = (bin_idx_hazard + 1) % _lidar_num_bins
                hazard_lidar = hazard_lidar.at[bin_plus_idx_hazard].set(
                    jp.maximum(hazard_lidar[bin_plus_idx_hazard], alias_factor_hazard * sensor_val_hazard)
                )
                bin_minus_idx_hazard = (bin_idx_hazard - 1 + _lidar_num_bins) % _lidar_num_bins
                hazard_lidar = hazard_lidar.at[bin_minus_idx_hazard].set(
                    jp.maximum(hazard_lidar[bin_minus_idx_hazard], (1.0 - alias_factor_hazard) * sensor_val_hazard)
                )
            return (hazard_lidar, agent_pos, agent_z_angle, cos_a, sin_a), None

        hazard_mocap_ids_array = jp.array(self._hazard_mocap_ids + [-1] * (8 - len(self._hazard_mocap_ids)))[:8]
        init_carry = (hazard_lidar_obs, agent_pos, data.qpos[2], jp.cos(data.qpos[2]), jp.sin(data.qpos[2]))
        (hazard_lidar_obs, _, _, _, _), _ = jax.lax.scan(process_hazard_lidar, init_carry, hazard_mocap_ids_array)

        # Hazard compasses
        def compute_compass_for_hazard(mocap_idx):
            hazard_pos_3d = jp.where(
                mocap_idx >= 0,
                data.mocap_pos[mocap_idx],
                jp.array([0.0, 0.0, 0.0])
            )
            rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos
            world_dx_hazard = rel_hazard_pos_3d_world[0]
            world_dy_hazard = rel_hazard_pos_3d_world[1]
            cos_a = jp.cos(data.qpos[2])
            sin_a = jp.sin(data.qpos[2])
            agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
            agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a
            rel_vec = jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard])
            compass = rel_vec / (safe_norm(rel_vec) + 1e-8)
            return jp.where(mocap_idx >= 0, compass, jp.zeros(2))

        hazard_mocap_ids_for_compass = jp.array(self._hazard_mocap_ids + [-1] * (8 - len(self._hazard_mocap_ids)))[:8]
        hazard_compasses = jax.vmap(compute_compass_for_hazard)(hazard_mocap_ids_for_compass)
        hazard_compasses_flat = hazard_compasses.flatten()

        obs = jp.concatenate([
            accelerometer,
            velocimeter,
            gyro,
            magnetometer,
            goal_lidar_obs,
            hazard_lidar_obs,
            goal_comp,
            hazard_compasses_flat,
        ])

        return obs

    def _get_metrics(self, data: mjx.Data, reward: jp.ndarray, cost: jp.ndarray,
                    dist_goal: jp.ndarray, last_dist_goal: jp.ndarray, ctrl_cost: jp.ndarray,
                    goals_reached_count: jp.ndarray, goals_per_episode: jp.ndarray, goals_per_step: jp.ndarray) -> Dict:
        agent_pos = data.xpos[self._agent_body]
        return {
            'reward': reward,
            'cost': cost,
            'x_position': agent_pos[0],
            'y_position': agent_pos[1],
            'distance_to_goal': dist_goal,
            'last_dist_goal': last_dist_goal,
            'ctrl_cost': ctrl_cost,
            'goals_reached_count': jp.float32(goals_reached_count),
            'goals_per_episode': jp.float32(goals_per_episode),
            'goals_per_step': jp.float32(goals_per_step),
        }

    @property
    def observation_size(self) -> int:
        return 12 + self._lidar_num_bins * 2 + 2 + self._num_hazards * 2
