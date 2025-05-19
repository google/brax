
from typing import Any, Dict, Optional
import os
from ml_collections import config_dict
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.io import mjcf
from brax import math
from brax.envs.base import PipelineEnv, State

# Construct absolute path to point_hazard_goal.xml
_current_dir = os.path.dirname(os.path.abspath(__file__))
POINT_HAZARD_GOAL_XML_PATH = os.path.join(_current_dir, 'assets', 'point_hazard_goal_mocap.xml')

def default_config() -> config_dict.ConfigDict:
  """Returns the default config for PointHazardGoal environment."""
  config = config_dict.create(
      # New safety-gymnasium reward parameters
      reward_distance=2,  # Dense reward scale for distance moved to the goal
      reward_goal=10.0,      # Sparse reward for reaching the goal
      goal_size=0.7,        # Distance threshold for achieving the goal
      reward_orientation=False, # Optional: Reward for maintaining upright orientation
      reward_orientation_scale=0.002, # Scale for orientation reward
      reward_orientation_body='agent', # Body to check orientation (unused if reward_orientation=False)
      ctrl_cost_weight=0.001, # Weight for control cost
      hazard_size=0.7,       # Distance threshold for hazard cost
      # Other parameters (kept or adjusted)
      terminate_when_unhealthy=True, # Keep termination based on health
      healthy_z_range=(0.05, 0.3),    # Keep health definition
      reset_noise_scale=0.005,
      exclude_current_positions_from_observation=True,
      max_velocity=5.0,  # Keep velocity limit for calculation stability
      debug=False,
  )
  return config

def safe_norm(x, axis=None, keepdims=False, eps=1e-8):
  """Safely compute the norm with a small epsilon to avoid NaN."""
  return jp.sqrt(jp.sum(jp.square(x), axis=axis, keepdims=keepdims) + eps)

# Add JAX-compatible helper function for NaN handling
def nan_to_zero(x):
  """Replace NaN values with zeros in a JAX-compatible way."""
  return jp.where(jp.isnan(x), jp.zeros_like(x), x)

class PointResettingGoalRandomHazardSensorObs(PipelineEnv):

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Any]] = None,
      **kwargs,
  ):
    # Load the point model from XML
    mj_model = mujoco.MjModel.from_xml_path(POINT_HAZARD_GOAL_XML_PATH)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 4
    mj_model.opt.ls_iterations = 4

    # Get body IDs directly from MuJoCo model before loading into Brax
    try:
        self._agent_body_name = 'agent' # Store name for orientation reward if needed
        self._agent_body = mj_model.body(self._agent_body_name).id
        self._goal_body = mj_model.body('goal').id # Still useful for observation? Or use mocap? Mocap is used later.
  
        # Get hazard body IDs (kept for potential future use, but not reward)
        self._hazard_bodies = []
        for i in range(1, 5):
            hazard_name = f'hazard{i}'
            try:
                self._hazard_bodies.append(mj_model.body(hazard_name).id)
            except Exception:
                pass # Skip if hazard doesn't exist
    except Exception as e:
        print(f"Warning: Error getting body IDs: {e}")
        # Fallback to index-based approach
        self._agent_body_name = 'agent'
        self._agent_body = 1
        self._goal_body = 2
        self._hazard_bodies = [3, 4, 5, 6]

    sys = mjcf.load_model(mj_model)
    
    # Find the mocap ID for the goal body
    self._goal_mocap_id = None
    self._hazard_mocap_ids = []
    if mj_model.nmocap > 0:
        for i in range(mj_model.nbody):
            print(f"Body {i} name: {mj_model.body(i).name}, mocapid: {mj_model.body(i).mocapid}")
            if mj_model.body(i).name == "goal" and mj_model.body(i).mocapid >= 0:
                print(f"Goal body found with mocapid: {mj_model.body(i).mocapid}")
                self._goal_mocap_id = mj_model.body(i).mocapid
            for hazard_idx in range(1, 4):
                hazard_name = f"hazard{hazard_idx}"
                if mj_model.body(i).name == hazard_name and mj_model.body(i).mocapid >= 0:
                    print(f"Hazard body found with mocapid: {mj_model.body(i).mocapid}")
                    self._hazard_mocap_ids.append(mj_model.body(i).mocapid)
        # Fallback if 'goal' mocap body not found by name
        if mj_model.nmocap > 0 and self._goal_mocap_id is None:
            print("Warning: 'goal' body with mocapid >= 0 not found. Defaulting to mocapid 0.")
            self._goal_mocap_id = 0
        if mj_model.nmocap > 0 and len(self._hazard_mocap_ids) == 0:
            print("Warning: No hazard bodies with mocapid >= 0 found. Defaulting to mocapid 1, 2, 3.")
            self._hazard_mocap_ids = [1, 2, 3]
    else:
        print("Error: No mocap bodies defined in the model.")
        self._goal_mocap_id = -1 # Indicate error or invalid state
        self._hazard_mocap_ids = [-1, -1, -1]
    # --- Find Sensor Indices, Addresses, and Dimensions ---
    self._sensor_info = {}
    required_sensors = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
    sensor_found_flags = {name: False for name in required_sensors}
    if mj_model.nsensor > 0:
        print(f"Model has {mj_model.nsensor} sensors. Searching for required sensors...")
        for i in range(mj_model.nsensor):
            # Use mj_model directly as sys.mj_model might not be populated yet fully or easily accessible here
            name = mj_model.sensor(i).name
            if name in required_sensors:
                start_adr = mj_model.sensor_adr[i]
                dim = mj_model.sensor_dim[i]
                self._sensor_info[name] = (start_adr, dim)
                sensor_found_flags[name] = True
                print(f"  Found sensor: {name}, ID: {i}, Address: {start_adr}, Dim: {dim}")
    else:
        print("Warning: Model has no sensors defined (mj_model.nsensor = 0).")

    # Check if all required sensors were found
    missing_sensors = [name for name, found in sensor_found_flags.items() if not found]
    if missing_sensors:
        print(f"Error: Could not find the following required sensors: {missing_sensors}")
        # Depending on the desired behavior, you might raise an error here:
        # raise ValueError(f"Missing required sensors: {missing_sensors}")
        # Or provide default dummy values (e.g., index 0, dim 0) if you want to proceed cautiously
        # For now, we'll print the error and potentially fail later in _get_obs if accessed
        pass
    # --- End Sensor Info ---

    physics_steps_per_control_step = 4
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    # Apply config overrides if provided
    if config_overrides:
        config = config.copy_and_resolve_references()
        for k, v in config_overrides.items():
            config[k] = v

    self._config = config
    # Store new reward parameters
    self._reward_distance = config.reward_distance
    self._reward_goal = config.reward_goal
    self._goal_size = config.goal_size
    self._ctrl_cost_weight = config.ctrl_cost_weight
    self._reward_orientation = config.reward_orientation
    self._reward_orientation_scale = config.reward_orientation_scale
    self._reward_orientation_body = config.reward_orientation_body # Name stored

    # Store other necessary parameters
    self._terminate_when_unhealthy = config.terminate_when_unhealthy
    self._healthy_z_range = config.healthy_z_range
    self._reset_noise_scale = config.reset_noise_scale
    self._exclude_current_positions_from_observation = (
        config.exclude_current_positions_from_observation
    )
    self._max_velocity = config.max_velocity
    self._debug = config.debug
    self._hazard_size = config.hazard_size
  
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state with randomized goal."""
    rng, rng1, rng2, rng_goal, rng_hazards = jax.random.split(rng, 5)
    
    # Randomize initial point position with small noise
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )
    
    # Ensure qpos has valid quaternion
    if qpos.shape[0] > 6:  # If quaternion exists
      quat_norm = safe_norm(qpos[3:7])
      qpos = qpos.at[3:7].set(qpos[3:7] / quat_norm)

    # Initialize environment
    data = self.pipeline_init(qpos, qvel)
    
    # Randomize goal position (with more moderate bounds)
    # Generate a random goal position
    goal_pos = jax.random.uniform(
        rng_goal,
        (3,),
        minval=jp.array([-2.0, -2.0, 0.09]),
        maxval=jp.array([2.0, 2.0, 0.09]),
    )
    
    # Randomize hazard positions
    num_hazards = len(self._hazard_mocap_ids)
    # Create a dummy initial hazard_pos array in case num_hazards is 0
    hazard_positions = jp.zeros((max(1, num_hazards), 3))

    if num_hazards > 0:
        initial_agent_pos_for_hazard_check = data.xpos[self._agent_body] # Use initial qpos/qvel based data
        hazard_pos_updates = []
        rng_hazards = jax.random.split(rng_hazards, num_hazards)
        for i in range(num_hazards):
            key = rng_hazards[i]
            # Generate random hazard position
            hazard_pos_candidate = jax.random.uniform(
                key,
                (3,),
                minval=jp.array([-2.0, -2.0, 0.09]),
                maxval=jp.array([2.0, 2.0, 0.09]),
            )
            
            # Ensure hazard is at least 0.5 distance away from agent start
            dist_to_agent_start = safe_norm(hazard_pos_candidate[:2] - initial_agent_pos_for_hazard_check[:2])
            hazard_pos = jp.where(
                dist_to_agent_start < 0.5,
                initial_agent_pos_for_hazard_check[:2] + (hazard_pos_candidate[:2] - initial_agent_pos_for_hazard_check[:2]) / (dist_to_agent_start + 1e-8) * 0.7,
                hazard_pos_candidate[:2]
            )
            hazard_pos = jp.array([hazard_pos[0], hazard_pos[1], 0.09])
            hazard_pos_updates.append(hazard_pos)
        
        # Convert list of positions to a JAX array
        hazard_positions = jp.stack(hazard_pos_updates)
        
        # Update mocap positions for hazards
        # Assuming _hazard_mocap_ids are contiguous or handled correctly
        # We need to update multiple mocap positions. data.replace might not be ideal for indexed updates directly.
        # Let's update mocap_pos index by index
        for i, mocap_id in enumerate(self._hazard_mocap_ids):
             if mocap_id >= 0:
                 data = data.replace(mocap_pos=data.mocap_pos.at[mocap_id].set(hazard_positions[i]))

    # Get agent position (after potential mocap updates)
    agent_pos = data.xpos[self._agent_body]
    
    # Ensure goal is at least 1.0 distance away from agent
    dist_to_agent = safe_norm(goal_pos[:2] - agent_pos[:2])
    
    # If goal is too close, move it away from the agent
    goal_pos = jp.where(
        dist_to_agent < 1.0,
        # Normalize direction vector and set distance to 1.0 + small margin
        agent_pos[:2] + (goal_pos[:2] - agent_pos[:2]) / (dist_to_agent + 1e-8) * 1.2,
        goal_pos[:2]
    )
    
    # Ensure z-coordinate remains the same
    goal_pos = jp.array([goal_pos[0], goal_pos[1], 0.09])
    print(f"Reset method - Goal Position: {goal_pos}")
    # # Set goal position in the identified mocap body
    # # Add check to ensure _goal_mocap_id is valid
    if self._goal_mocap_id is not None and self._goal_mocap_id >= 0:
        data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(goal_pos))
    else:
        print("Error: Invalid or missing goal mocap ID. Cannot set goal position.")
        # Handle error case: maybe set done=True or raise exception
        # For now, we proceed but the goal won't be set correctly
        pass 

    # Calculate initial distance to goal for the first step
    initial_dist_goal = safe_norm(agent_pos[:2] - goal_pos[:2])
    
    # Store goal position and initial distance for reward calculation
    info = {
        "goal_pos": goal_pos,
        "hazard_positions": hazard_positions, # Store hazard positions
        "step_count": 0,
        "last_obs": None,
        "last_dist_goal": initial_dist_goal, # Initialize last_dist_goal
        "goals_reached_count": 0, # Initialize goal count
        "cost": 0.0, # Initialize cost
    }
    
    # Get observation
    obs = self._get_obs(data)
    
    # Store observation in info for stability check
    info["last_obs"] = obs
    
    reward, done, zero = jp.zeros(3)
    metrics = {
        # New reward components
        'dist_reward': zero,        # Reward from distance change
        'goal_reward': zero,        # Reward for reaching goal
        'orientation_reward': zero, # Reward for orientation (if enabled)
        'reward': zero,             # Total reward (expected by wrappers)
        'cost': zero,               # Cost metric
    
        # Other potentially useful metrics
        'x_position': jp.array(data.xpos[self._agent_body, 0]),
        'y_position': jp.array(data.xpos[self._agent_body, 1]),
        'distance_to_goal': initial_dist_goal,
        'ctrl_cost': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'z_alignment': zero, # Z-alignment metric (even if reward disabled)
        'goals_reached_count': 0.0, # Track goals reached
    }
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    # Get the last valid observation and distance
    last_obs = state.info.get('last_obs', state.obs)
    last_dist_goal = state.info['last_dist_goal']
    goals_reached_count = state.info['goals_reached_count']
    
    # Increment step counter
    step_count = state.info.get('step_count', 0) + 1
    
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    # Get positions
    agent_pos = data.xpos[self._agent_body]
    goal_pos = state.info['goal_pos']
    hazard_positions = state.info['hazard_positions']
    
    # --- Calculate Reward Components --- 

    # 1. Distance-Based Reward
    dist_goal = safe_norm(agent_pos[:2] - goal_pos[:2])
    dist_reward = (last_dist_goal - dist_goal) * self._reward_distance
    
    # 2. Goal Achievement Reward
    goal_achieved = dist_goal <= self._goal_size
    goal_reward = jp.where(goal_achieved, self._reward_goal, 0.0)

    # --- Handle Goal Reset --- 
    rng_goal = jax.random.PRNGKey(step_count)

    # Generate a potential new goal position
    new_goal_pos_candidate = jax.random.uniform(
        rng_goal,
        (3,),
        minval=jp.array([-2.0, -2.0, 0.09]),
        maxval=jp.array([2.0, 2.0, 0.09]),
    )
    # Ensure new goal is far enough from current agent pos
    new_dist_to_agent = safe_norm(new_goal_pos_candidate[:2] - agent_pos[:2])
    new_goal_pos = jp.where(
        new_dist_to_agent < 1.0,
        agent_pos[:2] + (new_goal_pos_candidate[:2] - agent_pos[:2]) / (new_dist_to_agent + 1e-8) * 1.2,
        new_goal_pos_candidate[:2]
    )
    new_goal_pos = jp.array([new_goal_pos[0], new_goal_pos[1], 0.09])

    # Update goal position, count, and mocap if goal achieved
    new_goals_reached_count = jp.where(goal_achieved, goals_reached_count + 1, goals_reached_count)
    updated_goal_pos = jp.where(goal_achieved, new_goal_pos, goal_pos)
    
    # Conditionally compute the new mocap_pos array
    condition = goal_achieved & (self._goal_mocap_id is not None) & (self._goal_mocap_id >= 0)
    new_mocap_pos_if_goal = data.mocap_pos.at[self._goal_mocap_id].set(updated_goal_pos)
    final_mocap_pos = jp.where(condition, new_mocap_pos_if_goal, data.mocap_pos)
    # Create the final data object with the potentially updated mocap_pos
    updated_data = data.replace(mocap_pos=final_mocap_pos)
    
    # Update last_dist_goal based on the potentially new goal position
    new_last_dist_goal = jp.where(
        goal_achieved,
        safe_norm(agent_pos[:2] - updated_goal_pos[:2]), # Dist to new goal
        dist_goal # Dist to old goal (since it wasn't achieved)
    )

    # 3. Orientation Reward (Optional)
    z_alignment = 0.0
    orientation_reward = 0.0
    if self._reward_orientation:
        # Get quaternion of the agent body (assuming it's the first part of qpos)
        # Check if the orientation body exists in the model
        # Note: This assumes the agent's quaternion is at the start of qpos after potential exclusion.
        #       Adjust indexing if necessary based on your MJCF structure and observation exclusion.
        quat = data.qpos[3:7] # Assuming standard floating base: [x, y, z, qw, qx, qy, qz, ...] -> qw, qx, qy, qz indices 3-7
        
        # Normalize quaternion
        quat_norm = safe_norm(quat)
        quat = quat / jp.maximum(quat_norm, 1e-8)
        
        # Calculate rotation matrix
        rot_matrix = math.quat_to_rot(quat)
        
        # zalign = R[2, 2]
        z_alignment = rot_matrix[2, 2]
        orientation_reward = self._reward_orientation_scale * z_alignment
      
    ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight
            
    # --- Cost Calculation --- 
    cost = 0.0
    if hazard_positions.shape[0] > 0 and self._hazard_size > 0: # Only calculate if hazards exist and size is positive
        # Calculate distances to all hazards (vectorized)
        # agent_pos_xy needs shape (2,) 
        # hazard_positions_xy needs shape (num_hazards, 2)
        agent_pos_xy = agent_pos[:2] 
        hazard_positions_xy = hazard_positions[:, :2] 
        
        # Calculate squared distances first, then take sqrt for efficiency
        dist_sq_to_hazards = jp.sum(jp.square(agent_pos_xy - hazard_positions_xy), axis=1)
        dist_to_hazards = jp.sqrt(dist_sq_to_hazards + 1e-8) # Add epsilon for safety
        
        # Check if agent is inside any hazard zone
        inside_any_hazard = jp.any(dist_to_hazards <= self._hazard_size)
        
        # Cost is 1.0 if inside any hazard, 0.0 otherwise
        cost = jp.where(inside_any_hazard, 1.0, 0.0)

    # --- Total Reward --- 
    reward = dist_reward + goal_reward + orientation_reward - ctrl_cost

    # --- Health Check (for termination, not reward) ---
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.logical_and(
        agent_pos[2] >= min_z,
        agent_pos[2] <= max_z
    ).astype(jp.float32)

    # --- Termination Conditions ---
    # Check for NaN in state using JAX-compatible operations
    # Calculate velocity for NaN check (keep calculation local if only needed here)
    dt = jp.maximum(self.dt, 1e-6)  # Ensure dt is not too small
    velocity = (agent_pos - data0.xpos[self._agent_body]) / dt
    has_nan_pos = jp.any(jp.isnan(agent_pos))
    has_nan_vel = jp.any(jp.isnan(velocity))
    has_nan_state = jp.logical_or(has_nan_pos, has_nan_vel)
    
    # Terminate if unhealthy (optional), 3 goals reached, or NaN state
    done = jp.logical_or(
        (1.0 - is_healthy) * self._terminate_when_unhealthy,
        has_nan_state
    )
    
    # --- Observation and State Update ---
    obs = self._get_obs(data)
    
    # Handle NaN observation using JAX-compatible operations
    has_nan_obs = jp.any(jp.isnan(obs))
    obs = jp.where(has_nan_obs, last_obs, obs)

    # Update info dictionary - update existing dict instead of replacing
    # Ensure all keys from input state.info are preserved
    new_info = state.info.copy() # Start with a copy of the input info
    new_info.update({
        "goal_pos": updated_goal_pos,       # Update goal position (usually static but good practice)
        "hazard_positions": hazard_positions, # Keep hazard positions (they don't move)
        "step_count": step_count,    # Update internal step counter if needed
        "last_obs": obs,             # Store current observation for next step's fallback
        "last_dist_goal": new_last_dist_goal, # Store current distance for next step's reward calculation
        "goals_reached_count": new_goals_reached_count,
        "cost": cost, # Update cost in info
    })

    # Update metrics safely
    metrics = {
        'dist_reward': dist_reward,
        'goal_reward': goal_reward,
        'orientation_reward': orientation_reward,
        'ctrl_cost': ctrl_cost, # Added control cost
        'reward': reward, # Add total reward to metrics
        'cost': cost, # Add cost metric
        'x_position': agent_pos[0],
        'y_position': agent_pos[1],
        'distance_to_goal': dist_goal,
        'x_velocity': velocity[0],
        'y_velocity': velocity[1],
        'z_alignment': z_alignment, # Log z_alignment
        'goals_reached_count': new_goals_reached_count.astype(jp.float32), # Track goals reached, ensure float32
    }

    # Create fresh State
    return State(updated_data, obs, reward, done.astype(jp.float32), metrics, new_info)

  def _get_obs(self, data: mjx.Data) -> jp.ndarray:  
    """Creates an observation that matches Safety-Gymnasium Goal0 format."""  
    agent_pos = data.xpos[self._agent_body]  
      
    # Get goal position from the identified mocap body  
    if self._goal_mocap_id is not None and self._goal_mocap_id >= 0:  
        goal_pos_3d = data.mocap_pos[self._goal_mocap_id]  
    else:  
        # Fallback  
        goal_pos_3d = jp.zeros(3)  
      
    # 1. Agent sensor observations  
    # Access the flat sensordata array
    sensor_data = data.sensordata

    # Extract sensor values using pre-calculated addresses and dimensions
    # Handle potential missing sensors by providing default zero vectors if info not found
    # (Assumes sensors are 3D, adjust if necessary)
    default_val = jp.zeros(3, dtype=sensor_data.dtype)

    accel_adr, accel_dim = self._sensor_info.get('accelerometer', (0, 0))
    accelerometer = jax.lax.dynamic_slice(sensor_data, (accel_adr,), (accel_dim,))
    accelerometer = jp.where(accel_dim == 3, accelerometer, default_val) # Ensure correct shape

    velo_adr, velo_dim = self._sensor_info.get('velocimeter', (0, 0))
    velocimeter = jax.lax.dynamic_slice(sensor_data, (velo_adr,), (velo_dim,))
    velocimeter = jp.where(velo_dim == 3, velocimeter, default_val) # Ensure correct shape

    gyro_adr, gyro_dim = self._sensor_info.get('gyro', (0, 0))
    gyro = jax.lax.dynamic_slice(sensor_data, (gyro_adr,), (gyro_dim,))
    gyro = jp.where(gyro_dim == 3, gyro, default_val) # Ensure correct shape

    mag_adr, mag_dim = self._sensor_info.get('magnetometer', (0, 0))
    magnetometer = jax.lax.dynamic_slice(sensor_data, (mag_adr,), (mag_dim,))
    magnetometer = jp.where(mag_dim == 3, magnetometer, default_val) # Ensure correct shape

    # 2. Calculate relative position to goal (world frame)
    rel_goal_pos_3d_world = goal_pos_3d - agent_pos # World frame
    
    # Ensure rel_goal_pos_3d_world is (3,) before further slicing for Lidar
    rel_goal_pos_3d_world_lidar = rel_goal_pos_3d_world.reshape(3)

    # --- Agent-centric transformation --- 
    # Get agent's current Z rotation from qpos (assuming 3rd DoF is Z rotation)
    # Point agent qpos is [x_slide, y_slide, z_hinge_angle]
    agent_z_angle = data.qpos[2] 
    cos_a = jp.cos(agent_z_angle)
    sin_a = jp.sin(agent_z_angle)

    # World-frame relative XY vector to goal
    world_dx_goal = rel_goal_pos_3d_world_lidar[0]
    world_dy_goal = rel_goal_pos_3d_world_lidar[1]

    # Transform world-frame relative vector to agent's local frame
    # local_dx = world_dx * cos(agent_z_angle) + world_dy * sin(agent_z_angle)
    # local_dy = -world_dx * sin(agent_z_angle) + world_dy * cos(agent_z_angle)
    agent_centric_dx_goal = world_dx_goal * cos_a + world_dy_goal * sin_a
    agent_centric_dy_goal = -world_dx_goal * sin_a + world_dy_goal * cos_a
    # --- End Agent-centric transformation ---

    # 3. Create compass observation (agent-centric)
    agent_centric_rel_goal_xy = jp.array([agent_centric_dx_goal, agent_centric_dy_goal])
    goal_comp = agent_centric_rel_goal_xy / (safe_norm(agent_centric_rel_goal_xy) + 1e-8)
    # goal_comp = goal_comp.reshape(2) # Should already be (2,) after normalization
        
    # 4. Create Safety-Gymnasium style Lidar with 16 bins
    # Lidar configuration (can be moved to self._config or class attributes later)
    _lidar_num_bins = 16
    _lidar_max_dist = 3.0  # Max distance Lidar can see, objects further are not seen
    _lidar_exp_gain = 1.0 # Not used if max_dist is set, but for compatibility
    _lidar_alias = True   # Enable/disable aliasing

    # Initialize Lidar observation
    lidar_obs = jp.zeros(_lidar_num_bins)
    
    # For now, Lidar only "sees" the goal.
    # To see multiple objects (e.g., hazards), you would loop here over their positions.
    # Object position for Lidar (goal's XY in world frame, as agent is point and assumed not rotating)
    # agent_pos_xy_world = agent_pos[:2] # Not directly needed if using rel_goal_pos

    # Use the agent-centric dx and dy for Lidar angle calculation
    dx = agent_centric_dx_goal 
    dy = agent_centric_dy_goal

    # ---- JAX DEBUG PRINT START ----
    # jax.debug.print("OBS dx: {dx}", dx=dx)
    # jax.debug.print("OBS dy: {dy}", dy=dy)
    # ---- JAX DEBUG PRINT END ----

    dist = safe_norm(jp.array([dx, dy]))
    
    # ---- JAX DEBUG PRINT START ----
    # jax.debug.print("OBS dist: {dist}", dist=dist)
    # ---- JAX DEBUG PRINT END ----

    angle = jp.arctan2(dy, dx) # Angle from positive x-axis, range [-pi, pi]
    angle = (angle + 2 * jp.pi) % (2 * jp.pi) # Convert to [0, 2*pi]

    bin_size = (2 * jp.pi) / _lidar_num_bins
    
    # Determine which bin the object falls into
    # Subtracting a small epsilon before casting to int to handle edge cases near bin boundaries
    # and ensure angle / bin_size = _lidar_num_bins goes to bin _lidar_num_bins -1
    bin_idx_float = angle / bin_size
    bin_idx = jp.floor(bin_idx_float) 
    bin_idx = jp.minimum(bin_idx, _lidar_num_bins - 1).astype(int) # Ensure it's within [0, num_bins-1]

    # ---- JAX DEBUG PRINT START ----
    # jax.debug.print("OBS angle: {angle}", angle=angle)
    # jax.debug.print("OBS bin_idx: {bin_idx}", bin_idx=bin_idx)
    # ---- JAX DEBUG PRINT END ----


    # Calculate sensor reading (linear decay "closeness")
    # Ensure dist is not greater than max_dist for sensor calculation, effectively capping vision
    # Sensor value is 0 if dist > _lidar_max_dist
    sensor_val = jp.maximum(0.0, _lidar_max_dist - dist) / _lidar_max_dist
    sensor_val = jp.where(dist > _lidar_max_dist, 0.0, sensor_val)

    # ---- JAX DEBUG PRINT START ----
    # jax.debug.print("OBS sensor_val: {sensor_val}", sensor_val=sensor_val)
    # ---- JAX DEBUG PRINT END ----


    # Update the Lidar observation for the primary bin
    # Using .at[bin_idx].set() for JAX compatibility with arrays
    lidar_obs = lidar_obs.at[bin_idx].set(jp.maximum(lidar_obs[bin_idx], sensor_val))

    if _lidar_alias:
        # Calculate alias interpolation factor
        # alias_factor is how far into the current bin the angle is (0 to 1)
        alias_factor = bin_idx_float - bin_idx 
        
        # Bin plus one (wraps around)
        bin_plus_idx = (bin_idx + 1) % _lidar_num_bins
        lidar_obs = lidar_obs.at[bin_plus_idx].set(
            jp.maximum(lidar_obs[bin_plus_idx], alias_factor * sensor_val)
        )
        
        # Bin minus one (wraps around)
        # For the "1 - alias_factor" part, it represents how much it belongs to the previous part of the bin,
        # which is equivalent to the "next" bin in the opposite direction of aliasing.
        # The SG code does (1-alias)*sensor for bin_minus, implying how much it's "not" in the next part.
        # Let's stick to the SG logic: sensor proportional to how close to bin center.
        # If alias_factor is small (close to start of current bin), it's more in current bin.
        # If alias_factor is large (close to end of current bin), it's more in next bin.
        # SG: obs[bin_plus] = max(obs[bin_plus], alias * sensor)
        #     obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        # This seems to distribute based on the fractional part.
        
        bin_minus_idx = (bin_idx - 1 + _lidar_num_bins) % _lidar_num_bins # Ensure positive index before modulo
        lidar_obs = lidar_obs.at[bin_minus_idx].set(
            jp.maximum(lidar_obs[bin_minus_idx], (1.0 - alias_factor) * sensor_val)
        )
          
    # Build observation matching Goal0 format  
    obs = jp.concatenate([  
        accelerometer,  
        velocimeter,  
        gyro,  
        magnetometer,  
        lidar_obs,  # New Lidar
        goal_comp,  
    ])  
      
    return obs
