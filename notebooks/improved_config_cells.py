# ==========================================
# IMPROVED NOTEBOOK CELLS - COPY THESE TO YOUR NOTEBOOK
# ==========================================

# Cell 2: Improved Environment Config
def improved_config() -> config_dict.ConfigDict:
    """Returns improved config for PointHazardGoal environment."""
    config = config_dict.create(
        # IMPROVED: Better reward balance
        reward_distance=1.0,    # REDUCED from 2 - less penalty for exploration
        reward_goal=20.0,       # INCREASED from 10 - stronger goal incentive
        goal_size=0.5,          # REDUCED from 0.7 - encourage precision
        
        # IMPROVED: Control and safety
        ctrl_cost_weight=0.005, # INCREASED from 0.001 - smoother actions
        hazard_size=0.6,        # REDUCED from 0.7 - easier navigation
        
        # IMPROVED: Add orientation reward for stability  
        reward_orientation=True,           # Enable for stability
        reward_orientation_scale=0.01,     # Small positive reward for staying upright
        
        # Keep other parameters
        terminate_when_unhealthy=True,
        healthy_z_range=(0.05, 0.3),
        reset_noise_scale=0.005,
        exclude_current_positions_from_observation=True,
        max_velocity=5.0,
        debug=False,
    )
    return config

# Cell 5: Improved Training Args (MOST CRITICAL CHANGES)
args = Args(
    # CRITICAL: Fix reward scaling - this is your biggest issue
    reward_scaling=1.0,          # INCREASED from 0.1 - reward values were too small
    
    # IMPROVED: Faster learning
    episode_length=1500,         # REDUCED from 2000 - shorter episodes
    num_timesteps=25_000_000,    # REDUCED from 30M - stop before overfitting
    
    # IMPROVED: Exploration and stability
    learning_rate=1e-4,          # REDUCED from 3e-4 - more stable learning
    entropy_cost=5e-3,           # INCREASED from 1e-3 - encourage exploration
    discounting=0.99,            # INCREASED from 0.97 - value long-term rewards
    
    # IMPROVED: Batch settings for stability
    num_envs=1024,               # REDUCED from 2048 - more stable
    batch_size=512,              # REDUCED from 1024 - smaller batches
    num_updates_per_batch=8,     # INCREASED from 4 - more updates per batch
    
    # Keep other parameters
    num_evals=5,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    max_devices_per_host=None,
    seed=0
)

# Alternative: If you want even faster results, try this aggressive config:
args_aggressive = Args(
    reward_scaling=2.0,          # Even higher reward signal
    episode_length=1000,         # Very short episodes  
    num_timesteps=20_000_000,    # Shorter training
    learning_rate=5e-4,          # Faster learning
    entropy_cost=1e-2,           # High exploration
    discounting=0.98,            # Moderate discounting
    num_envs=2048,               # Keep high parallelization
    batch_size=2048,             # Large batches
    num_updates_per_batch=2,     # Fewer updates per batch
    # ... other params same as above
)

# ==========================================
# USAGE INSTRUCTIONS:
# ==========================================
# 1. Replace your Cell 2 with the improved_config() function
# 2. Replace your Cell 5 args with the improved args above  
# 3. Update your default_config() call to use improved_config()
# 4. Run training and monitor the results
#
# Key metrics to watch:
# - episode/reward should increase more steadily
# - episode/goal_reward should stay high when goals are reached
# - episode/dist_reward should be less negative
# - eval/episode_reward should peak higher and stay more stable
# ==========================================

# Updated environment instantiation (replace in your Cell 3):
# Create environment with improved config
env_config_overrides = {
    "reward_distance": 1.0,
    "reward_goal": 20.0, 
    "goal_size": 0.5,
    "ctrl_cost_weight": 0.005,
    "hazard_size": 0.6,
    "reward_orientation": True,
    "reward_orientation_scale": 0.01,
}

train_environment = envs.get_environment(env_name, config_overrides=env_config_overrides)
eval_env = envs.get_environment(env_name, config_overrides=env_config_overrides) 