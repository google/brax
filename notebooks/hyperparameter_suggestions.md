# Hyperparameter Tuning Suggestions for PointResettingGoalRandomHazardLidarSensorObs

## Current Issues from Training Charts

1. **Goal reward declining** while goals_reached_count increases
2. **Distance reward becoming negative** 
3. **High cost** (agent in hazards too often)
4. **Performance degradation** after 15-20M steps
5. **Training instability** and overfitting

## Environment Config Improvements

### Current Environment Config:
```python
def default_config() -> config_dict.ConfigDict:
  config = config_dict.create(
      reward_distance=2,      # CURRENT
      reward_goal=10.0,       # CURRENT  
      goal_size=0.7,          # CURRENT
      ctrl_cost_weight=0.001, # CURRENT
      hazard_size=0.7,        # CURRENT
  )
```

### Suggested Environment Config Changes:

```python
def improved_config() -> config_dict.ConfigDict:
  config = config_dict.create(
      # Reward tuning for better balance
      reward_distance=1.0,    # REDUCED from 2 - less penalty for inefficient paths
      reward_goal=20.0,       # INCREASED from 10 - stronger goal incentive
      goal_size=0.5,          # REDUCED from 0.7 - make goals slightly harder to encourage precision
      
      # Control cost
      ctrl_cost_weight=0.005, # INCREASED from 0.001 - encourage smoother actions
      
      # Safety parameters  
      hazard_size=0.6,        # REDUCED from 0.7 - make hazards slightly smaller for easier navigation
      
      # Optional: Add orientation reward for stability
      reward_orientation=True,           # Enable orientation reward
      reward_orientation_scale=0.01,     # Small positive reward for staying upright
  )
```

## Training Config Improvements

### Current Training Args:
```python
args = Args(
    num_timesteps=30_000_000,    # CURRENT
    reward_scaling=0.1,          # CURRENT - TOO LOW
    episode_length=2000,         # CURRENT
    learning_rate=3e-4,          # CURRENT  
    entropy_cost=1e-3,           # CURRENT
    discounting=0.97,            # CURRENT
    num_envs=2048,               # CURRENT
    batch_size=1024,             # CURRENT
)
```

### Suggested Training Args Changes:

```python
# Option 1: Stability-focused (recommended)
args_stable = Args(
    num_timesteps=25_000_000,    # REDUCED - stop before overfitting
    reward_scaling=1.0,          # INCREASED from 0.1 - reward values too small
    episode_length=1500,         # REDUCED from 2000 - shorter episodes for faster learning
    learning_rate=1e-4,          # REDUCED from 3e-4 - more stable learning  
    entropy_cost=5e-3,           # INCREASED from 1e-3 - encourage exploration
    discounting=0.99,            # INCREASED from 0.97 - value long-term rewards more
    
    # Regularization
    num_envs=1024,               # REDUCED from 2048 - less parallelization, more stable
    batch_size=512,              # REDUCED from 1024 - smaller batches for stability
    num_updates_per_batch=8,     # INCREASED from 4 - more updates per batch
    
    # New parameters for stability
    max_grad_norm=0.5,           # Add gradient clipping
    normalize_advantages=True,   # Normalize advantages
)

# Option 2: Fast-learning focused (if you want quicker results)
args_fast = Args(
    num_timesteps=20_000_000,    # REDUCED 
    reward_scaling=2.0,          # HIGHER - stronger reward signal
    episode_length=1000,         # SHORTER - faster episodes
    learning_rate=5e-4,          # HIGHER - faster learning
    entropy_cost=1e-2,           # HIGHER - more exploration
    discounting=0.98,            # MODERATE
    
    num_envs=2048,               # KEEP - parallel learning
    batch_size=2048,             # INCREASED - larger batches
    num_updates_per_batch=2,     # REDUCED - fewer updates per batch
)
```

## Curriculum Learning Approach

For even better results, consider a curriculum approach:

```python
# Stage 1: Learn basic navigation (no hazards initially)
stage1_config = config.copy()
stage1_config.hazard_size = 0.0  # Disable hazards temporarily

# Stage 2: Introduce small hazards  
stage2_config = config.copy()
stage2_config.hazard_size = 0.4  # Smaller hazards

# Stage 3: Full difficulty
stage3_config = config.copy()
stage3_config.hazard_size = 0.6  # Full hazards
```

## Learning Rate Scheduling

```python
# Add learning rate decay
def lr_schedule(step):
    if step < 5_000_000:
        return 3e-4  # Initial LR
    elif step < 15_000_000:
        return 1e-4  # Reduce after 5M steps
    else:
        return 5e-5  # Fine-tune in final stages
```

## Key Changes Rationale

1. **Increased reward_scaling (0.1 → 1.0)**: Your rewards are too small relative to the value function
2. **Increased goal_reward (10 → 20)**: Stronger incentive to reach goals
3. **Reduced reward_distance (2 → 1)**: Less penalty for exploration
4. **Increased entropy_cost**: More exploration to find better policies
5. **Higher discounting**: Value long-term goal reaching over short-term rewards
6. **Shorter episodes**: Faster learning cycles
7. **Smaller batches**: More stable gradient updates

## Monitoring Improvements

Add these metrics to track training health:

```python
# In your progress function, also log:
- Learning rate
- Gradient norms  
- Policy entropy
- Value function variance
- Episode success rate (goals/episode)
- Average time to first goal
```

## Implementation Priority

1. **Start with reward_scaling=1.0** - this is likely your biggest issue
2. **Increase goal_reward to 20** - stronger goal incentive  
3. **Reduce episode_length to 1500** - faster learning
4. **Increase entropy_cost to 5e-3** - more exploration

Try these changes first, then adjust other parameters based on results! 