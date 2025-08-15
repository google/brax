# Velocity-Constrained Ant Locomotion

This project implements a velocity-constrained ant locomotion environment for safe reinforcement learning using PPO Lagrange v2. The ant must learn to move forward while respecting velocity constraints.

## Overview

The velocity-constrained ant extends the standard Brax ant environment by adding velocity constraints that serve as safety constraints for constrained reinforcement learning. The agent receives a cost (constraint violation) when its velocity exceeds a specified threshold.

## Files Created

### 1. Environment Implementation
- **`brax/envs/ant_velocity_constrained.py`**: The velocity-constrained ant environment
- **`brax/envs/__init__.py`**: Updated to register the new environment

### 2. Training Notebook
- **`notebooks/ant_velocity_constrained_training.ipynb`**: Clean training notebook with minimal dependencies

## Environment Features

### Velocity Constraint
- **Maximum velocity**: Configurable velocity limit (default: 2.0 m/s)
- **Cost function**: Constraint violation cost when velocity exceeds the limit
- **Constraint type**: Soft constraint with penalty-based enforcement

### Reward Structure
- **Forward reward**: Encourages movement in the +x direction
- **Survival reward**: Reward for staying healthy/upright
- **Control cost**: Penalty for large actions
- **Velocity cost**: Penalty for constraint violations (this becomes the "cost" for PPO Lagrange)

### Key Metrics
- `velocity_magnitude`: Current velocity magnitude
- `velocity_violation`: Amount by which velocity exceeds the limit
- `velocity_cost`: Cost incurred due to constraint violation
- `cost`: Total constraint violation (used by PPO Lagrange)

### Technical Features
- **Vectorization Support**: Handles both single and multi-environment setups seamlessly
- **Shape Consistency**: All metrics maintain consistent JAX array shapes for scan compatibility
- **JAX Compatibility**: Optimized for JAX transformations (vmap, scan, jit)
- **Training Stability**: Fixed shape mismatches that caused scan errors during training
- **Proper Cost Handling**: Cost is maintained in both `state.info` and `state.metrics` for PPO Lagrange v2
- **Constraint Calibration**: Default max velocity (0.8 m/s) set below ant's natural maximum (~0.81 m/s)

## PPO Lagrange v2 Training

The environment is designed to work with PPO Lagrange v2, which uses Lagrange multipliers to handle safety constraints:

### Key Parameters

**Environment Parameters:**
- **`max_velocity`**: Maximum allowed velocity in m/s (default: 0.8)
- **`velocity_cost_weight`**: Weight for velocity violation cost (default: 1.0)

**PPO Lagrange Parameters:**
- **`safety_bound`**: Maximum allowed average cost per episode
- **`lagrangian_coef_rate`**: Learning rate for Lagrange multiplier updates
- **`initial_lambda_lagr`**: Initial value of the Lagrange multiplier

**Constraint Calibration:**
The default `max_velocity=0.8` is set slightly below the ant's natural maximum velocity (~0.81 m/s) to ensure constraint violations occur during training, allowing the Lagrange multiplier to increase appropriately.

### Training Process
1. Agent learns to maximize reward while minimizing constraint violations
2. Lagrange multiplier (λ) automatically adjusts based on constraint satisfaction
3. Higher λ values make the agent more conservative
4. Lower λ values allow more aggressive behavior

## Usage

### Basic Usage
```python
import brax.envs as envs

# Create environment with custom velocity limit
env = envs.get_environment(
    'ant_velocity_constrained',
    max_velocity=1.5,  # m/s
    velocity_cost_weight=1.0
)
```

### Training with PPO Lagrange v2
```python
from brax.training.agents.ppo_lagrange_v2 import train as ppo_lagrange_v2

# Configure training
make_inference_fn, params, metrics = ppo_lagrange_v2(
    environment=train_env,
    eval_env=eval_env,
    num_timesteps=10_000_000,
    safety_bound=0.5,  # Maximum allowed constraint violation
    lagrangian_coef_rate=0.01,
    # ... other parameters
)
```

### Running the Notebook
1. Open `notebooks/ant_velocity_constrained_training.ipynb`
2. The notebook will automatically validate GPU and MuJoCo installation
3. Configure environment and training parameters using the Args class (cell 2)
4. Update W&B project name if desired (defaults to 'velocity_constrained_ant')
5. Run all cells to train the model and generate plots

### Enhanced Features (Inspired by mourad_lag.ipynb)
The notebook includes comprehensive enhancements for professional experiment tracking:

**Setup & Validation:**
- GPU detection and validation
- MuJoCo installation verification
- Enhanced imports with mujoco integration
- XLA optimizations for performance

**Configuration Management:**
- Args class for clean parameter management
- Environment-specific W&B configuration
- Training metrics logging with proper intervals
- Cost wrapper for PPO Lagrange v2 compatibility

**Advanced Progress Tracking:**
- Enhanced progress function with metric categorization
- Real-time constraint violation monitoring
- Lambda (Lagrange multiplier) evolution tracking
- Velocity-specific metrics logging

**Weights & Biases Integration:**
- **Automatic initialization**: Run name includes timestamp
- **Real-time logging**: All training and evaluation metrics
- **Metric categorization**: Organized into training/, eval/, episode/ prefixes
- **Final metrics**: Training summary and rollout results
- **Plot uploads**: Training progress and rollout analysis plots uploaded
- **Run finishing**: Proper cleanup at notebook completion

## Recent Fixes (v2)

### Lambda Not Increasing Issue Fixed
The environment has been updated to properly handle cost signals for PPO Lagrange v2:

**Root Cause**: Cost was not properly maintained in `state.info`, which PPO Lagrange v2 requires
**Solution**: 
1. Added proper `state.info` initialization and maintenance in both `reset()` and `step()`
2. Cost now stored in both `state.info` and `state.metrics`
3. Calibrated `max_velocity=0.8` to ensure constraint violations occur during training
4. Fixed JAX scan compatibility issues with consistent array shapes

**Result**: Lambda (Lagrange multiplier) now properly increases when constraints are violated, enabling effective constrained RL training.

### ConcretizationTypeError Fixed
**Root Cause**: Calling `.item()` on JAX arrays during compilation/tracing caused `ConcretizationTypeError`
**Solution**: 
1. Removed `.item()` calls from environment code
2. Keep JAX arrays in `state.info` instead of converting to Python scalars
3. Eliminated need for cost wrapper - environment handles cost directly

**Result**: Training now runs without JAX compilation errors.

## Configuration Options

### Environment Parameters
- **`max_velocity`**: Maximum allowed velocity (m/s)
- **`velocity_cost_weight`**: Weight for velocity constraint violation
- **`ctrl_cost_weight`**: Weight for control cost
- All standard ant environment parameters

### Training Parameters
- **`safety_bound`**: Constraint violation limit
- **`lagrangian_coef_rate`**: Lagrange multiplier learning rate
- **`initial_lambda_lagr`**: Initial Lagrange multiplier value
- All standard PPO parameters

## Expected Behavior

### Successful Training
- Agent learns to move forward efficiently
- Velocity stays close to but below the constraint limit
- Lagrange multiplier stabilizes at a reasonable value
- Low constraint violation rate

### Training Outputs
- **Model checkpoints**: Saved in `models/` directory
- **Trajectories**: Saved in `trajectories/` directory
- **Plots**: Training progress and rollout analysis in `plots/` directory

## Key Differences from Original Notebook

### Removed Components
- Complex environment wrappers
- Extensive hyperparameter grids
- Debug analysis code

### Kept Components
- Essential imports and setup
- Weights & Biases logging integration
- Core training loop with PPO Lagrange v2
- Model saving and loading
- Evaluation rollouts
- Comprehensive plotting
- Summary statistics

## Monitoring Training

### Key Metrics to Watch
1. **Episode reward**: Should increase over time
2. **Episode cost**: Should approach but not exceed safety_bound
3. **Lambda (Lagrange multiplier)**: Should stabilize
4. **Velocity magnitude**: Should respect the constraint
5. **Constraint violations**: Should decrease over time

### Troubleshooting
- **High costs**: Increase safety_bound or decrease lagrangian_coef_rate
- **Low rewards**: Decrease safety_bound or increase reward_scaling
- **Unstable lambda**: Decrease lagrangian_coef_rate
- **No learning**: Check environment setup and training parameters

## Performance Expectations

With proper tuning, the agent should achieve:
- Forward locomotion with velocities close to the constraint limit
- Minimal constraint violations (< safety_bound)
- Efficient movement patterns
- Stable policy behavior

## Extension Ideas

1. **Multiple constraints**: Add orientation or position constraints
2. **Dynamic constraints**: Time-varying velocity limits
3. **Hierarchical constraints**: Different limits for different body parts
4. **Curriculum learning**: Gradually tighten constraints during training
5. **Multi-objective**: Balance multiple conflicting objectives 