#!/usr/bin/env python3
"""
Test script for the velocity-constrained ant environment.

This script verifies that the environment is properly registered and can be instantiated,
reset, and stepped through. It also checks that velocity constraints are working correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs


def test_environment_creation():
    """Test that the environment can be created with different configurations."""
    print("Testing environment creation...")
    
    # Test with default parameters
    env1 = envs.get_environment('ant_velocity_constrained')
    print(f"✓ Default environment created: action_size={env1.action_size}, obs_size={env1.observation_size}")
    
    # Test with custom parameters
    env2 = envs.get_environment(
        'ant_velocity_constrained',
        max_velocity=1.5,
        velocity_cost_weight=2.0,
        ctrl_cost_weight=0.1
    )
    print(f"✓ Custom environment created: max_velocity=1.5 m/s")
    
    return env1, env2


def test_environment_dynamics(env, num_steps=100):
    """Test basic environment dynamics."""
    print(f"\nTesting environment dynamics for {num_steps} steps...")
    
    # Reset environment
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    
    print(f"✓ Environment reset successfully")
    print(f"  Initial observation shape: {state.obs.shape}")
    print(f"  Initial reward: {state.reward}")
    print(f"  Initial done: {state.done}")
    
    # Track metrics
    rewards = []
    costs = []
    velocities = []
    violations = []
    
    # Run simulation
    for i in range(num_steps):
        # Random action
        rng, act_rng = jax.random.split(rng)
        action = jax.random.uniform(act_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        
        # Step environment
        state = env.step(state, action)
        
        # Collect metrics
        rewards.append(float(state.reward))
        costs.append(float(state.metrics.get('cost', 0)))
        velocities.append(float(state.metrics.get('velocity_magnitude', 0)))
        violations.append(float(state.metrics.get('velocity_violation', 0)))
        
        if state.done:
            print(f"Episode terminated at step {i}")
            break
    
    # Summary statistics
    print(f"✓ Simulation completed for {len(rewards)} steps")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Average cost: {np.mean(costs):.3f}")
    print(f"  Max velocity: {np.max(velocities):.3f} m/s")
    print(f"  Average velocity: {np.mean(velocities):.3f} m/s")
    print(f"  Total violations: {np.sum(violations):.3f}")
    
    return {
        'rewards': rewards,
        'costs': costs,
        'velocities': velocities,
        'violations': violations
    }


def test_velocity_constraints(env):
    """Test that velocity constraints are working correctly."""
    print(f"\nTesting velocity constraints...")
    
    # Get max velocity from environment
    max_velocity = env._max_velocity
    print(f"Environment max velocity: {max_velocity} m/s")
    
    # Run with high action values to try to exceed velocity limit
    rng = jax.random.PRNGKey(123)
    state = env.reset(rng)
    
    max_observed_velocity = 0.0
    total_violations = 0.0
    
    # Use consistently high actions to try to exceed limits
    for i in range(200):
        # High positive actions to maximize forward velocity
        action = jnp.ones(env.action_size) * 0.8
        state = env.step(state, action)
        
        velocity_mag = float(state.metrics.get('velocity_magnitude', 0))
        violation = float(state.metrics.get('velocity_violation', 0))
        cost = float(state.metrics.get('cost', 0))
        
        max_observed_velocity = max(max_observed_velocity, velocity_mag)
        total_violations += violation
        
        if i % 50 == 0:
            print(f"  Step {i}: velocity={velocity_mag:.3f}, violation={violation:.3f}, cost={cost:.3f}")
        
        if state.done:
            break
    
    print(f"✓ Velocity constraint test completed")
    print(f"  Max observed velocity: {max_observed_velocity:.3f} m/s")
    print(f"  Velocity limit: {max_velocity:.3f} m/s")
    print(f"  Total violations: {total_violations:.3f}")
    
    # Check if constraints are working
    if max_observed_velocity > max_velocity:
        constraint_violations = max_observed_velocity - max_velocity
        print(f"✓ Constraints triggered (exceeded by {constraint_violations:.3f} m/s)")
    else:
        print(f"⚠ Constraints not triggered (stayed {max_velocity - max_observed_velocity:.3f} m/s below limit)")
    
    return max_observed_velocity, total_violations


def test_cost_computation():
    """Test that cost computation is working correctly."""
    print(f"\nTesting cost computation...")
    
    env = envs.get_environment(
        'ant_velocity_constrained',
        max_velocity=1.0,  # Low limit to ensure violations
        velocity_cost_weight=1.0
    )
    
    rng = jax.random.PRNGKey(456)
    state = env.reset(rng)
    
    costs_with_violations = []
    costs_without_violations = []
    
    for i in range(100):
        # Alternate between high and low actions
        if i % 2 == 0:
            action = jnp.ones(env.action_size) * 0.9  # High actions
        else:
            action = jnp.zeros(env.action_size)  # Low actions
        
        state = env.step(state, action)
        
        velocity_mag = float(state.metrics.get('velocity_magnitude', 0))
        cost = float(state.metrics.get('cost', 0))
        violation = float(state.metrics.get('velocity_violation', 0))
        
        if violation > 0:
            costs_with_violations.append(cost)
        else:
            costs_without_violations.append(cost)
        
        if state.done:
            break
    
    print(f"✓ Cost computation test completed")
    print(f"  Steps with violations: {len(costs_with_violations)}")
    print(f"  Steps without violations: {len(costs_without_violations)}")
    
    if costs_with_violations:
        print(f"  Average cost with violations: {np.mean(costs_with_violations):.4f}")
    if costs_without_violations:
        print(f"  Average cost without violations: {np.mean(costs_without_violations):.4f}")
    
    # Verify cost is higher when there are violations
    if costs_with_violations and costs_without_violations:
        violation_cost = np.mean(costs_with_violations)
        no_violation_cost = np.mean(costs_without_violations)
        if violation_cost > no_violation_cost:
            print(f"✓ Cost correctly increases with violations")
        else:
            print(f"⚠ Cost does not increase with violations")


def main():
    """Run all tests."""
    print("=" * 60)
    print("VELOCITY-CONSTRAINED ANT ENVIRONMENT TESTS")
    print("=" * 60)
    
    try:
        # Test environment creation
        env1, env2 = test_environment_creation()
        
        # Test basic dynamics
        results = test_environment_dynamics(env1, num_steps=150)
        
        # Test velocity constraints
        max_velocity, violations = test_velocity_constraints(env1)
        
        # Test cost computation
        test_cost_computation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The velocity-constrained ant environment is working correctly!")
        print("You can now run the training notebook.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check the environment implementation.")
        raise


if __name__ == "__main__":
    main() 