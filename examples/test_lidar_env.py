import jax
import jax.numpy as jp
# Ensure brax.envs is available, and then your specific environment class
from brax.envs.PointResettingGoalRandomHazardLidarSensorObs import PointResettingGoalRandomHazardLidarSensorObs

def main():
    print("Attempting to create environment by direct import and instantiation...")
    try:
        env = PointResettingGoalRandomHazardLidarSensorObs()
        print("Environment created successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Please ensure the file 'brax/envs/PointResettingGoalRandomHazardLidarSensorObs.py' exists and is in the Python path.")
        return

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    
    print(f"Initial observation shape: {state.obs.shape}")
    print(f"Initial observation: {state.obs}")

    # Determine lidar sizes from observation shape (assuming 16 bins each)
    # Obs: accel(3), velo(3), gyro(3), mag(3), goal_lidar(16), hazard_lidar(16), goal_comp(2)
    # Total = 3+3+3+3+16+16+2 = 46
    expected_obs_size = 3 + 3 + 3 + 3 + 16 + 16 + 2 
    if state.obs.shape[0] != expected_obs_size:
        print(f"Warning: Observation size mismatch. Expected {expected_obs_size}, got {state.obs.shape[0]}")
        print("Please check sensor configuration and lidar additions.")

    print("\nTaking a few steps with random actions...")
    for i in range(5):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)
        print(f"\nStep {i+1}:")
        print(f"  Observation shape: {state.obs.shape}")
        print(f"  Observation: {state.obs}")
        print(f"  Reward: {state.reward}")
        print(f"  Done: {state.done}")
        print(f"  Cost: {state.info.get('cost', 'N/A')}")
        
        if state.obs.shape[0] == expected_obs_size:
            print(f"  Goal Lidar (first 8 bins): {state.obs[12:20]}")
            print(f"  Hazard Lidar (first 8 bins): {state.obs[28:36]}")
            print(f"  Goal Compass: {state.obs[44:45]}") # Corrected slice for 2 components (44 and 45)
        else:
            print("  Cannot reliably parse lidar from observation due to size mismatch.")

        if state.done:
            print("Episode finished.")
            break
            
    print("\nTest script finished.")

if __name__ == '__main__':
    main() 