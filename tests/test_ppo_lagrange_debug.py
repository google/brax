import jax
import jax.numpy as jnp
import pytest

from brax import envs
from brax.training import acting
from brax.training.agents.ppo_lagrange import losses as ppo_lagrange_losses

# NOTE: Update this import if you are using a custom environment
EnvClass = envs.PointResettingGoalRandomHazardSensorObs


@pytest.mark.parametrize("steps", [32])
def test_environment_cost_signal(steps):
    """Environment should occasionally emit a non-zero cost."""
    env = EnvClass()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    non_zero_cost_seen = False
    for i in range(steps):
        key, key_action = jax.random.split(key)
        action = jax.random.uniform(key_action, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)
        cost_val = float(state.metrics.get("cost", 0.0)) 
        if cost_val != 0.0:
            non_zero_cost_seen = True
            break

    assert non_zero_cost_seen, "Cost signal never became non-zero; check environment configuration."


def test_actor_step_contains_cost():
    """acting.actor_step must forward the 'cost' metric into state_extras."""
    env = EnvClass()
    key = jax.random.PRNGKey(1)
    state = env.reset(key)

    def dummy_policy(obs, key):
        action = jnp.zeros((env.action_size,))
        # Minimal extras expected downstream – include raw_action & log_prob placeholders
        extras = {
            "raw_action": action,
            "log_prob": jnp.array(0.0, dtype=jnp.float32),
        }
        return action, extras

    next_state, transition = acting.actor_step(env, state, dummy_policy, key, extra_fields=("cost",))
    # Ensure the key is present in the collected transition
    assert "cost" in transition.extras["state_extras"], "'cost' missing in state_extras after actor_step"


def test_update_lagrange_multiplier_basic():
    """update_lagrange_multiplier should increase lambda when cost exceeds limit."""
    lambda_value = jnp.array(0.0)
    cost_return = jnp.array(5.0)
    cost_limit = 1.0
    new_lambda = ppo_lagrange_losses.update_lagrange_multiplier(
        lambda_value,
        cost_return,
        cost_limit,
        lambda_lr=0.1,
        lambda_max=10.0,
    )
    expected = jnp.array(0.4)  # 0 + 0.1 * (5 – 1)
    assert jnp.isclose(new_lambda, expected), f"Expected {expected}, got {new_lambda}" 