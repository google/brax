# tests/test_video_circular.py
import os
from pathlib import Path

import jax.numpy as jnp

from brax.envs.SafePointGoal import SafePointGoal_8Hazards
from train_from_config import record_episode_video


# --- Dummy circular policy maker ---
def make_circular_policy(action_dim: int, thrust: float, yaw_rate: float, thrust_idx: int, yaw_idx: int):
    """
    Returns a make_inference_fn(params)->inference(obs, rng)->(action, info)
    that always outputs constant [thrust, yaw_rate] (others = 0).
    Works with your record_episode_video which JITs the policy.
    """
    base = jnp.zeros((action_dim,), dtype=jnp.float32)
    base = base.at[thrust_idx].set(thrust)
    base = base.at[yaw_idx].set(yaw_rate)

    def _make_inference_fn(_params):
        # Must be JAX-friendly (no Python counters/state), since it gets jitted.
        def _infer(obs, rng):
            return base, {}  # (action, extras)

        return _infer

    return _make_inference_fn


def test_record_video_circular(out_path):
    env = SafePointGoal_8Hazards()

    # Make the constant-action circular policy
    make_infer = make_circular_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        yaw_rate=0.3,  # steady left turn -> circle
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
    )

    record_episode_video(
        env=env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=2500,
        camera="fixedfar",
        width=640,
        height=480,
        fps=500,
        out_name=str(out_path.name),
        log_to_wandb=False,
        seed=0,
    )

    saved = os.path.join("videos", out_path.name)
    assert os.path.exists(saved), f"Expected video at {saved}"
    assert os.path.getsize(saved) > 0, "Video file is empty"


if __name__ == "__main__":
    file_path = Path(__file__).parent / "videos" / "circle.mp4"
    test_record_video_circular(file_path)
