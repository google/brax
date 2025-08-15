#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import jax
from jax import random as jrandom

from brax import envs
from brax.io import image as brax_image


def _try_backend(gl_backend: str) -> bool:
    os.environ["MUJOCO_GL"] = gl_backend
    try:
        # use a lightweight env for probing
        test_env = envs.create(env_name="inverted_pendulum", auto_reset=False)
        key = jrandom.PRNGKey(0)
        state = test_env.reset(key)
        _ = brax_image.render(test_env.sys, [state.pipeline_state], height=64, width=64, fmt="png")
        return True
    except Exception:
        return False


def _select_backend() -> str:
    # Respect pre-set MUJOCO_GL if it works
    preset = os.environ.get("MUJOCO_GL")
    if preset and _try_backend(preset):
        print(f"Using preset MUJOCO_GL={preset}", flush=True)
        return preset

    # Try a list of common headless-capable backends
    candidates = ["egl", "osmesa", "swiftshader", "angle", "glfw"]
    for cand in candidates:
        if _try_backend(cand):
            print(f"Selected MUJOCO_GL={cand}", flush=True)
            return cand

    # Fallback to whatever is set; rendering likely to fail
    fallback = os.environ.get("MUJOCO_GL", "")
    print(f"Warning: No working MUJOCO_GL backend found. Current='{fallback}'", flush=True)
    return fallback


def main(output_dir: str = "thesis/assets/env_screenshots", width: int = 640, height: int = 480):
    # Select a working rendering backend
    _select_backend()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Try to get all registered environment names
    try:
        # Access registry directly
        env_names = list(envs._envs.keys())  # type: ignore[attr-defined]
    except Exception:
        # Fallback: a minimal common set if registry is not accessible
        env_names = [
            "ant",
            "ant_velocity_constrained",
            "fast",
            "halfcheetah",
            "hopper",
            "humanoid",
            "humanoid_height_constrained",
            "humanoidstandup",
            "inverted_pendulum",
            "inverted_double_pendulum",
            "pusher",
            "reacher",
            "swimmer",
            "walker2d",
            "point_resetting_goal_random_hazard_sensor_obs",
            "point_resetting_goal_random_hazard_lidar_sensor_obs",
        ]

    key = jrandom.PRNGKey(0)

    succeeded, failed = [], []
    for name in env_names:
        try:
            # Create environment without batching, with auto_reset disabled to keep initial state
            env = envs.create(env_name=name, auto_reset=False)

            # Reset to get initial state
            key, subkey = jrandom.split(key)
            state = env.reset(subkey)

            # Render a single frame using the pipeline state
            img_bytes = brax_image.render(
                env.sys,
                [state.pipeline_state],
                height=height,
                width=width,
                camera=None,
                fmt="png",
            )

            out_file = out_path / f"{name}.png"
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            print(f"Saved: {out_file}", flush=True)
            succeeded.append(name)
        except Exception as e:
            print(f"Failed: {name} -> {e}", flush=True)
            failed.append((name, str(e)))

    print("\nSummary:", flush=True)
    print(f"  Succeeded ({len(succeeded)}): {succeeded}", flush=True)
    if failed:
        print(f"  Failed ({len(failed)}): {[n for n, _ in failed]}", flush=True)
        # Optionally exit non-zero if any failed
        # sys.exit(1)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "thesis/assets/env_screenshots"
    main(out_dir)


