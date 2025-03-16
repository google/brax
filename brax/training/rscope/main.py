import time
from pathlib import Path

import mujoco
import mujoco.viewer as mujoco_viewer
import glfw

from config import BASE_PATH
from model_loader import load_model_and_data
import rollout
from event_handler import MjUnrollHandler
import viewer_utils as vu
from image_processing import process_img
from watchdog.observers import Observer
from state import ViewerState

# Create an instance of ViewerState to encapsulate state.
viewer_state = ViewerState()


# Load the Mujoco model and data.
mj_model, mj_data, meta = load_model_and_data()

# Load initial unroll files.
def load_initial_unrolls():
    unroll_files = rollout.find_unrolls(BASE_PATH)
    while not unroll_files:
        print(f"No unrolls found in {BASE_PATH}, waiting...")
        time.sleep(4)
        unroll_files = rollout.find_unrolls(BASE_PATH)
    unroll_files = sorted(unroll_files)
    for f in unroll_files:
        rollout.append_unroll(Path(BASE_PATH) / f)
    print(f"main.py Env dt: {rollout.env_ctrl_dt}")

load_initial_unrolls()

# Setup file system observer.
event_handler = MjUnrollHandler()
observer = Observer()
observer.schedule(event_handler, str(BASE_PATH), recursive=False)
try:
    observer.start()
except Exception as e:
    print(f"Error starting observer: {e}")

# Initialize figures using metrics keys from the first rollout.
metrics_keys = list(rollout.rollouts[0].metrics.keys())
vu.reset_figures(metrics_keys)

# Determine the initial replay length.
replay_len = rollout.rollouts[0].qpos.shape[0]

with mujoco_viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False, show_right_ui=False,
        key_callback=viewer_state.key_callback) as viewer:
    
    while viewer.is_running():
        step_start = time.time()
        
        # Trajectory selection: if a new rollout is requested.
        if viewer_state.change_rollout:
            vu.reset_figures(metrics_keys)
            viewer_state.change_rollout = False
            full_rollout = rollout.rollouts[viewer_state.cur_eval]
            if isinstance(full_rollout.obs, dict):
                obs = rollout.dict_obs_pixels_env_select(full_rollout.obs, viewer_state.cur_env)
            else:
                obs = full_rollout.obs[:, viewer_state.cur_env]
            cur_rollout = full_rollout._replace(
                qpos=full_rollout.qpos[:, viewer_state.cur_env],
                qvel=full_rollout.qvel[:, viewer_state.cur_env],
                mocap_pos=full_rollout.mocap_pos[:, viewer_state.cur_env],
                mocap_quat=full_rollout.mocap_quat[:, viewer_state.cur_env],
                obs=obs,
                reward=full_rollout.reward[:, viewer_state.cur_env],
                time=full_rollout.time[:, viewer_state.cur_env],
                metrics=rollout.metrics_env_select(full_rollout.metrics, viewer_state.cur_env)
            )
            replay_index = 0
            replay_len = cur_rollout.qpos.shape[0]
        
        with viewer.lock():
            # Overlay text.
            text_1 = "Eval\nEnv\nStep\nStatus"
            text_2 = (f"{viewer_state.cur_eval+1}/{rollout.num_evals}\n"
                     f"{viewer_state.cur_env+1}/{rollout.num_envs}\n"
                     f"{replay_index}")
            text_2 += "\nPause" if viewer_state.pause else "\nPlay"
            overlays = [
                (mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT, text_1, text_2)
            ]
            if viewer_state.show_help:
                menu_text_1, menu_text_2 = vu.get_menu_text()
                overlays.append(
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, menu_text_1, menu_text_2)
                )
            viewer.overlay_text(overlays)
            
            # Render figures (metrics).
            if viewer_state.show_metrics:
                if not viewer_state.pause:
                    cur_metrics = {key: metrics[replay_index] for key, metrics in cur_rollout.metrics.items()}
                    for key in cur_metrics:
                        vu.add_data_to_fig(key, cur_metrics[key])
                viewports = vu.get_viewports(len(cur_rollout.metrics), viewer.viewport)
                viewport_figures = list(zip(viewports, list(vu.figures.values())))
                viewer.set_figures(viewport_figures)
            else:
                viewer.clear_figures()
            
            # Render pixel observations if available.
            from collections.abc import Mapping
            if isinstance(cur_rollout.obs, Mapping):
                if any(key.startswith('pixels/') for key in cur_rollout.obs.keys()):
                    if viewer_state.show_pixel_obs:
                        cur_obs = rollout.dict_obs_t_select(cur_rollout.obs, replay_index)
                        viewports = vu.get_viewports(len(cur_obs), viewer.viewport)
                        processed_obs = {key: process_img(cur_obs[key]) for key in cur_obs.keys()}
                        viewer.set_images(list(zip(viewports, list(processed_obs.values()))))
                    else:
                        viewer.clear_images()
        
        # Advance simulation: update the state.
        def advance_rollout(mj_model, mj_data, idx):
            mj_data.qpos, mj_data.qvel = cur_rollout.qpos[idx], cur_rollout.qvel[idx]
            if cur_rollout.mocap_pos.size:
                mj_data.mocap_pos, mj_data.mocap_quat = cur_rollout.mocap_pos[idx], cur_rollout.mocap_quat[idx]
            mj_data.time = cur_rollout.time[idx]
            mujoco.mj_forward(mj_model, mj_data)
        
        advance_rollout(mj_model, mj_data, replay_index)
        if not viewer_state.pause:
            replay_index = (replay_index + 1) % replay_len
            viewer.sync()
        
        time_until_next_step = float(rollout.env_ctrl_dt) - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

observer.stop()
observer.join()
