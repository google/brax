from typing import NamedTuple, Union, List, Dict
import pickle
from pathlib import Path
from brax.training.acme.types import NestedArray

MAX_VIEWPORTS = 12

class Rollout(NamedTuple):
    qpos: NestedArray
    qvel: NestedArray
    mocap_pos: NestedArray
    mocap_quat: NestedArray
    obs: NestedArray
    reward: NestedArray
    time: NestedArray
    metrics: NestedArray

# Global rollout state.
rollouts: List[Rollout] = []
num_evals = 0
num_envs = 0
env_ctrl_dt = 0.0
change_rollout = False

def append_unroll(fpath: Union[str, Path]):
    """Load an unroll file and append it to the list of rollouts."""
    global num_evals, num_envs, env_ctrl_dt, change_rollout
    with open(fpath, "rb") as f:
        transitions = pickle.load(f)
    raw_rollout = transitions.extras['state_extras']['rscope']
    assert raw_rollout['qpos'].shape[1] == raw_rollout['qvel'].shape[1], (
        f"qpos and qvel shapes don't match: {raw_rollout['qpos'].shape} vs {raw_rollout['qvel'].shape}"
    )
    num_envs = raw_rollout['qpos'].shape[1]
    rollouts.append(
        Rollout(
            qpos=raw_rollout['qpos'],
            qvel=raw_rollout['qvel'],
            mocap_pos=raw_rollout['mocap_pos'],
            mocap_quat=raw_rollout['mocap_quat'],
            obs=transitions.observation,
            reward=transitions.reward,
            time=raw_rollout['time'],
            metrics=raw_rollout['metrics']
        )
    )
    num_evals += 1
    env_ctrl_dt = raw_rollout['time'][1, 0] - raw_rollout['time'][0, 0]
    if len(rollouts) == 1:
        change_rollout = True

def find_unrolls(base_path: Union[str, Path]) -> List[str]:
    """Return a list of filenames in base_path that end with .mj_unroll."""
    base = Path(base_path)
    return [f.name for f in base.iterdir() if f.name.endswith(".mj_unroll")]

def dict_obs_pixels_env_select(obs: Dict, i_env: int) -> Dict:
    """
    Select the first MAX_VIEWPORTS keys from the observation dictionary
    that start with 'pixels/' (excluding ones with 'latent') and extract column i_env.
    """
    obs_pixels = {}
    num_shown = 0
    for key in obs.keys():
        if num_shown >= MAX_VIEWPORTS:
            break
        if key.startswith('pixels/') and 'latent' not in key:
            obs_pixels[key] = obs[key][:, i_env]
            num_shown += 1
    return obs_pixels

def dict_obs_t_select(obs: Dict, t: int) -> Dict:
    """
    Select the first MAX_VIEWPORTS keys from the observation dictionary
    that start with 'pixels/' (excluding ones with 'latent') and extract index t.
    """
    obs_t = {}
    num_shown = 0
    for key in obs.keys():
        if num_shown >= MAX_VIEWPORTS:
            break
        if key.startswith('pixels/') and 'latent' not in key:
            obs_t[key] = obs[key][t]
            num_shown += 1
    return obs_t

def metrics_env_select(metrics: Dict, i_env: int) -> Dict:
    """Select column i_env from each metric."""
    return {key: metrics[key][:, i_env] for key in metrics.keys()}
