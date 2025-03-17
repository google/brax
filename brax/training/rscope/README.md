## Rscope: MJX + Brax Reinforcement Learning Debugger

![rscope_header](https://github.com/user-attachments/assets/225d0290-501d-4a2e-ace9-f2122786ffb6)

Understanding what your RL agent is learning can be challenging when relying solely on statistics and plots. `rscope` is a lightweight visualizer built on the Mujoco viewer that lets you interactively explore trajectories without adding any training time overhead. It enables you to scroll through and inspect trajectories collected from different parallel environments during training.

Key use cases include:
- Visualizing the randomization ranges during agent initialization.
- Determining appropriate scales for reward shaping.
- Investigating pathological behavior, such as local minima or issues leading to simulator NaNs.

**Features**
1. View trajectories from various policy checkpoints and parallel environments.
2. Visualize shaping reward terms.
3. Display pixel observations for vision-based training.
4. Utilize Mujoco Viewer functionalities like pan/zoom, toggle wireframe, and view light/camera locations.

### Usage
Between policy updates, Brax training algorithms evaluate the policy by unrolling parallel environments for a full episode and computing reward statistics. `rscope` then loads and visualizes these evaluation trajectories on the CPU, ensuring that accelerator hardware remains free for the training run.

To use `rscope`, configure your training algorithm to save evaluation trajectories under `/tmp/rscope/active_run`. This setup allows you to view both existing trajectories in the folder and any new ones as they are added.

**Training Script**

*Imports*
```python
from brax.training.rscope import rscope_utils
```

*Within the progress function*
```python
# Dumps trajectories from the evaluator run to /tmp/rscope/active_run for visualization
rscope_utils.dump_eval(metrics['eval/data'])
```

*Before training begins*
```python
# Clears /tmp/rscope/active_run and dumps initial data for rscope to load the model
rscope_utils.rscope_init(env.xml_path, env.model_assets)
```

*Training argument*
```python
...
# Save trajectories from environment 0:rscope_envs for visualization
rscope_envs=16
...
```

**CLI**
```bash
# Launch rscope
python -m brax.rscope
```

### Sharp Bits
- Currently supports only PPO-based training.
- Evaluating in **deterministic mode** (`deterministic_eval=True`) shows the policy's capabilities, whereas **stochastic mode** helps gauge training dynamics.
- Renders incorrectly for domain-randomized training because the loaded assets are from the nominal model definition.
- Plots only the first 14 keys in the metrics without filtering for shaping rewards.
- Visualizes only the first 14 pixel observations.
- Cannot capture curriculum progression during training, as curriculums depend on `state.info`, which is reset at the start of an evaluator run.
