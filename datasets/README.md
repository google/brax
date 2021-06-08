In this directory, we provide results from hyperparameter sweeps for SAC and PPO as zipped json files.  Within, hyperparameters are sorted by environment, and then by performance.\
\
Hyperparameter ranges for PPO:\
\
total_env_steps: 10_000_000 , 500_000_000 (split between files with these names)\
eval_frequency: 20\
reward_scaling: 1, 5\
episode_length: 1000\
normalize_observations: True\
action_repeat: 1\
entropy_cost: 1e-3\
learning_rate: 3e-4\
discounting: 0.99, 0.997\
num_envs: 2048\
unroll_length: 1, 5, 20\
batch_size: 512,1024\
num_minibatches: 4, 8, 16, 32\
num_update_epochs: 2, 4, 8\
\
\
Hyperparameter ranges for SAC:\
\
env: 'halfcheetah'\
total_env_steps: 1048576 * 5\
eval_frequency: 131012\
reward_scaling: 5, 10, 30\
episode_length: 1000\
normalize_observations: True\
action_repeat: 1\
learning_rate: 3e-4, 6e-4\
discounting: 0.95, 0.99, 0.997\
num_envs: 64, 128, 256\
min_replay_size: 8192\
max_replay_size: 1048576\
batch_size: 128, 256, 512\
grad_updates_per_step: 0.125 / 2, 0.125, 0.125 * 2
