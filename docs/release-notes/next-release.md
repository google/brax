# Brax Release Notes

* Add boolean `wrap_env` to all brax `train` functions, which optionally wraps the env for training, or uses the env as is.
* Fix bug in PPO train to return loaded checkpoint when `num_timesteps` is 0.
* Add `layer_norm` to `make_q_network` and set `layer_norm` to `True` in `make_sace_networks` Q Network.
