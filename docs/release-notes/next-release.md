# Brax Release Notes

* Add training metrics to brax PPO, which allows users to avoid running evals during training while getting more frequent metric updates (à là RSL-RL). Set `num_evals=0` and `log_training_metrics=True`.
* Add checkpointing directly to brax PPO, rather than relying on the `policy_params_fn` callback.
* Fix bug in inverted pendulum (#574) where the position of the tip was being mis-calculated.
