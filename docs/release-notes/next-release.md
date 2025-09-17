# Brax Release Notes

* Support custom activation functions in checkpointing.
* Add value function coefficient to PPO.
* Pass through kernel initializers to SAC/PPO networks, and allow checkpointing of such parameters.
* Allow episode metrics during eval to be normalized by the episode length, as long as the metric name ends with "per_step".
* Add adaptive learning rate to PPO. Desired KL is sensitive to network initialization weights and entropy cost and may require some tuning for your environment.
* Add loss metrics to the PPO training logger.
