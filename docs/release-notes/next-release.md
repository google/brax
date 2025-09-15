# Brax Release Notes

* Support custom activation functions in checkpointing.
* Add value function coefficient to PPO.
* Pass through kernel initializers to SAC/PPO networks, and allow checkpointing of such parameters.
* Allow episode metrics during eval to be normalized by the episode length, as long as the metric name ends with "per_step".
