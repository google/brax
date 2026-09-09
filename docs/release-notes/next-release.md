# Brax Release Notes

* Removed Madrona-specific flags from the brax PPO.
* Added more configurable vision network options: CNN kernel/output initializers, padding, activation, global pooling, and max-pool settings are now exposed as arguments in `make_ppo_networks_vision`.
* Added `spatial_softmax` global pooling option to `VisionMLP`, h/t zakka.
