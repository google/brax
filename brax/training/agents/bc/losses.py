import jax
import jax.numpy as jp


# Vanilla L2 with postprocessing
def bc_loss(params, normalizer_params, data, make_policy):
  policy = make_policy((normalizer_params, params))
  _, action_extras = policy(data['observations'], key_sample=None)
  actor_loss = (
      (
          (
              jp.tanh(action_extras['loc'])
              - jp.tanh(data['teacher_action_extras']['loc'])
          )
          ** 2
      )
      .sum(-1)
      .mean()
  )
  actor_loss = actor_loss.mean()
  return actor_loss, {'actor_loss': actor_loss, 'mse_loss': actor_loss}
