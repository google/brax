from brax.envs import Env, State, Wrapper
import jax
import jax.numpy as jnp

class ViewerWrapper(Wrapper):
    """A wrapper that provides rendering functionality for a Brax viewer."""

    def __init__(self, env: Env, viewer):
        """Initializes the ViewerWrapper.

        Args:
            env: The environment to wrap.
            viewer: An instance of a viewer (e.g., WebViewerBatched) that has a
                `send_frame` method and a `rendering_enabled` property.
        """
        super().__init__(env)
        self.viewer = viewer

    @property
    def should_render(self) -> jax.Array:
        """Returns a JAX array indicating whether rendering should occur."""
        return jnp.array(True, dtype=jnp.bool_)

    def render_fn(self, state: State):
        """The function to be called for rendering a state.

        This function is designed to be used with `jax.experimental.io_callback`.
        It sends a single, unbatched state to the viewer.
        """
        if not self.viewer.rendering_enabled:
            return

        # If the state is batched, iterate and send each frame.
        if state.pipeline_state.q.ndim > 1:
            num_envs = state.pipeline_state.q.shape[0]
            for i in range(num_envs):
                single_state = jax.tree_util.tree_map(lambda x: x[i], state)
                self.viewer.send_frame(single_state)
        else:
            # If the state is not batched, send it directly.
            self.viewer.send_frame(state)
