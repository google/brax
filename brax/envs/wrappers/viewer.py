import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper, State, Env
from typing import Optional

from braxviewer.WebViewer import WebViewer


class ViewerWrapper(Wrapper):
    """An environment wrapper that sends state to a WebViewer after each step."""

    def __init__(self, env: Env, viewer: Optional[WebViewer] = None):
        """Initializes the ViewerWrapper.

        Args:
            env: A Brax environment instance to wrap.
            viewer: An optional WebViewer instance for visualizing state.
        """
        super().__init__(env)
        self.viewer = viewer

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment and sends the initial state to the viewer.

        Args:
            rng: A JAX random number generator.

        Returns:
            The initial environment state.
        """
        state = self.env.reset(rng)

        if self.viewer is not None:
            jax.debug.callback(self.viewer.send_frame, state)

        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Performs one environment step and conditionally sends the state to the viewer.

        Args:
            state: The current environment state.
            action: The action to apply.

        Returns:
            The next environment state.
        """
        next_state = self.env.step(state, action)

        if self.viewer is not None:
            def _send_frame(s):
                jax.debug.callback(self.viewer.send_frame, s)

            def _do_nothing(s):
                pass

            jax.lax.cond(
                self.viewer.rendering_enabled,
                _send_frame,
                _do_nothing,
                operand=state  # Matches behavior in acting.py
            )

        return next_state
