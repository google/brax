import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from brax.envs.base import Wrapper, State, Env
from brax.envs.wrappers.training import VmapWrapper
from typing import Optional, Callable

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
            # The check for rendering enabled is now inside the viewer's send_frame
            io_callback(self.viewer.send_frame, None, state)

        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Performs one environment step and sends the state to the viewer.

        Args:
            state: The current environment state.
            action: The action to apply.

        Returns:
            The next environment state.
        """
        next_state = self.env.step(state, action)

        if self.viewer is not None:
            # The check for rendering enabled is now inside the viewer's send_frame
            io_callback(self.viewer.send_frame, None, next_state)

        return next_state


class RenderableVmapWrapper(VmapWrapper):
    """A VmapWrapper that supports conditional rendering with optimal performance.
    
    This wrapper implements the "Pass-the-Flag" pattern to avoid performance
    degradation when using conditional callbacks in vmapped JAX functions.
    """

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env, batch_size)

    def step_with_render(
        self, 
        state: State, 
        action: jax.Array, 
        should_render: bool, 
        render_fn: Optional[Callable[[State], None]]
    ) -> State:
        """Performs a batched environment step with conditional rendering.
        
        Args:
            state: The current batched environment state.
            action: The batched actions to apply.
            should_render: A scalar boolean flag controlling rendering.
            render_fn: Optional callback function for rendering.
            
        Returns:
            The next batched environment state.
        """
        from jax import lax
        
        # First, perform the environment step
        next_state = self.env.step(state, action)
        
        # Then, conditionally render using lax.cond with scalar flag
        if render_fn is not None:
            def _render_branch(batched_state):
                # True branch: render all environments
                def send_all_frames(state_batch):
                    num_envs = state_batch.pipeline_state.q.shape[0]
                    for i in range(num_envs):
                        # Extract single environment state
                        single_state = jax.tree_util.tree_map(lambda x: x[i], state_batch)
                        render_fn(single_state)
                
                io_callback(send_all_frames, None, batched_state)
                return 0  # dummy return value
            
            def _no_op_branch(_):
                # False branch: do nothing
                return 0  # dummy return value
            
            # Use lax.cond with scalar should_render flag
            # This preserves true conditional execution in vmap contexts
            _ = lax.cond(should_render, _render_branch, _no_op_branch, next_state)
        
        return next_state
