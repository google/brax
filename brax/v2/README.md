<img src="https://github.com/google/brax/raw/main/docs/img/brax_logo.gif" width="336" height="80" alt="BRAX"/>

Welcome to a preview of **Brax v2**, a massive update full of new capabilities!

# Note on the `v2` module

Do not depend on the `v2` module in your own stable releases! This is a preview:
the `v2` module is temporary. A future version bump of Brax will replace its API
with the new one found in `v2` and the `v2` preview namespace will disappear.
Users who want to stick with the old Brax should pin to v0.0.16.

# What's new?

Besides Brax remaining differentiable and blazingly fast, we have major upgrades
that are really exciting. For a summary of changes see the
[v0.1.0](https://github.com/google/brax/releases/tag/v0.1.0) release notes, or
take a look at a more in-depth review below:

*   New physics backend that works in **generalized coordinates**:
    *   Simulator fidelity vastly improves in generalized coordinates compared
        to PBD and Spring backends (used in Brax v0.0.16). We demonstrate
        comparable simulator fidelity to Mujoco through backend tests. You can
        find the new backend implementation in
        [generalized](https://github.com/google/brax/blob/main/brax/v2/generalized).
    *   PBD and
        [Spring](https://github.com/google/brax/blob/main/brax/v2/spring)
        backends will continue to be supported as they were in v0.0.16.
*   Direct support for Mujoco XML and URDF, replacing the old custom config
    format:
    *   See
        [mjcf.py](https://github.com/google/brax/blob/main/brax/v2/io/mjcf.py)
        for more details.
*   Generalized, PBD, and Spring backends support **non-spherical Inertias**.
*   A new System object that is natively traceable, making domain randomization
    and SysID first class citizens in Brax v2.
*   An Env API that better supports new/custom physics backends.
*   An open sourced visualizer backend:
    *   You can now run the visualizer locally for debugging, see
        [visualizer](https://github.com/google/brax/blob/main/brax/v2/visualizer).

## Quickstart:

[Brax v2 Training](https://colab.research.google.com/github/google/brax/blob/main/notebooks/Brax_v2_Training_Preview.ipynb)
introduces the Brax v2 API, and shows how to train a policy with the
generalized backend.
