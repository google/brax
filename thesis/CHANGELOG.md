## 2025-08-09

- Clarified PPO clipped surrogate wording in `chapters/4_theoretical_background_old.tex`:
  - Emphasized that the surrogate objective is clipped (not the ratio itself).
  - Rephrased bullets to state the surrogate uses the upper/lower clipped ratio when $A_t^{\text{modified}}$ is positive/negative.
  - Updated importance sampling ratio description to use log-prob formulation and noted the tanh-squashed Normal policy includes the Jacobian term.

- Clarified loss weighting in `chapters/4_theoretical_background_old.tex`:
  - Documented fixed weights for value and cost (0.25 each) and that only the entropy coefficient $\beta$ (\texttt{entropy\_cost}) is tunable. Simplified wording to avoid redundancy.

- Deduplicated Lagrange update description in `chapters/4_theoretical_background_old.tex`:
  - Removed the earlier redundant subsection; kept one authoritative, labeled subsection `sec:lagrange_updates`.

- Added implementation note under Constraint Violation Measurement:
  - Documented that we use the un-discounted batch mean per-step cost for the dual update and interpret the bound as a per-step average target.

- Added implementation note under Adaptive Learning Rate Considerations:
  - Stated that we use a fixed dual learning rate (lagrangian_coef_rate) in experiments; other schedules are alternatives. Added a numerical guard (|d_i|+ε) to the adaptive formula.

- Deduplicated penalty strength explanation:
  - Removed separate “Adaptive Penalty Strength” subsection; escalation/relaxation is explained under the main update rule.

- Updated Automatic Differentiation section:
  - Clarified that we use a single reverse-mode gradient over the combined loss (via value_and_grad) and removed higher-order derivative examples.


