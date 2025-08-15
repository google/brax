# Minimal 1-Seed Experiment Matrix and Plot Checklist

## Ant Velocity

### PPO
- **Plots:**
  - Reward vs. environment steps (train/eval if both available)
  - Cost or violation-rate vs. environment steps, with safety bound line
  - Final bar (single point now; later mean±CI) comparing PPO vs. PPO-Lagrange on reward and cost
  - *Optional (appendix):* KL divergence, clip fraction, policy entropy, value/cost-value loss

### PPO-Lagrange
- **Plots:**
  - Reward vs. environment steps
  - Cost or violation-rate vs. environment steps, with safety bound line
  - Lagrange multiplier λ vs. environment steps
  - Final bar (single point now; later mean±CI) vs. PPO
  - *Optional (appendix):* KL, clip fraction, entropy, value/cost-value loss

---

## Humanoid Height

### PPO
- **Plots:**
  - Reward vs. environment steps
  - Cost or violation-rate vs. environment steps, with safety bound line
  - Final bar vs. PPO-Lagrange
  - *Optional (appendix):* KL, clip fraction, entropy, value/cost-value loss

### PPO-Lagrange
- **Plots:**
  - Reward vs. environment steps
  - Cost or violation-rate vs. environment steps, with safety bound line
  - Lagrange multiplier λ vs. environment steps
  - Final bar vs. PPO
  - *Optional (appendix):* KL, clip fraction, entropy, value/cost-value loss

---

## Point-Goal

### PPO
- **Plots:**
  - Reward vs. environment steps
  - Violation-rate vs. environment steps, with safety bound line
  - Final bar vs. PPO-Lagrange
  - *Optional (appendix):* KL, clip fraction, entropy, value/cost-value loss

### PPO-Lagrange
- **Plots:**
  - Reward vs. environment steps
  - Violation-rate vs. environment steps, with safety bound line
  - Lagrange multiplier λ vs. environment steps
  - Final bar vs. PPO
  - *Optional (appendix):* KL, clip fraction, entropy, value/cost-value loss

---

## Optional Point-Goal Variants (if configured) — PPO-Lagrange

- **Variants:** Few large hazards / many small hazards
- **Plots:**
  - Reward vs. environment steps
  - Violation-rate vs. environment steps, with safety bound line
  - λ vs. environment steps
  - Final bar across the two variants

---

## Efficiency and Scalability (Single Device; 1 Seed per Sweep Point)

- **Throughput vs. number of parallel environments**
- For each environment (Ant, Humanoid, Point-Goal):
  - Steps-per-second vs. num_envs (line plot)
  - *Optional:* Wall-clock time to N steps (bar or line)
- **Steps-per-second over time** (one representative environment)
  - Plot SPS vs. wall-clock minutes across training
- **Memory/stability at high num_envs**
  - Max stable num_envs per environment (bar)
  - *Optional:* Memory usage vs. num_envs (line)

---

## Benchmark Validation

- **Reward and cost sanity checks** (short runs, 1 seed)
  - Histogram of per-step cost values (to show binary vs. continuous cost)
  - Reward components (if logged): stacked bar at episode end (*optional*)
  - Hazard occupancy or violation-rate distribution (histogram)

---

## Optional Cross-Framework (if included)

- **Matched task(s) in Safety-Gymnasium — PPO-Lagrange**
  - Reward vs. environment steps (SafeBrax vs. Safety-Gymnasium)
  - Violation-rate vs. environment steps (with bound)
  - Final bar (reward and cost) comparing frameworks
  - *Optional:* Scatter of final reward vs. cost (two points now; later with CI)

---

## Ablations (Run on One Representative Environment; 1 Seed Each)

- **Architecture:** Shared trunk vs. separate value/cost-value heads — PPO-Lagrange
  - Reward vs. environment steps (overlay)
  - Violation-rate vs. environment steps (overlay)
  - λ vs. environment steps (overlay)
  - Final bar (reward and cost)
- **Domain randomization:** Off vs. on — PPO-Lagrange
  - Same plots as above

---

## Plot Formatting Notes (Applies to All)

- X-axis: environment steps (not updates/epochs)
- Lines: raw and a lightly smoothed version (e.g., EMA) for single-seed runs
- Safety bound: horizontal dashed line on cost/violation-rate plots
- Later, when you add more seeds, add shaded CI bands and swap final bars to mean±CI

---

## Suggested Filenames per Run

- `{env}_{alg}_reward_curve.pdf`
- `{env}_{alg}_cost_curve.pdf`
- `{env}_{alg}_lambda_curve.pdf`
- `throughput_{env}_vs_num_envs.pdf`
- `sps_over_time_{env}.pdf`
- `final_bars_{env}_ppo_vs_lagrange.pdf`
- `validation_cost_hist_{env}.pdf`

---

## Suggested CSV Logs for Reproducibility

- `{env}_{alg}_train_metrics.csv`
- `{env}_{alg}_eval_metrics.csv`
- `throughput_sweep_{env}.csv`

---

## Minimal Run List to Produce All Plots (One Seed Each)

- **Ant:** PPO, PPO-Lagrange
- **Humanoid:** PPO, PPO-Lagrange
- **Point-Goal:** PPO, PPO-Lagrange
- **Throughput sweeps:** Ant, Humanoid, Point-Goal (vary num_envs)
- **Ablations (if included):** Choose one environment and run 2 variants

---

- You can reuse training logs to generate: reward/cost curves, λ curves, SPS-over-time, and final bars without extra runs.
- Short sanity runs produce validation histograms; throughput sweeps produce the efficiency/scaling plots.
- Add seeds later to the same setups to upgrade plots with CIs; no new plot types needed.
- **All plots should be reproducible from logged CSVs and a fixed plotting script.**