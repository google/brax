# Changelog

All notable changes to this project will be documented in this file.

## 2025-08-14

- Added PPO-Cost wrapper and training helper.
  - File: `brax/training/agents/ppo/ppo_cost.py`
  - Description: Introduces `RewardMinusCostWrapper` that sets reward to `reward - cost_weight * cost`, and `train_ppo_cost` which reuses standard PPO with this wrapper. This enables a quick baseline where safety costs are directly treated as negative reward.

## 2025-08-13

- Added `tools/collect_metrics.py` to run PPO or PPO-Lagrange and collect CSV logs:
  - `{env}_{alg}_train_metrics.csv`
  - `{env}_{alg}_eval_metrics.csv`
  - Captures reward, cost/violation-rate, lambda (for PPO-Lagrange), and steps-per-second.
  - Reason: Enable reproducible, scriptable data collection for plots defined in `notebooks/tests.md`.

- Added `tools/throughput_sweep.py` to sweep `num_envs` and log throughput to `throughput_sweep_{env}.csv`.
  - Reason: Support efficiency/scaling plots (SPS vs. num_envs; SPS over time via periodic metrics).

## Changelog

### 2025-08-12
- Added a new `Research Questions` section to `thesis/6873d12a507dfe00bbdae442/chapters/1_introduction.tex` before the Methodology section.
  - Rationale: Address missing research questions and clarify the thesis' guiding inquiries.
- Created a timestamped backup: `thesis/6873d12a507dfe00bbdae442/chapters/1_introduction_backup_2025-08-12_1.tex`.


### 2025-08-12
- Added `thesis/slides/main.py` to generate slides using Deckbuilder (python-pptx backend) with CLI for Markdown/JSON inputs.
  - Reason: Begin thesis slide workflow integrated with Deckbuilder as requested.
- Created starter markdown deck `thesis/slides/presentation.md` with common sections (Motivation, Problem, Contributions, Results).
  - Reason: Provide a ready-to-run example to validate the pipeline.


