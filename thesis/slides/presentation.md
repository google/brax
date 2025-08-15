---
layout: Title Slide
title: Safe-Brax Thesis
subtitle: Safety-Constrained Reinforcement Learning with Differentiable Physics
---

---
layout: Section Header
title: Motivation
---

- Safety is critical for deploying RL in real-world systems
- Differentiable physics enables efficient learning and constraints
- Goal: unify safety constraints with high-performance control

---
layout: Title and Content
title: Problem Statement
---

We study reinforcement learning under explicit safety constraints in the Brax
simulator. The objective is to maximize cumulative reward while satisfying
state and action constraints throughout training and deployment.

---
layout: Two Content
title: Contributions
left:
  - A velocity-constrained Ant environment and benchmarks
  - Safe policy optimization with constraint satisfaction
right:
  - Reproducible training/evaluation pipeline
  - Open-source experiments and results
---

---
layout: Picture with Caption
title: Safe Ant Environment
image: thesis/6873d12a507dfe00bbdae442/images/ant_safe.png
caption: Velocity-constrained Ant in Brax
---

---
layout: Table
title: Key Results (Example)
table:
  column_widths: [6, 4, 4]
  data:
    - ["Model", "Reward", "Constraint Violations"]
    - ["Baseline PPO", "3180", "45"]
    - ["Safe-PPO (ours)", "3040", "3"]
  header_style: dark_blue_white_text
  row_style: alternating_light_gray
  border_style: thin_gray
---

---
layout: Section Header
title: Thank You
---






