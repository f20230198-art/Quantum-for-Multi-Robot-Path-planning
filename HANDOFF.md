# Handoff Notes for Future Claude Code Sessions

This file is for any Claude Code session picking up work on this project.
Read this first before doing anything else.

---

## Project context

- **Owner:** Srivathsa H Honyal (shhonyal@gmail.com), BITS Pilani Dubai.
- **Teammate:** Kalyani Baiju — did the most recent push (`simulator` commit, c098ef0, 2026-04-18 on `main`). She is **done** with the project; Srivathsa is now driving solo.
- **Repo:** https://github.com/f20230198-art/Quantum-for-Multi-Robot-Path-planning.git
- **Active branch:** `main`.
- **Local working dir:** `c:\Users\shhon\OneDrive - BITS PILANI\Documents\Quantam\Quantum-for-Multi-Robot-Path-planning`.

## Problem statement (assigned)

Dynamic Multi-Robot Path Planning via QUBO-Based Rolling Horizon Optimization (Quantum-Ready MAPF).
Required constraints: vertex, edge, dynamic-obstacle, kinematic, goal. Reference papers are `1.pdf`, `2.pdf`, `3.pdf` in repo root.

## What is actually built (as of 2026-04-28)

Active pipeline = A\* candidate generation + QAOA selection over a QUBO. Everything else (APF, Q-learning, hybrid planner) was stripped per `README.md`.

Key files:
- `src/main.py` → forwards to `run_qaoa_experiments.py`
- `src/multi_robot.py` → candidate paths, conflict matrix, classical baselines
- `src/qaoa_optimizer.py` → QUBO build, numpy QAOA simulator, real Qiskit AerSimulator gate-level QAOA
- `src/simulator_paper_demo.py` → pygame interactive demo (replans on events)
- `experiments/results/12_qaoa_honest/` → latest results + report

Pipeline confirmed working on Srivathsa's machine 2026-04-28. Qiskit available. Numpy QAOA caps at ~16 qubits.

## Latest measured results (good numbers)

- QAOA optimality vs brute-force: **99.4% average** across 8x8 to 15x15 grids, 2–4 robots.
- Brute-force is faster than QAOA at all tested sizes (admitted in report).
- QAOA can't run beyond 4 robots with K=4 (>16 qubits).
- Qiskit gate-level QAOA works (Exp 2 ran it on 2/3/4 robot cases successfully).

## Known gaps vs problem statement (DO NOT FORGET)

Ranked by pain-vs-payoff. None are done yet.

1. **Edge-swap conflicts not detected.** `multi_robot.py:206-218` only counts same-cell-same-time. Two robots swapping positions in one tick is undetected. Problem statement requires it. ~1 hour fix.
2. **Demo has no coordination metrics overlay.** Pygame demo looks pretty but has no on-screen counter for conflicts avoided / replans triggered / goals reached. ~2 hours fix.
3. **QAOA stops at 16 qubits.** Drop K=4 → K=2 to fit 8 robots in 16 qubits. Or partition robots into groups. Half a day.
4. **No formal rolling-horizon loop.** Title says "Rolling Horizon Optimization" but there is no `plan(window=H) → execute H → replan` loop in main experiments. Pygame demo replans only on events. ~1 day fix.

## Demo / professor context

- Professor is described as "super chill" — does not read code, cares that work was done and explained convincingly.
- Demo target: today (2026-04-28). No time to implement the gaps above before demo.
- Strategy for today: present what works, frame the gaps as "future work / honest limitations," lean on the 99.4% optimality number and the live pygame demo.

## Positioning story (use this framing)

The 3 reference papers are all **single-robot** quantum-inspired methods (QER-LPD3QN, CAAQPSO, Fuzzy A\*+APF+Quantum-Q). Paper 1 explicitly lists *multi-robot coordination* as future work. Our project fills that exact gap by mapping multi-robot path selection to QUBO and solving with real QAOA circuits. We do not claim hardware speedup — we claim *formulation novelty* and *near-optimal solution quality* (99.4%) at demonstrated scale.

## Commands that work

```bash
cd "Quantum-for-Multi-Robot-Path-planning/src"
python main.py                    # full paper experiments, ~2 min
python simulator_paper_demo.py    # pygame interactive demo
```

## What NOT to do without asking

- Do not push to `main` without confirming with Srivathsa.
- Do not delete `1.pdf`, `2.pdf`, `3.pdf`, `report.tex`, or anything in `docs/`.
- Do not re-add APF/Q-learning/hybrid planner — the team deliberately removed them.
- Do not claim the project beats CBS/ECBS/MILP or scales to 1000 robots. It does not.

## When picking up: first three things to do

1. Read this file end-to-end.
2. `git -C Quantum-for-Multi-Robot-Path-planning log --oneline -10` to see what changed since this note.
3. Ask Srivathsa what the goal of *this* session is — demo prep, gap-closing, report-writing, etc.
