# Quantum-Assisted Multi-Robot Path Coordination

This repository is now focused on a **paper-oriented A* + QAOA architecture**:

- **A*** generates diverse candidate paths per robot.
- **QAOA** solves the multi-robot path-selection problem as a QUBO.

Hybrid/APF/Q-learning components were removed from the active pipeline to keep the codebase aligned with the current research scope.

## Scope

Given N robots with K candidates each, the objective is:

1. Minimize total path cost.
2. Penalize inter-robot conflicts.
3. Enforce one-path-per-robot constraints.

The combinatorial space grows as K^N, so QAOA is used as a quantum optimization approach for coordination.

## Core Files

```text
src/
    a_star.py                # A* and QIA* utilities (A* used in active pipeline)
    grid.py                  # Grid and environment generation
    multi_robot.py           # Candidate generation + conflict matrix + baselines
    qaoa_optimizer.py        # QUBO build + QAOA solvers
    run_qaoa_experiments.py  # Main paper experiments (A* + QAOA)
    main.py                  # Entrypoint, forwards to run_qaoa_experiments

docs/
    Final_Report.md

report.tex                 # Paper draft
```

## Run

```bash
cd src
python main.py
```

Outputs are generated under:

```text
experiments/results/12_qaoa_honest/
```

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- numpy
- matplotlib
- scipy
- qiskit
- qiskit-aer
- qiskit-optimization
- qiskit-algorithms
