# Dynamic Environment Benchmark

## Scenario

20x20 grid with a **U-shaped trap** and **8 moving obstacles**
placed directly on the optimal A\* path. Obstacles bounce back
and forth every timestep, blocking the planned route.

| Planner | Reaches Goal | Path Cost | Steps | Collisions | Why |
|---------|:-----------:|----------:|------:|-----------:|-----|
| **A*** | Yes | 16.2 | 16 | 1 | Static plan — blind to dynamics |
| **APF** | No | 0.0 | 31 | 0 | Stuck in U-trap |
| **Classical QL** | Yes | 18.0 | 19 | 3 | Static policy — ignores dynamics |
| **Quantum QL** | Yes | 26.0 | 27 | 6 | Bounded angles but static policy |
| **Hybrid** | Yes | 43.2 | 37 | 0 | 3-layer adaptive — 6 ms real-time |

## Why Hybrid Wins

| Challenge | Which layer handles it |
|-----------|----------------------|
| Find shortest route | **A\*** (global planner) |
| Dodge moving obstacles in real-time | **APF** (local reactive layer) |
| Escape U-shaped trap | **Quantum Q-Learning** (learned escape policy) |
| Replan after escape | **A\*** re-runs from new position |

No single algorithm solves all three problems. The hybrid three-layer architecture is the **only** planner that reaches the goal with zero collisions.