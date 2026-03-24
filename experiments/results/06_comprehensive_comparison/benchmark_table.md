# Benchmark Results

| Density | Planner | Success | Cost | Steps | Time (ms) |
|---------|---------|---------|------|-------|-----------|
| 10% | A* | Yes | 28.0 | 22 | 1.1 |
| 10% | APF | Yes | 28.9 | 22 | 2.1 |
| 10% | Classical QL | Yes | 40.0 | 41 | 677.4 |
| 10% | Quantum QL | Yes | 42.0 | 43 | 2415.3 |
| 10% | Hybrid | Yes | 30.6 | 25 | 2476.0 |
| 15% | A* | Yes | 31.0 | 27 | 1.8 |
| 15% | APF | No | 0.0 | 18 | 1.9 |
| 15% | Classical QL | Yes | 64.0 | 65 | 682.4 |
| 15% | Quantum QL | No | 0.0 | 501 | 2736.7 |
| 15% | Hybrid | Yes | 79.0 | 75 | 2801.1 |
| 20% | A* | Yes | 31.6 | 28 | 2.8 |
| 20% | APF | No | 0.0 | 39 | 8.2 |
| 20% | Classical QL | Yes | 40.0 | 41 | 603.6 |
| 20% | Quantum QL | Yes | 50.0 | 51 | 2845.8 |
| 20% | Hybrid | Yes | 62.7 | 60 | 2769.6 |
| 25% | A* | Yes | 32.1 | 29 | 1.2 |
| 25% | APF | No | 0.0 | 34 | 4.5 |
| 25% | Classical QL | Yes | 38.0 | 39 | 569.4 |
| 25% | Quantum QL | No | 0.0 | 501 | 3006.8 |
| 25% | Hybrid | Yes | 50.7 | 48 | 2960.4 |
| 30% | A* | Yes | 32.7 | 30 | 1.6 |
| 30% | APF | No | 0.0 | 35 | 5.3 |
| 30% | Classical QL | Yes | 38.0 | 39 | 542.5 |
| 30% | Quantum QL | No | 0.0 | 501 | 3277.9 |
| 30% | Hybrid | No | 0.0 | 479 | 3342.9 |

## Key Findings

- **A*** is optimal on static grids but cannot adapt to moving obstacles
- **APF** reacts locally but gets stuck in traps (U-shaped walls)
- **Classical QL** learns but Q-values can diverge to +/- infinity
- **Quantum QL** stays bounded [0, pi] — never diverges
- **Hybrid** combines all three layers: global (A*) + reactive (APF) + escape (Quantum QL)