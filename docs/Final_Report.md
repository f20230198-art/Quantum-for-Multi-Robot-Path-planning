# Final Report
## QAOA-Based Multi-Robot Path Coordination via QUBO Formulation

---

## 1. Abstract

This project presents a hybrid quantum-classical framework for multi-robot path planning, with the primary contribution focused on multi-robot coordination. Classical A* is used to generate candidate paths per robot, and Quantum Approximate Optimization Algorithm (QAOA) is used to solve the combinatorial path-assignment problem through a QUBO formulation.

The framework also includes an auxiliary quantum-inspired Q-Learning module that represents action values as bounded qubit rotation angles (theta in [0, pi]) to stabilize learning dynamics. Across benchmark scenarios, QAOA achieves near-optimal assignment quality (about 99.3% of brute-force on average in paper experiments) while remaining conflict-aware. We report results honestly: at current tested scales, brute-force is often faster, and the practical scaling advantage of QAOA remains a larger-scale and hardware-dependent objective.

---

## 2. Problem Definition

Given N robots and K candidate paths per robot, the coordination problem is to choose one path per robot such that total path cost is minimized while inter-robot collisions are avoided.

The search space has size K^N, which grows exponentially with team size.

### Why this is hard

| Method | Strength | Limitation |
|--------|----------|------------|
| A* (single robot) | Optimal path on static map | Does not solve joint coordination across many robots |
| Greedy path picking | Very fast | Ignores global interaction and can create collisions |
| Brute-force | Global optimum | Exponential complexity O(K^N) |

---

## 3. Methodology

### 3.1 Classical-Quantum Pipeline

1. Generate K diverse candidate paths for each robot using A* and penalty-zone perturbations.
2. Build conflict matrix and per-path costs.
3. Formulate assignment as QUBO with one-hot constraints.
4. Convert QUBO to Ising form and solve with QAOA (gate-level circuits).

### 3.2 QUBO Formulation

Binary variable x[i,k] = 1 means robot i selects candidate path k.

Objective:

```
min  sum(c[i,k] * x[i,k])
   + lambda_conflict * sum(conflict[(i,k),(j,l)] * x[i,k] * x[j,l])
   + lambda_onehot * sum_i (sum_k x[i,k] - 1)^2
```

This jointly encodes path quality, collision penalties, and one-path-per-robot feasibility.

### 3.3 QAOA Circuit Implementation

- Initial state: Hadamard layer to create uniform superposition.
- Cost unitary: RZZ and RZ rotations from Ising terms.
- Mixer unitary: RX rotations.
- Parameter optimization: COBYLA with multiple restarts.
- Execution backends: NumPy statevector simulator and Qiskit AerSimulator.

The NumPy backend is used as the default simulator because it directly implements the QAOA statevector with plain array operations, which keeps the prototype lightweight, dependency-friendly, and easy to run on ordinary laptops. This is appropriate for the small qubit counts tested in this project; Qiskit remains available as an alternative backend when installed and when we want a gate-level comparison.

### 3.4 Auxiliary Quantum-Inspired Q-Learning

Each state-action pair is represented by rotation angle theta in [0, pi], with action probability:

```
P(a|s) = sin^2(theta[s,a] / 2)
```

The update uses TD error plus boundary-aware damping so values remain bounded and avoid classical Q-value blow-up.

---

## 4. Experimental Results (Latest)

Environment family in paper experiments: 8x8 to 15x15 grids, 2 to 4 robots, K=4 candidates unless stated.

### 4.1 Solution Quality: QAOA vs Classical Selectors

| Scenario | Greedy | Brute-Force (opt.) | QAOA | BF/QAOA |
|----------|--------|--------------------|------|---------|
| 8x8, 2 robots | 22.1 (0c) | 22.1 (0c) | 22.1 (0c) | 1.000 |
| 8x8, 3 robots | 33.2 (0c) | 33.2 (0c) | 33.8 (0c) | 0.983 |
| 10x10, 3 robots | 63.5 (2c) | 44.6 (0c) | 45.2 (0c) | 0.987 |
| 10x10, 4 robots | 87.9 (3c) | 59.7 (0c) | 60.3 (0c) | 0.990 |
| 12x12, 3 robots | 73.1 (2c) | 54.3 (0c) | 54.3 (0c) | 1.000 |
| 15x15, 3 robots | 66.4 (0c) | 66.4 (0c) | 66.4 (0c) | 1.000 |

Average quality is approximately 99.3% of brute-force optimum, and QAOA consistently avoids collision-heavy greedy choices in conflict-prone scenarios.

### 4.2 Scalability Study

| Robots | Qubits | Combinations (4^N) | Brute-Force Time | QAOA Time |
|--------|--------|--------------------|------------------|-----------|
| 2 | 8 | 16 | <1 ms | 514 ms |
| 3 | 12 | 64 | <1 ms | 1,079 ms |
| 4 | 16 | 256 | 1 ms | 1,732 ms |
| 5 | 20 | 1,024 | 4 ms | not run (qubit limit in NumPy backend) |
| 6 | 24 | 4,096 | 33 ms | not run |
| 7 | 28 | 16,384 | 123 ms | not run |
| 8 | 32 | 65,536 | 589 ms | not run |

Current practical result: brute-force is faster at tested sizes; QAOA scaling claim is asymptotic and hardware-dependent.

### 4.3 Candidate Diversity (K sweep)

10x10, 3 robots:

| K | Greedy | Brute-Force | QAOA | Greedy Conflicts |
|---|--------|-------------|------|------------------|
| 2 | 63.5 | 45.2 | 45.2 | 2 |
| 3 | 63.5 | 45.2 | 45.2 | 2 |
| 4 | 63.5 | 44.6 | 45.2 | 2 |
| 5 | 63.5 | 44.6 | 45.2 | 2 |
| 6 | 63.5 | 44.6 | 45.2 | 2 |

Increasing candidate diversity improves globally coordinated solutions, while greedy remains conflict-prone.

### 4.4 QAOA Depth Sweep

10x10, 3 robots, K=4 (brute-force optimum = 44.6):

| Depth p | Best BF/QAOA Ratio | Best Time | Note |
|---------|--------------------|-----------|------|
| 1 | 1.000 | 1,357 ms | optimal reached with restarts |
| 2 | 1.000 | 2,836 ms | optimal reached with restarts |
| 3 | 1.000 | 4,806 ms | optimal reached with restarts |
| 4 | 1.000 | 5,653 ms | optimal reached with restarts |

For these instances, p=1 already provides strong quality-time tradeoff.

### 4.5 Quantum-Inspired Q-Learning Stability

In the 30x30 training experiment (2000 episodes), classical Q-values ranged roughly from -24 to 100, while quantum angles remained bounded in [0.06, 2.99] (pi bound is 3.14), demonstrating improved numerical stability.

---

## 5. Novelty and Contributions

1. **QAOA-based multi-robot coordination formulation**
   The core novelty is mapping robot path assignment to QUBO/Ising and solving it with explicit QAOA circuits.

2. **Clear classical-quantum role separation**
   A* handles candidate generation; QAOA handles combinatorial coordination.

3. **Conflict-aware objective with one-hot constraints**
   The optimization directly models inter-robot interactions, not only individual path lengths.

4. **Auxiliary bounded quantum-inspired RL layer**
   Rotation-angle representation provides stable, bounded learning dynamics for local navigation experiments.

5. **Honest evaluation protocol**
   The report explicitly distinguishes demonstrated results (near-optimal quality and conflict handling) from future-scale claims (runtime advantage on larger hardware).

---

## 6. Limitations

1. No empirical wall-clock speedup over brute-force at current tested sizes.
2. Simulator/backend limits prevent direct large-qubit demonstration in this project.
3. QAOA remains variational; quality can depend on depth and optimizer restarts.
4. Multi-robot benchmarks are still moderate in size compared with industrial-scale MRPP.

---

## 7. Reproducibility

```bash
cd src
python main.py
```

Additional scripts for paper-focused experiments are available in the source tree (including QAOA and QIA comparison scripts).

---

## 8. References

1. Hart, Nilsson, Raphael (1968): A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
2. Khatib (1986): Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.
3. Watkins and Dayan (1992): Q-Learning.
4. Dong et al. (2008): Quantum Reinforcement Learning.
5. Farhi, Goldstone, Gutmann (2014): A Quantum Approximate Optimization Algorithm.
6. Lucas (2014): Ising formulations of many NP problems.
