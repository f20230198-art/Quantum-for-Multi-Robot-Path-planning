# Quantum-Inspired Multi-Robot Path Planning

A hybrid quantum-classical system where multiple robots navigate dynamic 2D grid environments. Classical algorithms handle the basics, quantum-inspired algorithms solve the hard parts — escaping traps and coordinating multiple robots at scale.

Everything runs on a regular laptop. No quantum hardware needed.

---

## The Problem

Multiple robots need to find collision-free paths through a grid with static walls and moving obstacles. This breaks down into two challenges:

1. **Single-robot navigation** — Simple planners like APF get stuck in U-shaped traps. A* finds optimal paths but can't react to moving obstacles.
2. **Multi-robot coordination** — Picking the best combination of paths for N robots is an exponential problem. 3 robots with 4 path options = 64 combinations. 20 robots = over a trillion.

## Our Approach

We built 6 algorithms, layered from simple to advanced:

| Algorithm | Type | What It Does | Limitation |
|-----------|------|-------------|------------|
| **A*** | Classical | Finds shortest path on static grid | Can't handle dynamic obstacles |
| **APF** | Classical | Reacts to obstacles in real-time | Gets stuck in concave traps |
| **Q-Learning** | Classical RL | Learns navigation policy offline | Q-values can diverge, slow to train |
| **Quantum Q-Learning** | Quantum-Inspired | Replaces Q-values with qubit rotation angles | Slower than A* on simple grids |
| **Hybrid Planner** | Multi-Layer | A* (global) + APF (local) + Quantum QL (escape) | Training overhead |
| **QAOA** | Quantum Optimization | Selects collision-free path combinations for all robots | Simulation cost scales with qubits |

The key insight: **no single algorithm works everywhere**. A* is unbeatable on static grids, but useless when obstacles move. APF reacts fast but gets trapped. Quantum QL escapes traps. The Hybrid planner combines all three. QAOA handles multi-robot coordination.

---

## Project Structure

```
src/
    main.py                   # Run this to generate all results
    simulator_classical.py    # Live demo: A*, APF, Q-Learning
    simulator_quantum.py      # Live demo: Quantum QL, Hybrid, QAOA
    simulator_base.py         # Shared simulator code

    a_star.py                 # A* search algorithm
    apf.py                    # Artificial Potential Field planner
    q_learning.py             # Tabular Q-Learning
    quantum_q_learning.py     # Quantum-inspired Q-Learning (rotation gates)
    hybrid_planner.py         # 3-layer hybrid: A* + APF + Quantum QL
    qaoa_optimizer.py         # QAOA multi-robot path optimizer

    grid.py                   # Grid environment with obstacles
    multi_robot.py            # Candidate path generation & selection
    dynamic_env.py            # Time-stepped simulation engine
    visualize.py              # All plotting and visualization functions
    utils.py                  # Shared utilities

    run_classical.py          # Generates classical algorithm results
    run_quantum.py            # Generates quantum algorithm results
    run_comparison.py         # Generates comparisons & novelty demos

docs/
    Project_Proposal.md       # Full project proposal with references
    Final_Report.md           # Final report with results and analysis
    Progress_and_Demo_Guide.md

experiments/results/          # Generated output (see "Results" section below)

1.pdf, 2.pdf, 3.pdf          # Reference papers
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone <repo-url>
cd "Quantam for Multi Robot Path planning"
pip install -r requirements.txt
```

The core dependencies are `numpy`, `matplotlib`, `pygame`, and `scipy`. Qiskit is optional — the project uses a numpy-based quantum simulation fallback when Qiskit is not installed.

### Generate All Results

```bash
cd src
python main.py
```

This runs all algorithms and saves 27 images, GIFs, and a benchmark table into numbered folders under `experiments/results/`. Takes about 3-5 minutes depending on your machine (QAOA scalability benchmark is the bottleneck).

### Interactive Demos

These open a pygame window where you can watch robots navigate in real-time, place/remove obstacles, and switch between algorithms.

**Classical algorithms:**
```bash
cd src
python simulator_classical.py
```

**Quantum algorithms:**
```bash
cd src
python simulator_quantum.py
```

#### Simulator Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `P` | Cycle to next planner |
| `RIGHT` | Step one tick (when paused) |
| `UP / DOWN` | Speed up / slow down |
| `R` | Reset simulation |
| `G` | Toggle ghost paths |
| `T` | Toggle trails |
| Left-click | Place / remove wall |
| Middle-click | Spawn moving obstacle |
| Right-click | Remove / drag dynamic obstacle |
| `Q` or `ESC` | Quit |

#### What to show during a demo

1. **Start with `simulator_classical.py`** — press SPACE, watch A* work perfectly
2. Middle-click to spawn moving obstacles — A* keeps replanning, APF reacts but gets stuck near walls
3. Build a U-shaped wall with left-clicks — press P to switch to APF, watch it get trapped
4. **Switch to `simulator_quantum.py`** — start with Quantum QL, watch it escape the trap
5. Press P for Hybrid — see all 3 layers cooperate (A* plans, APF steers, Quantum QL escapes)
6. Press P for QAOA — coordinated multi-robot path selection

---

## Results

After running `python main.py`, results are organized into 6 folders that tell the story of the project:

### `01_classical_baseline/`
A* search fundamentals — optimal paths, search visualization, Greedy BFS comparison, trap and narrow passage scenarios.

### `02_multi_robot_challenge/`
Multi-robot environment setup — 3 robots, 4 dynamic obstacles, candidate path generation, classical selection methods (greedy, random, brute-force).

### `03_classical_limitations/`
Where classical algorithms break down — APF stuck in U-shaped trap, Q-Learning comparison, 3-way planner comparison showing each method's weakness.

### `04_quantum_advantage/`
Quantum-inspired solutions — Quantum Q-Learning escaping the U-trap, learning curve comparison (classical vs quantum), Hybrid planner combining A* + APF + Quantum QL, 4-way planner comparison.

### `05_qaoa_coordination/`
QAOA multi-robot optimizer — path selection comparison (QAOA vs classical methods), step-by-step QAOA explanation figure, scalability benchmark (QAOA vs brute-force), simulation with QAOA-selected paths.

### `06_comprehensive_comparison/`
Head-to-head evidence — all 5 planners across obstacle densities (10-30%), Q-value divergence demo (classical values explode to -28..+100, quantum angles stay bounded in 0..3.14), dynamic environment demo (A* fails when obstacles move, Hybrid adapts), benchmark table.

---

## How the Algorithms Work

### Quantum Q-Learning

Standard Q-Learning stores a scalar Q-value for each (state, action) pair. These values can grow unbounded, leading to instability.

Quantum Q-Learning replaces Q-values with **rotation angles** on the Bloch sphere. Each (state, action) pair gets an angle theta, and the probability of choosing that action is sin^2(theta/2). The angle is clamped to [0, pi], so values never diverge.

```
Classical:  Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
                     ^ unbounded, can go to +/- infinity

Quantum:    theta(s,a) = clamp(theta + alpha * delta, 0, pi)
                         ^ always bounded in [0, pi]
```

### Hybrid Planner

Three layers that activate based on the situation:

1. **A* (Layer 1)** — Plans the global route from current position to goal
2. **APF (Layer 2)** — Follows the A* route but reacts locally to moving obstacles
3. **Quantum QL (Layer 3)** — Activated only when APF detects it's stuck in a local minimum. Uses the trained quantum policy to escape, then hands back to A* for replanning.

### QAOA (Quantum Approximate Optimization Algorithm)

For multi-robot coordination, we formulate path selection as a QUBO (Quadratic Unconstrained Binary Optimization) problem:

- Binary variable x[i,k] = 1 if robot i takes path k
- Minimize: total path cost + penalty * number of collisions
- Constraint: each robot picks exactly one path (one-hot encoding)

QAOA uses alternating cost and mixer unitaries to search the solution space. On a classical simulator, we do full statevector simulation. On real quantum hardware, QAOA circuit depth scales polynomially while brute-force scales exponentially.

---

## Key Findings

| Scenario | Best Algorithm | Why |
|----------|---------------|-----|
| Static grid, single robot | A* | Guaranteed optimal, fast |
| U-shaped trap | Quantum QL or Hybrid | APF gets stuck, A* doesn't help if you're already in the trap |
| Dynamic obstacles | Hybrid | A* can't react, APF gets stuck, Hybrid combines both + escape |
| Multi-robot coordination | QAOA | Brute-force is exponential, QAOA scales polynomially on quantum hardware |
| High obstacle density (30%) | A* or Classical QL | Quantum QL struggles with very cluttered grids |

A* wins on simple static grids — that's expected and honest. The quantum-inspired approaches shine in dynamic environments, trap scenarios, and multi-robot coordination where classical methods hit their limits.

---

## References

1. Hart, P.E., Nilsson, N.J., Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" — A* algorithm
2. Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots" — Potential fields
3. Dong, D., Chen, C., Li, H., Tarn, T.J. (2008). "Quantum Reinforcement Learning" — Quantum-inspired RL with rotation gates
4. Farhi, E., Goldstone, J., Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm" — QAOA

See `1.pdf`, `2.pdf`, `3.pdf` in the project root for the full reference papers.

---

## Dependencies

**Required:**
- `numpy` — numerical computation
- `matplotlib` — visualization and plotting
- `pygame` — interactive simulator
- `scipy` — QAOA optimizer (COBYLA)

**Optional:**
- `qiskit`, `qiskit-aer` — real quantum circuit simulation (falls back to numpy if not installed)
- `qiskit-optimization` — QAOA via Qiskit (falls back to custom numpy implementation)

Install everything with:
```bash
pip install -r requirements.txt
```
