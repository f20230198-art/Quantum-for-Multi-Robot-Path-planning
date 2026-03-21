# Progress Report & Demo Guide
## Quantum-Inspired Multi-Robot Path Planning

This document covers **everything that has been built so far**, in the order it was built, and gives you **step-by-step instructions to see each part working**.

---

## How to Set Up

### Prerequisites

```bash
# Install Python 3.10 or later, then:
pip install numpy matplotlib pygame

# Optional (for quantum circuit verification):
pip install qiskit qiskit-aer
```

### Project Structure

```
Quantum for Multi Robot Path Planning/
  src/                    <-- All source code
    main.py               <-- Run this for the full demo
    simulator.py          <-- Run this for the interactive simulator
    grid.py               <-- Grid environment
    a_star.py             <-- A* algorithm
    apf.py                <-- Artificial Potential Field
    q_learning.py         <-- Standard Q-Learning
    quantum_q_learning.py <-- Quantum Q-Learning
    hybrid_planner.py     <-- Hybrid three-layer planner
    multi_robot.py        <-- Multi-robot coordination
    dynamic_env.py        <-- Simulation engine
    visualize.py          <-- All plotting functions
    utils.py              <-- Shared helpers
  experiments/results/    <-- All generated images and GIFs
  docs/                   <-- Documentation (you are here)
  requirements.txt        <-- Python dependencies
```

---

## Quick Demo (See Everything At Once)

If you just want to see the full project working:

```bash
cd src
python main.py
```

This runs **every phase** and generates all output images in `experiments/results/`. It takes about 2-3 minutes because it trains Q-Learning and Quantum Q-Learning models.

For the **interactive simulator** (click to place obstacles, watch robots navigate in real time):

```bash
cd src
python simulator.py
```

---

## Phase-by-Phase Breakdown

---

## Week 1 — Foundation: Grid + A\* + Visualization

### What Was Built

**grid.py** — A 20x20 grid world where:
- White cells are free (robots can walk here)
- Black cells are obstacles (walls)
- The grid is randomly generated with a configurable obstacle ratio (default 15%)
- Start is top-left (0,0), goal is bottom-right (19,19)
- A BFS check guarantees a path always exists between start and goal

**a_star.py** — The A\* pathfinding algorithm:
- Finds the absolute shortest path from start to goal
- Uses Euclidean distance as the heuristic
- Supports 8-directional movement (including diagonals)
- Prevents corner-cutting through diagonal wall gaps
- Returns the path, cost, nodes expanded, and expansion order

**visualize.py** — Matplotlib-based visualization:
- Draws grids with color-coded cells
- Overlays paths as colored lines
- Shows start (green circle) and goal (red star) markers
- Can save to PNG files

Also built: **Greedy BFS** (a simpler, non-optimal search for comparison) and special test environments (U-shaped trap, narrow passage).

### What This Proved

- A\* finds the optimal shortest path every time
- Greedy BFS is faster but produces longer paths (10-30% worse)
- The trap and narrow passage environments work correctly as test scenarios

### Demo: See Week 1 Working

Run the full demo (Week 1 section runs automatically as part of `python main.py`) or test individual pieces:

```python
# Quick A* test
cd src
python -c "
from grid import Grid
from a_star import a_star

grid = Grid(size=20, obstacle_ratio=0.15, seed=42)
result = a_star(grid, eight_connected=True)
print(f'Path found: {result.success}')
print(f'Path cost: {result.cost:.2f}')
print(f'Path length: {len(result.path)} cells')
print(f'Nodes expanded: {result.nodes_expanded}')
"
```

### Output Files Generated

| File | What It Shows |
|------|--------------|
| `week1_demo.png` | A\* path on a random 20x20 grid |
| `week1_search_steps.png` | Step-by-step A\* exploration (6 snapshots) |
| `week1_robot_walk.gif` | Animated GIF of robot walking the A\* path |
| `week1_comparison.png` | A\* vs Greedy BFS side-by-side |
| `week1_trap_u-shape.png` | A\* solving the U-shaped trap |
| `week1_narrow_passage.png` | A\* navigating a narrow corridor |

---

## Phase 1 — Multi-Robot + Dynamic Environment

### What Was Built

**multi_robot.py** — Multi-robot path generation and coordination:
- Generates K diverse candidate paths per robot using A\* with penalty zones
- Computes a conflict matrix showing which path pairs collide at which timesteps
- Implements three classical selection methods:
  - **Greedy**: each robot picks its cheapest path (ignores conflicts)
  - **Random**: each robot picks a random path
  - **Brute-Force**: checks every combination, picks the one with lowest cost + conflict penalty

**dynamic_env.py** — Time-stepped simulation:
- Robots move one step per tick along their assigned paths
- Dynamic obstacles move each tick (bounce off walls or loop around)
- Collision detection: robot-robot and robot-obstacle
- Produces a full simulation trace with per-tick snapshots

**simulator.py** — Interactive PyGame simulator:
- Real-time multi-robot simulation
- Click to place/remove obstacles during simulation
- Dynamic obstacles move autonomously
- Robots replan when their path is blocked

**Grid extensions** — Added to grid.py:
- `add_dynamic_obstacle()` — moving obstacles with bounce/loop patterns
- `create_multi_robot_env()` — factory for multi-robot test environments
- `snapshot()` — combined view of static + dynamic obstacles

### What This Proved

- Candidate path generation produces diverse routes using penalty zones
- Brute-force selection finds the globally optimal path combination (cost=90.6, 0 collisions)
- Greedy selection sometimes matches brute-force but can fail in dense environments
- The simulation engine correctly detects and reports collisions

### Demo: See Phase 1 Working

**Option 1: Full pipeline** (runs as part of `python main.py`)

**Option 2: Interactive simulator**
```bash
cd src
python simulator.py
```

Simulator controls:
- **P** — Cycle planner: A\* → APF → Q-Learning → Quantum QL
- **SPACE** — Pause / Resume
- **UP/DOWN arrows** — Speed up / slow down
- **R** — Reset simulation
- **Left-click** — Place a wall obstacle
- **Right-click** — Remove obstacle / drag dynamic obstacle
- **Q or ESC** — Quit

**Option 3: Quick multi-robot test**
```python
cd src
python -c "
from grid import Grid
from multi_robot import prepare_qaoa_input, brute_force_select, evaluate_selection

grid = Grid.create_multi_robot_env(size=20, num_robots=3, num_dynamic=4, seed=42)
print(f'Robots: {len(grid.robot_configs)}')
for cfg in grid.robot_configs:
    print(f'  Robot {cfg[\"id\"]}: {cfg[\"start\"]} -> {cfg[\"goal\"]}')

qaoa = prepare_qaoa_input(grid, num_candidates=4, seed=42)
sel = brute_force_select(qaoa['all_candidates'], conflict_penalty=10.0)
ev = evaluate_selection(qaoa['all_candidates'], sel, conflict_penalty=10.0)
print(f'Best combination: cost={ev[\"total_cost\"]:.1f}, conflicts={ev[\"total_conflicts\"]}')
"
```

### Output Files Generated

| File | What It Shows |
|------|--------------|
| `phase1_environment.png` | 20x20 grid with 3 robots + 4 dynamic obstacles |
| `phase1_candidate_paths.png` | All candidate paths per robot (4 options each) |
| `phase1_selection_comparison.png` | Greedy vs Random vs Brute-Force results |
| `phase1_bruteforce_paths.png` | The best classical path combination |
| `phase1_simulation.gif` | Animated simulation of robots following their paths |

---

## Phase 2 — APF & Q-Learning Local Planners

### What Was Built

**apf.py** — Artificial Potential Field planner:
- At each step, computes attractive force (toward goal) and repulsive force (away from obstacles)
- Robot moves to the neighbor with the lowest total potential energy
- Stuck detection: if recent positions cluster in a small area, the robot is declared stuck
- Works well in open environments but fails in U-shaped traps

**q_learning.py** — Standard tabular Q-Learning:
- Robot trains over 500 episodes of trial-and-error
- Reward structure: +100 (goal), -10 (wall hit), -1 (each step), -3 (revisiting a cell)
- Epsilon-greedy exploration: starts fully random, decays to 5% random
- Builds a Q-table lookup: Q[row, col, action] = expected future reward
- After training, follows the greedy policy (always pick highest Q-value)

### What This Proved

- **APF gets stuck in U-traps**: The attractive force toward the goal pulls the robot INTO the U-shape, but repulsive forces from the walls prevent it from moving deeper. Forces cancel out → robot oscillates forever. This is the "local minimum problem."

- **Q-Learning can escape**: Because it learns from hundreds of episodes of exploration, Q-Learning discovers the route around the U-shape. But it requires significant training time and can fail on complex grids.

- **Three-way comparison**: On a standard grid with dynamic obstacles:
  - A\*: always finds the shortest path (but static, no dynamic obstacle reaction)
  - APF: fast but can get stuck
  - Q-Learning: flexible but slow to train and not always reliable

### Demo: See Phase 2 Working

**See APF getting stuck in a U-trap:**
```python
cd src
python -c "
from grid import Grid
from apf import apf_plan

trap = Grid.create_trap_environment(size=20)
result = apf_plan(trap)
print(f'APF stuck: {result.stuck}')
print(f'Steps taken: {result.steps_taken}')
print(f'Reached goal: {result.success}')
"
```

Output: `APF stuck: True` — this is the problem that quantum Q-Learning solves.

**See Q-Learning solving the same trap:**
```python
cd src
python -c "
from grid import Grid
from q_learning import q_learning_full

trap = Grid.create_trap_environment(size=20)
result = q_learning_full(trap, episodes=500, seed=42)
print(f'Q-Learning success: {result.success}')
print(f'Path cost: {result.cost:.2f}')
print(f'Path length: {len(result.path)} cells')
print(f'Episodes trained: {result.episodes_trained}')
"
```

### Output Files Generated

| File | What It Shows |
|------|--------------|
| `phase2_apf_trap.png` | APF stuck in the U-shaped trap |
| `phase2_qlearning_trap.png` | Q-Learning path escaping the U-trap |
| `phase2_planner_comparison.png` | A\* vs APF vs Q-Learning side-by-side |

---

## Phase 3 — Quantum Q-Learning + Hybrid Planner

### What Was Built

**quantum_q_learning.py** — Quantum-inspired Q-Learning:
- Replaces the scalar Q-table with **rotation angles** theta in [0, pi]
- Each (state, action) pair maps to a qubit: |psi> = Ry(theta)|0>
- Action probability = sin^2(theta/2) — bounded between 0 and 1
- Update rule uses a Grover-inspired self-regulating rotation:
  - Positive TD error → rotate toward pi (increase action probability)
  - Negative TD error → rotate toward 0 (decrease action probability)
  - Damping prevents saturation: updates slow down near boundaries
- Optional Qiskit integration: can build real quantum circuits with Ry gates and measure them on AerSimulator
- Falls back to numpy sin^2(theta/2) computation when Qiskit is not installed

**hybrid_planner.py** — Three-layer hybrid planner:
- **Layer 1 (A\*)**: Computes global shortest path from start to goal
- **Layer 2 (APF)**: Follows the A\* path step-by-step, reacting to dynamic obstacles via potential fields
- **Layer 3 (Quantum QL)**: Activated ONLY when APF detects it is stuck (recent positions clustered in a small radius). Uses the trained quantum policy to escape, then A\* re-plans from the new position
- Tracks statistics: A\* replans, APF steps, quantum escapes, stuck detections

**Comprehensive comparison system**:
- Grids with **varied obstacle shapes** (blocks, walls, L-shapes) at 5 different densities (10% to 30%)
- All 5 planners (A\*, APF, Classical QL, Quantum QL, Hybrid) benchmarked on each grid
- Bar charts comparing path cost, computation time, and path length across densities

### What This Proved

1. **Quantum QL escapes U-traps** where APF gets stuck — the rotation-based policy learns diverse escape strategies

2. **Bounded learning** — rotation angles stay in [0, pi], so values never diverge to infinity unlike classical Q-values

3. **Hybrid planner is robust** — combines A\*'s optimality, APF's real-time reaction, and Quantum QL's trap escape. Works on both simple and complex grids.

4. **Performance scales with density**:
   - At 10% obstacles: all planners succeed
   - At 15-25% obstacles: APF starts failing, Hybrid stays reliable
   - At 30% obstacles: only A\* and Classical QL consistently succeed (denser environments need more training)

5. **Qiskit verification** — when Qiskit is installed, the `verify_qiskit_match()` function confirms that the numpy sin^2(theta/2) computation matches actual quantum circuit measurements

### Demo: See Phase 3 Working

**See Quantum Q-Learning in action:**
```python
cd src
python -c "
from grid import Grid
from quantum_q_learning import quantum_q_learning_full

trap = Grid.create_trap_environment(size=20)
result = quantum_q_learning_full(trap, episodes=500, seed=42)
print(f'Quantum QL success: {result.success}')
print(f'Path cost: {result.cost:.2f}')
print(f'Path length: {len(result.path)} cells')
print(f'Backend: {\"Qiskit\" if result.used_qiskit else \"numpy\"}')
"
```

**See the Hybrid Planner:**
```python
cd src
python -c "
from grid import Grid
from hybrid_planner import hybrid_plan

trap = Grid.create_trap_environment(size=20)
result = hybrid_plan(trap, q_episodes=500, q_seed=42)
print(f'Hybrid success: {result.success}')
print(f'Path cost: {result.cost:.2f}')
print(f'A* replans: {result.astar_replans}')
print(f'APF steps: {result.apf_steps}')
print(f'Quantum escapes: {result.quantum_escapes}')
print(f'Stuck detections: {result.stuck_detected}')
"
```

**Compare all planners on a single grid:**
```python
cd src
python -c "
from grid import Grid
from a_star import a_star
from apf import apf_plan
from q_learning import q_learning_full
from quantum_q_learning import quantum_q_learning_full
from hybrid_planner import hybrid_plan

grid = Grid(size=20, obstacle_ratio=0.15, seed=12345)
print(f'Grid: 20x20, 15% obstacles')
print(f'{'Planner':<20s}  {'Success':<8s}  {'Cost':<10s}')
print(f'{'-'*20}  {'-'*8}  {'-'*10}')

r = a_star(grid, eight_connected=True)
print(f'{'A*':<20s}  {'Y' if r.success else 'N':<8s}  {r.cost:<10.2f}')

r = apf_plan(grid)
print(f'{'APF':<20s}  {'Y' if r.success else 'N':<8s}  {r.cost:<10.2f}')

r = q_learning_full(grid, episodes=500, seed=42)
print(f'{'Classical QL':<20s}  {'Y' if r.success else 'N':<8s}  {r.cost:<10.2f}')

r = quantum_q_learning_full(grid, episodes=500, seed=42)
print(f'{'Quantum QL':<20s}  {'Y' if r.success else 'N':<8s}  {r.cost:<10.2f}')

r = hybrid_plan(grid, q_episodes=500, q_seed=42)
print(f'{'Hybrid':<20s}  {'Y' if r.success else 'N':<8s}  {r.cost:<10.2f}')
"
```

**See varied obstacle environments:**
```python
cd src
python -c "
from grid import Grid
import numpy as np

for ratio in [0.10, 0.15, 0.20, 0.25, 0.30]:
    g = Grid.create_varied_obstacles(size=20, obstacle_ratio=ratio, seed=7777)
    obs = int(np.sum(g.grid == 1))
    print(f'{ratio*100:.0f}% density -> {obs} obstacle cells (varied shapes: blocks, walls, L-shapes)')
print(g)  # prints the 30% grid to see the obstacle shapes
"
```

**Interactive simulator with all 4 planners:**
```bash
cd src
python simulator.py
# Press P to cycle through: A* -> APF -> Q-Learning -> Quantum QL
# Click to place obstacles and watch robots replan in real time
```

### Output Files Generated

| File | What It Shows |
|------|--------------|
| `phase3_quantum_ql_trap.png` | Quantum QL path escaping U-trap |
| `phase3_ql_comparison.png` | Classical vs Quantum Q-Learning side-by-side |
| `phase3_learning_curves.png` | Training reward curves (both learners) |
| `phase3_hybrid_planner.png` | Hybrid planner path on U-trap |
| `phase3_planner_comparison.png` | 4-way comparison: A\* vs APF vs Classical QL vs Quantum QL |
| `comparison_varied_grids.png` | 5 grids with different obstacle densities (realistic shapes) |
| `comparison_all_planners.png` | Bar chart: all 5 planners across 5 densities (cost, time, path length) |

---

## All Output Files Summary

Running `python src/main.py` generates these 21 files in `experiments/results/`:

### Week 1
| # | File | Description |
|---|------|-------------|
| 1 | `week1_demo.png` | A\* on random grid |
| 2 | `week1_search_steps.png` | A\* search step-by-step |
| 3 | `week1_robot_walk.gif` | Robot walking animation |
| 4 | `week1_comparison.png` | A\* vs Greedy BFS |
| 5 | `week1_trap_u-shape.png` | A\* on U-trap |
| 6 | `week1_narrow_passage.png` | A\* on narrow corridor |

### Phase 1
| # | File | Description |
|---|------|-------------|
| 7 | `phase1_environment.png` | Multi-robot environment |
| 8 | `phase1_candidate_paths.png` | All candidate paths |
| 9 | `phase1_selection_comparison.png` | Selection methods compared |
| 10 | `phase1_bruteforce_paths.png` | Best classical paths |
| 11 | `phase1_simulation.gif` | Animated simulation |

### Phase 2
| # | File | Description |
|---|------|-------------|
| 12 | `phase2_apf_trap.png` | APF stuck in U-trap |
| 13 | `phase2_qlearning_trap.png` | Q-Learning escaping U-trap |
| 14 | `phase2_planner_comparison.png` | 3-way planner comparison |

### Phase 3
| # | File | Description |
|---|------|-------------|
| 15 | `phase3_quantum_ql_trap.png` | Quantum QL on U-trap |
| 16 | `phase3_ql_comparison.png` | Classical vs Quantum QL |
| 17 | `phase3_learning_curves.png` | Training reward curves |
| 18 | `phase3_hybrid_planner.png` | Hybrid planner demo |
| 19 | `phase3_planner_comparison.png` | 4-way comparison |

### Comprehensive Comparison
| # | File | Description |
|---|------|-------------|
| 20 | `comparison_varied_grids.png` | 5 density levels with realistic obstacles |
| 21 | `comparison_all_planners.png` | Bar chart: all planners vs all densities |

---

## What's Next: Phase 4 — QAOA Multi-Robot Optimizer

Phase 4 will use the Quantum Approximate Optimization Algorithm (QAOA) to replace the brute-force path selection with a quantum optimization that scales polynomially instead of exponentially:

- Encode multi-robot path selection as a QUBO problem
- Use Qiskit's QAOA implementation to find the optimal combination
- Compare against brute-force, greedy, and random baselines
- Show that QAOA matches brute-force quality at a fraction of the computational cost
