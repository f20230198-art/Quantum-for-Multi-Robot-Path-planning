# Week 1 Summary (Updated to Current Codebase)

## 1. Current Project Status

This document replaces the older Week 1 narrative and reflects what is actually implemented now.

Current status:
- Week 1 static planning pipeline is complete.
- Phase 1 multi-robot + dynamic environment pipeline is complete.
- Phase 2 local planners (APF, Q-learning) are implemented and integrated.
- The system currently uses classical algorithms (A*, Greedy BFS, APF, Q-learning, and Greedy/Random/Brute-force selectors).
- Real-time dynamic replanning is available in the interactive simulator with planner selection (A*, APF, Q-learning).

Not implemented yet (still planned):
- Quantum Q-learning module
- QAOA optimizer module
- Hybrid planner (A* + APF + quantum escape)

## 2. What Algorithms Are Being Used Right Now

### 2.1 Single-Robot Global Planner
- A* (`src/a_star.py`) is the main baseline planner.
- Heuristic: Euclidean distance (default), Manhattan also available.
- Movement: 8-connected grid (cardinal + diagonal).
- Diagonal corner-cutting is blocked (both adjacent cardinal cells must be free).

### 2.2 Single-Robot Comparison Baseline
- Greedy Best-First Search (`greedy_bfs` in `src/visualize.py`).
- Uses only heuristic-to-goal, does not use accumulated cost `g`.
- Used for A* vs non-optimal baseline visualization.

### 2.3 Multi-Robot Candidate Path Generation
Implemented in `src/multi_robot.py`:
- For each robot, K candidate paths are generated.
- Candidate 0: vanilla A* (optimal on current static snapshot).
- Candidates 1..K-1: A* on modified grids with center-region penalty/blocks to create detours.
- Duplicate paths are removed.

### 2.4 Multi-Robot Path Selection (Current Baselines)
Implemented in `src/multi_robot.py`:
- `greedy_select`: shortest path per robot, ignores conflicts.
- `random_select`: random candidate per robot.
- `brute_force_select`: tries all combinations and minimizes:

```text
score = total_cost + conflict_penalty * total_conflicts
```

where `total_conflicts` counts same-cell, same-timestep overlaps between robots.

### 2.5 Dynamic Environment and Simulation
Two simulation modes are currently present:

1) Offline time-stepped simulation (`src/dynamic_env.py`)
- Dynamic obstacles move each tick.
- Robots follow preselected paths one step per tick.
- Collision checks:
  - robot-robot (same cell at same tick)
  - robot-dynamic obstacle
- Returns full tick-by-tick logs and per-robot stats.

2) Interactive real-time simulator (`src/simulator.py`) — **Supports Multiple Planners**
- Pygame-based live interactive simulator.
- Dynamic obstacles move autonomously (bounce/loop pattern).
- User can add/remove static obstacles mid-simulation with mouse clicks.
- **Planner selection:** Press `P` to cycle between A*, APF, and Q-Learning planners.
- **Robot Algorithm:** Each robot continuously checks if its next planned path step is blocked (by static walls, other robots, or moving obstacles).
  - If the next cell is free and robots don't collide → robot moves forward one cell.
  - If next cell is blocked OR robot gets stuck for 3+ ticks → runs the selected planner from current position to goal using the **live updated grid**.
  - This creates reactive, real-time replanning behavior.
- Priority-order collision handling: robots move in ID order (robot 0 → 1 → 2...), lower ID has higher priority.
- Collision detection: robot-robot and robot-dynamic obstacle overlaps are counted and displayed in UI.

### 2.6 Artificial Potential Field (APF)
Implemented in `src/apf.py`:
- **Local, reactive planner** — makes one-step-at-a-time decisions.
- At each step, computes forces on the robot:
  - **Attractive force**: pulls toward the goal (parabolic potential, proportional to distance²).
  - **Repulsive force**: pushes away from nearby obstacles within an influence radius.
- Robot moves to the neighbour with the lowest total potential.
- **Stuck detection**: if robot stays within a 2-cell radius for 15+ steps, it is declared stuck (local minimum).
- **Known weakness**: Gets stuck in U-shaped traps and concave obstacles where attractive and repulsive forces cancel out. This is the problem quantum Q-learning will solve.

### 2.7 Standard Q-Learning
Implemented in `src/q_learning.py`:
- **Reinforcement learning baseline** — robot learns by trial and error.
- State: `(row, col)` cell position.
- Actions: 4 cardinal directions (up, down, left, right).
- Reward structure: +100 goal, -10 wall/obstacle, -1 step cost, -3 revisit penalty.
- **Training**: ε-greedy exploration with decaying ε (1.0 → 0.05). Default 500 episodes.
- **Planning**: After training, follow greedy policy (highest Q-value at each state).
- **Update rule**: Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
- Loop detection in planning phase: tries second-best action if best leads to revisited cell.
- **Purpose**: Baseline RL. Quantum Q-learning (Week 3) will replace the Q-table with quantum probability amplitudes (α,β) and rotation gates to achieve faster convergence.

## 3. Current Architecture

## 3.1 Module Responsibilities

- `src/grid.py`
  - Environment model (static + dynamic obstacles)
  - Reachable random map generation
  - Neighbor queries and collision-aware cell checks
  - Scenario factories: trap, narrow passage, multi-robot env

- `src/a_star.py`
  - Core shortest-path solver
  - Search bookkeeping: expansion order, g/f scores, nodes expanded

- `src/apf.py`
  - Artificial Potential Field local planner
  - Attractive + repulsive potential computation
  - Stuck detection (local minimum identification)

- `src/q_learning.py`
  - Tabular Q-learning with ε-greedy exploration
  - Training (configurable episodes, learning rate, discount)
  - Greedy policy planning from trained Q-table
  - Loop detection in planning phase

- `src/multi_robot.py`
  - Candidate generation per robot
  - Conflict matrix computation
  - Cost vector creation
  - Classical selection baselines + evaluation

- `src/dynamic_env.py`
  - Batch simulation engine
  - Tick snapshots and performance statistics

- `src/simulator.py`
  - Interactive real-time simulation UI
  - Live replanning with selectable planner (A*, APF, Q-learning)
  - User-driven environment edits
  - Q-table pre-training when Q-learning planner is selected

- `src/visualize.py`
  - Static plots, algorithm comparisons, candidate path plots
  - GIF generation for robot motion and full simulation playback

- `src/main.py`
  - End-to-end demo orchestrator
  - Runs Week 1 + Phase 1 + Phase 2 outputs in one script

## 3.2 Data Flow (main pipeline)

`src/main.py` executes this sequence:

1. Build random single-robot grid.
2. Run A* and save:
   - path image
   - search step strip
   - robot-walk GIF
   - A* vs Greedy BFS comparison
   - trap and narrow passage outputs
3. Build multi-robot dynamic environment.
4. Generate candidate paths per robot.
5. Run classical selectors (greedy/random/brute-force).
6. Evaluate each selector using cost+conflict score.
7. Simulate selected paths in dynamic environment.
8. Save multi-robot images and simulation GIF.
9. **Phase 2:** Run APF on U-shaped trap → demonstrates local minimum.
10. **Phase 2:** Train Q-learning on trap → demonstrates RL solution.
11. **Phase 2:** Run all 3 planners on standard grid → comparison image.

## 4. Dynamic Environment Details (Now Implemented)

In `src/grid.py` each dynamic obstacle has:
- position
- direction
- speed
- movement pattern (`bounce` or `loop`)

Movement logic:
- `bounce`: reverse direction on wall/boundary hit; stay if both forward/reverse blocked.
- `loop`: wrap around edges; stay if wrapped cell is static obstacle.

This dynamic layer is now integrated into:
- visualization (`src/visualize.py`)
- batch simulation (`src/dynamic_env.py`)
- interactive replanning simulation (`src/simulator.py`)
- APF obstacle sensing (`src/apf.py`)

## 5. Outputs Generated by Current `main.py`

The following files are produced in `experiments/results/`:

### Week 1 Outputs
- `week1_demo.png`
- `week1_search_steps.png`
- `week1_robot_walk.gif`
- `week1_comparison.png`
- `week1_trap_u-shape.png`
- `week1_narrow_passage.png`

### Phase 1 Outputs
- `phase1_environment.png`
- `phase1_candidate_paths.png`
- `phase1_selection_comparison.png`
- `phase1_bruteforce_paths.png`
- `phase1_simulation.gif`

### Phase 2 Outputs (NEW)
- `phase2_apf_trap.png` — APF attempting U-trap (gets STUCK)
- `phase2_qlearning_trap.png` — Q-Learning solving U-trap
- `phase2_planner_comparison.png` — Side-by-side A* vs APF vs Q-Learning

## 6. Updated Technology Stack

- Python 3.x
- NumPy
- Matplotlib
- Pygame (for interactive simulator)

(From `requirements.txt`: `numpy`, `matplotlib`, `pygame`)

## 7. How to Run (Current)

From project root:

```powershell
# Main pipeline (Week 1 + Phase 1 + Phase 2 outputs)
python src/main.py

# Interactive real-time simulator
python src/simulator.py

# Optional simulator arguments
python src/simulator.py --robots 5 --size 25 --dynamic 8
```

### Simulator Controls
| Key | Action |
|-----|--------|
| SPACE | Pause / Resume |
| RIGHT | Step forward (when paused) |
| UP/DOWN | Speed up / slow down |
| R | Reset simulation |
| **P** | **Cycle planner (A* → APF → Q-Learning)** |
| G | Toggle ghost paths |
| T | Toggle trails |
| L-Click | Place/remove wall |
| M-Click | Spawn dynamic obstacle |
| R-Click | Remove/drag obstacle |
| Q/ESC | Quit |

## 8. What This Version Demonstrates

- Classical static planning works (A*, Greedy BFS baseline).
- Multi-robot candidate-path architecture is implemented.
- Conflict-aware path selection baselines are implemented.
- Dynamic obstacles and time-stepped simulation are implemented.
- Real-time reactive replanning is implemented in interactive mode.
- **APF local planner is implemented and demonstrates local minimum weakness.**
- **Q-learning RL baseline is implemented with training + policy extraction.**
- **Interactive simulator supports planner switching (A*, APF, Q-Learning).**

## 9. Next Milestone (Accurate to Current Code)

Next logical implementation step:
- Add quantum Q-learning module (`quantum_q_learning.py`) that uses probability amplitudes (α, β) and rotation gates instead of scalar Q-values.
- Add hybrid planner (`hybrid_planner.py`) that uses A* for global planning, APF for local navigation, and quantum Q-learning to escape local minima.
- Compare quantum Q-learning convergence speed against standard Q-learning.

After that:
- Add QAOA optimizer module for multi-robot path selection (currently referenced as future work).
