# Full Phase 2 Explanation — APF & Q-Learning Local Planners

## Complete Guide: What Was Built, Why, and What's Next

---

## 1. Where We Are in the Project Timeline

| Phase | Status | What It Covers |
|-------|--------|----------------|
| **Week 1** | ✅ Complete | Grid environment, A* pathfinding, Greedy BFS, visualization |
| **Phase 1** | ✅ Complete | Multi-robot candidate paths, conflict matrix, classical selectors (greedy/random/brute-force), dynamic environment, time-stepped simulation, interactive PyGame simulator |
| **Phase 2** | ✅ Complete | **APF local planner, Q-learning RL baseline, planner comparison, simulator planner switching** |
| Phase 3 | 🔲 Next | Quantum Q-learning module, hybrid planner (A* + APF + quantum escape) |
| Phase 4 | 🔲 Planned | QAOA optimizer for multi-robot path selection |

**Phase 2 added two new local planning algorithms** — Artificial Potential Fields (APF) and standard Q-learning — as building blocks and baselines for the quantum-enhanced system.

---

## 2. Why Phase 2 Exists — The Story So Far

### The Problem Chain

Each phase solves a weakness of the previous one:

```
Phase/Week 1: A* finds shortest paths → BUT can't handle moving obstacles
    ↓
Phase 1: Multi-robot coordination with dynamic obstacles → BUT robots follow
         FIXED pre-planned paths. If a robot gets stuck, it has no escape.
    ↓
Phase 2: Local planners (APF, Q-learning) that can react in real-time
         → BUT APF gets stuck in traps, and Q-learning is slow to learn.
    ↓
Phase 3 (NEXT): Quantum Q-learning that learns faster and escapes traps
         → The KEY CONTRIBUTION of the project.
```

**Phase 2's purpose**: Give the robot LOCAL (step-by-step) navigation abilities AND demonstrate their weaknesses, setting up the motivation for quantum Q-learning in Phase 3.

---

## 3. What Was Built in Phase 2

### 3.1 Artificial Potential Field (APF) — `src/apf.py`

**File size**: 280 lines of Python

#### The Concept (Analogy)

Imagine the robot as a **ball on a hilly landscape**:
- The **goal** creates a **valley** — the ball naturally rolls toward it
- Each **obstacle** creates a **hill** — the ball rolls away from it
- The ball always moves "downhill" to wherever the combined landscape is lowest

This is APF — the robot doesn't plan a full path. At each step it just checks: "Which neighboring cell has the lowest combined potential (most pull toward goal, least push from obstacles)?" and moves there.

#### How It Works (Technical Detail)

**Two force types computed at every position:**

1. **Attractive Potential** (pulls toward goal):
   ```
   U_att(pos) = 0.5 × k_att × distance(pos, goal)²
   ```
   - `k_att = 1.0` (attraction strength)
   - Parabolic — grows stronger the further you are from the goal

2. **Repulsive Potential** (pushes away from obstacles):
   ```
   For each obstacle within influence_radius (3 cells):
       U_rep = 0.5 × k_rep × (1/d − 1/radius)²
   ```
   - `k_rep = 100.0` (repulsion strength)
   - Only affects nearby obstacles (within 3 cells)
   - Grows to infinity as distance → 0

3. **Total Potential**: `U_total = U_att + U_rep`

**At each step:**
1. Calculate `U_total` at all neighboring cells
2. Add a small penalty for revisited cells (to reduce oscillation)
3. Move to the neighbor with the **lowest** `U_total`
4. Repeat until goal reached, stuck, or max steps

#### Stuck Detection (Local Minimum Identification)

APF includes automatic detection for when the robot is trapped:
- Looks at the last **15 positions** the robot visited
- Calculates the centroid (center point) of those positions
- If **ALL** recent positions are within a **2-cell radius** of that center AND the robot has been walking for **12+ steps** → declares the robot **STUCK**

This is the **local minimum problem** — the core weakness that motivates quantum Q-learning.

#### Key Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `k_att` | 1.0 | Goal attraction strength |
| `k_rep` | 100.0 | Obstacle repulsion strength |
| `influence_radius` | 3.0 | How close obstacles need to be to push the robot |
| `max_steps` | 500 | Timeout if goal not reached |
| `eight_connected` | True | Allow diagonal movement |

#### What APF Produces (Output)

An `APFResult` object containing:
- `path` — list of positions from start onward (may be partial if stuck)
- `cost` — Euclidean path cost
- `steps_taken` — how many steps the robot walked
- `success` — True if goal was reached
- `stuck` — True if local minimum was detected

---

### 3.2 Standard Q-Learning — `src/q_learning.py`

**File size**: 369 lines of Python

#### The Concept (Analogy)

Imagine training a dog to navigate a maze:
- Put the dog at the start
- Let it wander randomly
- When it bumps into a wall → "BAD!" (negative reward)
- When it takes a step → "meh, keep going" (small negative — encourages short paths)
- When it reaches the goal → "GREAT!" (big positive reward)
- Repeat hundreds of times
- Eventually the dog LEARNS the best route for every position

Q-learning does exactly this. The "dog's memory" is the **Q-table** — a lookup table storing how good each action is at each position.

#### How It Works (Technical Detail)

**Two distinct phases:**

##### Phase A: Training (500 episodes of trial-and-error)

```
For each episode (1 to 500):
    Place robot at start
    For each step:
        Choose action (ε-greedy):
            With probability ε → random move  (EXPLORE)
            With probability 1-ε → best Q-value move  (EXPLOIT)
        
        Execute action → observe reward
        
        Update Q-table:
            Q(s, a) ← Q(s, a) + α × [reward + γ × max Q(s', a') − Q(s, a)]
    
    Decay ε (less exploration over time):
        ε ← max(0.05, ε × 0.995)
```

**Reward structure:**

| Event | Reward | Why |
|-------|--------|-----|
| Reached goal | **+100** | The main objective |
| Hit wall/obstacle | **-10** | Don't do this |
| Revisited a cell | **-3** | Discourages loops |
| Normal step | **-1** | Encourages shorter paths |

**Key hyperparameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `α` (alpha) | 0.1 | Learning rate — how much new info overwrites old |
| `γ` (gamma) | 0.95 | Discount factor — how much future rewards matter |
| `ε` start | 1.0 | Start fully random (100% exploration) |
| `ε` end | 0.05 | End with minimal randomness (5% exploration) |
| `ε` decay | 0.995 | Multiply ε by this each episode |
| Episodes | 500 | Number of training runs |

##### Phase B: Planning (follow the learned policy)

After training, the Q-table is a complete "cheat sheet." The robot simply:
1. Look at current position → check Q-values for all 4 actions
2. Pick the action with the **highest Q-value**
3. Move there
4. Repeat until goal

**Loop detection in planning**: If the best action leads to an already-visited cell, it tries the second-best action. If ALL options are visited, it allows the revisit with the best Q-value.

#### What Q-Learning Produces (Output)

A `QLearningResult` object containing:
- `path` — the learned path from start to goal
- `cost` — Euclidean path cost
- `episodes_trained` — number of training episodes completed
- `success` — True if the learned policy reaches the goal
- `q_table` — the entire trained Q-table (can be reused)
- `training_rewards` — reward per episode (for plotting the learning curve)

---

### 3.3 Interactive Simulator Upgrade — `src/simulator.py`

**File size**: ~1000 lines (PyGame-based)

Phase 2 upgraded the interactive real-time simulator with **planner switching**:

| Key | Action |
|-----|--------|
| **P** | Cycle between A*, APF, and Q-Learning planners |
| SPACE | Pause / Resume |
| RIGHT | Step forward (when paused) |
| UP/DOWN | Speed up / slow down |
| R | Reset simulation |
| G | Toggle ghost paths |
| T | Toggle trails |
| L-Click | Place/remove wall |
| M-Click | Spawn dynamic obstacle |
| R-Click | Remove/drag obstacle |
| Q/ESC | Quit |

**How each planner works in the simulator:**

- **A* mode**: When a robot's next step is blocked, it runs a fresh A* from its current position to goal using the **live updated grid** (includes walls placed by user, dynamic obstacle positions, other robots). Fast, finds shortest path on current state.

- **APF mode**: Same re-planning trigger, but uses APF from current position. You can SEE it get stuck in traps in real time.

- **Q-Learning mode**: When selected, the simulator **pre-trains** a Q-table on the current grid, then uses the trained policy for navigation. If the grid changes significantly (user adds walls), it may need retraining.

**Each robot continuously checks:**
1. Is my next planned step blocked (by a wall, obstacle, or another robot)?
2. If yes OR I've been stuck for 3+ ticks → re-run the selected planner from current position

This creates **reactive, real-time replanning** behavior.

---

### 3.4 Main Demo Updates — `src/main.py`

Phase 2 added three new demo sections (steps 17-20 in the pipeline):

**Step 17: APF on U-Shaped Trap**
- Creates a `create_trap_environment()` — a 20×20 grid with a U-shaped wall around the goal
- Runs APF → demonstrates it getting **STUCK** (forces cancel out at the trap entrance)
- Saves `phase2_apf_trap.png`

**Step 18: Q-Learning on Same Trap**
- Trains Q-learning for 500 episodes on the same U-trap
- Runs the learned policy → demonstrates it **ESCAPING** the trap (learned through experience)
- Saves `phase2_qlearning_trap.png`

**Step 19: Three-Way Comparison**
- Creates a standard 20×20 grid with 3 dynamic obstacles
- Runs A*, APF, and Q-learning on the same grid
- Saves a side-by-side comparison plot: `phase2_planner_comparison.png`
- Prints a table:
  ```
  Algorithm            Success     Cost        Steps/Expanded
  ----------           -------     ----        ---------------
  A*                   ✓           XX.XX       Y nodes
  APF                  ✓/✗         XX.XX       Z steps
  Q-Learning           ✓/✗         XX.XX       500 episodes
  ```

---

## 4. All Output Files Generated

### Week 1 Outputs
| File | Description |
|------|-------------|
| `week1_demo.png` | A* path on random grid |
| `week1_search_steps.png` | A* search visualization step-by-step |
| `week1_robot_walk.gif` | Animated robot walking the A* path |
| `week1_comparison.png` | A* vs Greedy BFS side-by-side |
| `week1_trap_u-shape.png` | A* on U-trap scenario |
| `week1_narrow_passage.png` | A* on narrow corridor scenario |

### Phase 1 Outputs
| File | Description |
|------|-------------|
| `phase1_environment.png` | 20×20 grid with 3 robots + 4 dynamic obstacles |
| `phase1_candidate_paths.png` | All candidate paths per robot (color-coded) |
| `phase1_selection_comparison.png` | Greedy vs Random vs Brute-Force selectors |
| `phase1_bruteforce_paths.png` | Best classical paths overlay |
| `phase1_simulation.gif` | Animated 25-tick multi-robot simulation |

### Phase 2 Outputs (NEW)
| File | Description |
|------|-------------|
| `phase2_apf_trap.png` | **APF attempting U-trap — gets STUCK** |
| `phase2_qlearning_trap.png` | **Q-Learning solving U-trap — escapes** |
| `phase2_planner_comparison.png` | **Side-by-side A* vs APF vs Q-Learning** |

All saved in `experiments/results/`.

---

## 5. Concepts You Need to Understand (for Presentation/Viva)

### 5.1 Local vs Global Planning

| | Global (A*) | Local (APF) | Learning-Based (Q-Learning) |
|--|---|---|---|
| **Plans** | Entire path upfront | One step at a time | Learns from experience |
| **Sees** | Whole map (frozen snapshot) | Only nearby obstacles | Trains on full map, then follows policy |
| **Handles moving obstacles** | ❌ No | ✅ Yes, naturally | ⚠ If retrained on new map |
| **Gets stuck in traps** | ❌ Never (if path exists) | ✅ Yes — local minima | ❌ Can learn to escape |
| **Speed** | Fast (one-shot) | Very fast (per step) | Slow (needs training) |

### 5.2 The Local Minimum Problem

This is the **central problem** of Phase 2 and the motivation for quantum Q-learning:

```
        Robot → ← trapped here, vibrating
               ██████
               █    █
               █ 🎯 █   ← Goal is inside the U
               █    █
               ██████
```

**Why APF gets stuck**: The goal pulls the robot IN (attractive force), but the walls push it OUT (repulsive force). These forces **cancel out** at the entrance. The robot oscillates back and forth — a **local minimum** where no neighbor has a lower potential than the current position.

**Why Q-learning can escape**: After hundreds of practice runs, it learns through trial-and-error that going AROUND the U-shape leads to the goal. The reward signal (+100 for reaching goal) eventually propagates back through the Q-table, teaching the robot the detour.

**Why quantum Q-learning will be better** (Phase 3): Standard Q-learning uses a dumb coin flip (ε-greedy) to decide when to explore vs exploit. Quantum Q-learning replaces this with rotation gates on quantum probability amplitudes (α, β), which **automatically adjust** the explore/exploit balance based on rewards. This makes it learn ~80% faster.

### 5.3 The ε-Greedy Problem

Standard Q-learning's exploration strategy is crude:
```
Each step:
    Roll dice → random number between 0 and 1
    If < ε → move RANDOMLY (explore)
    If ≥ ε → move BEST KNOWN (exploit)
```

**Problems:**
- Early in training (ε=1.0): robot moves **100% randomly** — very wasteful
- Late in training (ε=0.05): robot **almost never** explores — misses better paths
- The transition is preset by decay rate — no intelligence about WHEN to explore
- Same ε for ALL states — even states the robot has never visited get low exploration

**Quantum Q-learning's solution** (Phase 3):
- Each state-action pair stores (α, β) where α² + β² = 1
- Good reward → **rotate toward exploit** (increase β²)
- Bad reward → **rotate toward explore** (increase α²)
- Each state-action pair adjusts **independently** — smart, adaptive exploration

### 5.4 Priority-Based Multi-Robot Coordination

How multiple robots avoid crashing into each other (from the simulator):
1. Robots are assigned IDs: Robot 0, Robot 1, Robot 2, ...
2. **Lower ID = Higher priority**
3. Each tick, robots move **in ID order**
4. Robot 0 moves first (no constraints)
5. Robot 1 checks: "Is my next cell occupied by Robot 0?" If yes → replan around it
6. Robot 2 checks against both Robot 0 AND Robot 1
7. This prevents collisions without complex joint planning

**Limitation**: Lower-priority robots get worse paths because they must dodge everyone else. QAOA (Phase 4) will solve this by optimizing paths for ALL robots simultaneously.

---

## 6. Current Project Architecture

```
Quantum-for-Multi-Robot-Path-planning/
│
├── src/
│   ├── grid.py            → Environment model (static + dynamic obstacles)
│   ├── a_star.py          → A* global shortest-path planner
│   ├── apf.py             → [PHASE 2] Artificial Potential Field local planner
│   ├── q_learning.py      → [PHASE 2] Tabular Q-learning RL baseline
│   ├── multi_robot.py     → Multi-robot candidate paths + selection
│   ├── dynamic_env.py     → Batch tick-by-tick simulation engine
│   ├── simulator.py       → [PHASE 2 UPGRADED] PyGame interactive simulator
│   ├── visualize.py       → All plots, charts, and GIF generation
│   └── main.py            → [PHASE 2 EXTENDED] One-click demo orchestrator
│
├── docs/
│   ├── Proposal_QAOA_Multi_Robot_Path_Planning.md
│   ├── Week_1_Summary.md
│   ├── Why_Each_Algorithm.md
│   └── Full_Phase_2_Explanation.md  ← THIS FILE
│
├── experiments/results/    → All generated images and GIFs
├── 1.pdf, 2.pdf, 3.pdf    → Reference papers
├── requirements.txt        → numpy, matplotlib, pygame
└── Project_Instructions_and_Plan.md
```

### Module Dependency Flow

```
grid.py ──────→ a_star.py ──→ multi_robot.py ──→ dynamic_env.py
   │                │               │                    │
   │                ↓               │                    │
   ├────────→ apf.py (Phase 2)     │                    │
   │                               │                    │
   ├────────→ q_learning.py (Ph2)  │                    │
   │                               │                    │
   └────────→ simulator.py ←──────┘                    │
                    ↑                                    │
                    └────────────────────────────────────┘
                    
visualize.py ← called by main.py for all plotting
main.py ← orchestrates everything in sequence
```

---

## 7. How to Run Everything

```powershell
# From project root:

# Run the complete pipeline (Week 1 + Phase 1 + Phase 2)
python src/main.py

# Run the interactive simulator (with planner switching)
python src/simulator.py

# With custom options
python src/simulator.py --robots 5 --size 25 --dynamic 8
```

When running `main.py`, you'll see output like:
```
============================================================
  QUANTUM MULTI-ROBOT PATH PLANNING  –  Week 1 Demo
============================================================
  ... Week 1 outputs ...

============================================================
  PHASE 1 — Multi-Robot Dynamic Environment
============================================================
  ... Phase 1 outputs ...

============================================================
  PHASE 2 — APF & Q-Learning Local Planners
============================================================
[APF]  Running APF on U-shaped trap environment...
[APF]  APFResult(success=False, STUCK, steps=XX)

[QL]   Training Q-learning on U-shaped trap (500 episodes)...
[QL]   QLearningResult(success=True, cost=XX.XX, ...)

[compare]  Running all 3 planners on a standard grid...
  A*                   ✓          XX.XX       Y nodes
  APF                  ✓/✗        XX.XX       Z steps
  Q-Learning           ✓/✗        XX.XX       500 ep

============================================================
  PHASE 2 COMPLETE — ALL OUTPUTS SAVED
============================================================
```

---

## 8. What's Coming Next

### Phase 3: Quantum Q-Learning + Hybrid Planner

**New files to be created:**

| File | Purpose |
|------|---------|
| `quantum_q_learning.py` | Replaces Q-table scalars with quantum probability amplitudes (α, β) and rotation gates |
| `hybrid_planner.py` | Combines A* (global) + APF (local) + quantum Q-learning (escape) |

**The quantum concept:**
```
Standard Q-learning:    Q(s,a) = single number (e.g., 0.7)
                        Explore/exploit = coin flip (ε-greedy)

Quantum Q-learning:     Q(s,a) = two numbers (α, β) where α² + β² = 1
                        Explore/exploit = rotation gate:
                        
                        [α']   [cos θ  -sin θ] [α]
                        [β'] = [sin θ   cos θ] [β]
                        
                        θ depends on reward → auto-adjusts balance
```

**Expected improvement**: ~80% faster convergence compared to standard Q-learning.

### Phase 4: QAOA for Multi-Robot Path Selection

**New file:** `qaoa_optimizer.py`

Instead of brute-force searching all K^N path combinations (exponential), use the Quantum Approximate Optimization Algorithm to find the best combination in polynomial time.

**Tech stack addition:** `qiskit`, `qiskit-optimization`

**The QUBO formulation** (already designed in the proposal):
- Variables: `x_{i,k} = 1` if robot i takes path k
- Minimize: total path cost + λ × total inter-robot conflicts
- Constraint: each robot picks exactly one path
- Solve with QAOA circuit on 12 qubits (3 robots × 4 candidates)

---

## 9. Summary — Phase 2 in One Page

**What we built:**
- ✅ APF planner (`apf.py`) — local, reactive, step-by-step navigation using force fields
- ✅ Q-learning planner (`q_learning.py`) — RL baseline that learns by trial-and-error
- ✅ Simulator upgrade — press **P** to switch between A*, APF, Q-Learning in real-time
- ✅ Three demo scenarios in `main.py` — APF trap, Q-learning trap, 3-way comparison

**What we proved:**
- A* is great for static environments but can't react to changes
- APF reacts to changes but **gets stuck in local minima** (U-traps)
- Q-learning can escape traps but is **slow** due to ε-greedy exploration
- → **This sets up the need for quantum Q-learning** (Phase 3)

**The chain of logic:**
```
A* fails at dynamics → Use APF → APF gets stuck → Use Q-learning → 
Q-learning is slow → Use QUANTUM Q-learning (YOUR CONTRIBUTION)
```

This is the story your entire project tells. Phase 2 provides the crucial middle chapters — showing why simpler methods aren't enough and why quantum enhancement is needed.

---

*Document version: 1.0 — Phase 2 Complete*
*Last updated: March 2026*
*Branch: `kalyani`*
