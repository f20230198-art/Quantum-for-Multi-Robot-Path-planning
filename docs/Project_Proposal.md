# Quantum-Inspired Multi-Robot Path Planning
## Project Proposal

---

## 1. What Is This Project?

This project builds a **simulation system** where multiple robots navigate a 2D grid environment filled with obstacles — some static (walls) and some dynamic (moving). The robots must find efficient, collision-free paths from their starting positions to their goals.

The twist: we use **quantum-inspired algorithms** to make the robots smarter at escaping traps and coordinating with each other. Everything runs in Python on a regular laptop — no actual quantum hardware needed.

---

## 2. Why Does This Problem Matter?

Multi-robot path planning shows up everywhere in the real world:

- **Warehouses** — Amazon uses 750,000+ robots moving around fulfillment centers
- **Self-driving cars** — Vehicles at intersections need to coordinate
- **Search and rescue** — Drone swarms covering disaster areas
- **Factories** — Robots on assembly lines that share floor space

The challenge: as you add more robots, the number of possible path combinations **explodes exponentially**. With 3 robots and 4 path options each, there are 64 combinations. With 20 robots? Over a trillion. No computer can check them all.

We need algorithms that are both **fast** and **smart about avoiding collisions**.

---

## 3. The Problem with Existing Solutions

We studied five traditional approaches. Each one has a fatal flaw:

| Method | What It Does | The Problem |
|--------|-------------|-------------|
| **A\*** | Finds the absolute shortest path | Only works for ONE robot. Cannot handle moving obstacles. |
| **APF** (Artificial Potential Field) | Reacts to obstacles in real time | Gets stuck in U-shaped traps (local minima) |
| **Q-Learning** | Robot learns by trial and error over many episodes | Slow to converge. Uses a crude "coin flip" to decide when to explore vs exploit |
| **Greedy Selection** | Each robot just picks its shortest path | Completely ignores collisions between robots |
| **Brute-Force** | Checks every possible combination | Works perfectly but takes exponential time — impossible beyond ~5 robots |

The core issue:

```
Fast methods are blind to collisions.
Optimal methods are too slow to scale.
Local planners get stuck in traps.
```

---

## 4. Our Solution — A Hybrid Approach

We don't use just one algorithm. We combine **five** into a layered system where each one covers the weakness of the one before it:

```
A*  finds the big-picture route (global plan)
 |
 v
APF  follows the route step-by-step, dodging moving obstacles (local plan)
 |
 v
Quantum Q-Learning  kicks in ONLY when APF gets stuck (escape plan)
 |
 v
A*  re-plans from the new position after escaping
 |
 v
QAOA  selects the best combination of paths across ALL robots (coordination)
```

### Why each layer exists:

1. **A\* (Layer 1 — Global)**: Computes the shortest path considering the full map. This is the "GPS route."

2. **APF (Layer 2 — Local)**: Follows the A\* route but adjusts in real time. If a moving obstacle blocks the path, APF steers around it using force calculations — the goal pulls the robot forward, obstacles push it away.

3. **Quantum Q-Learning (Layer 3 — Escape)**: When APF's forces cancel out and the robot is stuck (detected by checking if recent positions are all clustered in a small area), the quantum policy takes over. It uses a learned strategy to break free, then hands control back to A\*.

4. **QAOA (Layer 4 — Coordination)**: For multiple robots, we generate several candidate paths per robot and use the Quantum Approximate Optimization Algorithm to pick the best combination that minimizes total cost AND collisions.

---

## 5. The Key Algorithms Explained Simply

### 5.1 A\* Pathfinding

Think of it like GPS navigation. The algorithm checks all possible next steps, scores each one (actual distance traveled + estimated distance remaining), and always picks the best. It guarantees the shortest path.

**Limitation**: It needs a complete map upfront and plans for one robot at a time.

### 5.2 Artificial Potential Field (APF)

Imagine the robot feels invisible forces:
- The **goal** acts like a magnet pulling the robot toward it
- Each **obstacle** acts like a force field pushing the robot away
- The robot moves in the direction of the combined force

**Limitation**: In U-shaped traps, the pull toward the goal and push from walls cancel out, leaving the robot frozen.

### 5.3 Standard Q-Learning

The robot learns by trial and error:
- Start at the beginning, try random actions
- Get a reward for reaching the goal (+100), penalty for hitting walls (-10) or taking steps (-1)
- After hundreds of episodes, build a "cheat sheet" (Q-table) of what to do in every situation
- Use an epsilon-greedy rule: most of the time follow the cheat sheet, occasionally try something random

**Limitation**: The coin-flip exploration strategy is crude. It explores the same amount everywhere, whether it needs to or not.

### 5.4 Quantum Q-Learning (Our Key Contribution)

Instead of storing a plain number for each action, we store a **rotation angle** (theta) on a quantum-inspired representation:

```
Standard Q-Learning:    Q[state, action] = 0.73  (just a number)
Quantum Q-Learning:     theta[state, action] = 1.2 radians  (an angle)
```

The "quality" of an action is computed as:

```
P(action) = sin^2(theta / 2)
```

This is the probability of measuring |1> when a qubit is rotated by theta using an Ry gate.

**Why this is better:**

| Feature | Standard Q-Learning | Quantum Q-Learning |
|---------|--------------------|--------------------|
| Value range | Unbounded (can go to +/- infinity) | Bounded [0, 1] (never diverges) |
| Update mechanism | Add/subtract numbers | Rotate angle on Bloch sphere |
| Exploration | Crude coin flip (epsilon-greedy) | Natural — small angles give equal probabilities |
| Convergence | Can be slow or unstable | Smooth, self-regulating |

The update rule uses a **Grover-inspired self-regulating rotation**:
- Positive TD error (action was better than expected) → rotate toward pi (increase probability)
- Negative TD error (action was worse than expected) → rotate toward 0 (decrease probability)
- **Damping**: as the angle gets close to its boundary, updates slow down automatically, preventing saturation

### 5.5 QAOA (Multi-Robot Coordination)

For multiple robots, we formulate path selection as a mathematical optimization problem (QUBO):

```
Variables:  x[i,k] = 1 if robot i takes path k, else 0
Minimize:   total_path_cost + penalty * total_collisions
Constraint: each robot picks exactly one path
```

QAOA uses quantum superposition to explore all K^N combinations simultaneously instead of checking them one by one. For 3 robots with 4 paths each, it uses 12 qubits to explore 64 combinations.

---

## 6. System Architecture

The project is organized into modular Python files:

```
src/
  grid.py               -->  The 2D world (obstacles, start/goal positions)
  a_star.py             -->  A* shortest path algorithm
  apf.py                -->  Artificial Potential Field local planner
  q_learning.py         -->  Standard Q-Learning (baseline for comparison)
  quantum_q_learning.py -->  Quantum-inspired Q-Learning (our method)
  hybrid_planner.py     -->  Three-layer hybrid: A* + APF + Quantum escape
  qaoa_optimizer.py     -->  QAOA multi-robot path optimizer (Phase 4)
  multi_robot.py        -->  Multi-robot candidate paths + conflict analysis
  dynamic_env.py        -->  Time-stepped simulation with moving obstacles
  visualize.py          -->  All plotting and graph generation
  simulator.py          -->  Interactive real-time simulator (PyGame)
  main.py               -->  One-click demo that runs everything
  utils.py              -->  Shared helper functions
```

Data flows through the system like this:

```
Grid Environment  -->  A* generates candidate paths per robot
                  -->  Conflict matrix computed between all paths
                  -->  QAOA (or classical baseline) selects best combination
                  -->  Simulation runs the selected paths with dynamic obstacles
                  -->  Visualizations generated for analysis
```

---

## 7. What We Compare

We run all algorithms on the same environments and measure:

| Metric | What It Tells Us |
|--------|-----------------|
| **Path Cost** | Total distance traveled (lower = more efficient) |
| **Success Rate** | Did the robot reach its goal? |
| **Computation Time** | How long did planning take? |
| **Path Length** | Number of cells visited (lower = more direct) |
| **Collision Count** | How many robot-robot or robot-obstacle collisions? |

We test across:
- **Different obstacle densities** (10% to 30%) with realistic varied-size obstacles
- **Trap environments** (U-shaped obstacles that cause APF to get stuck)
- **Multi-robot scenarios** (3-5 robots with crossing paths)
- **Dynamic environments** (moving obstacles that change the map over time)

---

## 8. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Core computation | NumPy |
| Visualization | Matplotlib |
| Interactive simulator | PyGame |
| Quantum simulation | Qiskit + Qiskit Aer (optional — falls back to NumPy) |
| QAOA optimization | Qiskit Optimization (Phase 4) |

---

## 9. Reference Papers

This project draws from three research papers:

1. **Paper 1** — QER-LPD3QN: Quantum-inspired deep reinforcement learning for path planning. Introduces qubit replay mechanisms.

2. **Paper 2** — CAA\*QPSO: Quantum Particle Swarm Optimization for 3D robot path planning. Background on quantum optimization concepts.

3. **Paper 3** — Fuzzy A\* + Quantum Q-Learning + APF: A hybrid planner for single robots. This is the closest to our approach — **we extend it to multiple robots**, which is the novel contribution.

---

## 10. Project Phases

| Phase | What Gets Built | Status |
|-------|----------------|--------|
| **Week 1** | Grid environment, A\* pathfinding, basic visualization | Done |
| **Phase 1** | Multi-robot support, candidate path generation, conflict analysis, classical selectors, dynamic environment, interactive simulator | Done |
| **Phase 2** | APF local planner, Q-Learning baseline, planner comparisons | Done |
| **Phase 3** | Quantum Q-Learning, Hybrid Planner (A\* + APF + Quantum), comprehensive performance comparisons across obstacle densities | Done |
| **Phase 4** | QAOA optimizer for multi-robot path selection (QUBO + numpy QAOA simulator + optional Qiskit) | Done |

---

## 11. Expected Outcomes

1. **Quantum Q-Learning escapes traps** that standard APF gets stuck in (demonstrated on U-shaped obstacle environments)

2. **Bounded, stable learning** — rotation angles stay in [0, pi], unlike classical Q-values which can diverge

3. **Hybrid planner combines strengths** — A\*'s global optimality + APF's real-time reaction + Quantum QL's trap escape

4. **QAOA scales better than brute-force** — polynomial time vs exponential time for multi-robot coordination

5. **Comprehensive comparison data** — bar charts and tables showing all 5 planners across 5 obstacle densities with realistic obstacle shapes

---

## 12. The 5-Minute Summary

"Our project solves multi-robot path planning in dynamic environments. Traditional A\* finds optimal paths but cannot handle moving obstacles. APF handles dynamics but gets stuck in local minima. We use quantum-inspired Q-learning — encoding action values as qubit rotation angles and updating them with Ry gates — to escape these traps faster than classical methods. We extend this from single robots to multiple robots using a hybrid three-layer planner and QAOA-based path coordination. Our experiments show the quantum-enhanced method escapes traps that standard A\*+APF cannot, with bounded and stable learning dynamics."
