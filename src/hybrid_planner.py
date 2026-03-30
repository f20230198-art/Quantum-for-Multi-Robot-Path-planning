"""
hybrid_planner.py  –  Hybrid Multi-Layer Path Planner
======================================================

WHAT THIS FILE DOES (plain English):
    Combines THREE planning strategies into a single decision loop:

        LAYER 1 — A* (Global Planner)
            Computes a full shortest-path from current position to goal.
            This is the "big picture" route.

        LAYER 2 — APF (Local Reactive Planner)
            Follows the A* route step-by-step, but also reacts to
            dynamic obstacles by computing local potential fields.
            If a dynamic obstacle blocks the planned next cell, APF
            steers around it without needing a full replan.

        LAYER 3 — Quantum Q-Learning (Escape Planner)
            Activated ONLY when APF detects it is stuck in a local
            minimum (e.g., U-shaped trap).  The quantum policy,
            trained offline, can escape traps because its rotation-
            gate updates explored diverse strategies during training.
            After escaping, A* re-plans from the new position.

WHY A HYBRID:
    • A* alone can't react to moving obstacles (static snapshot).
    • APF alone gets stuck in local minima (U-traps).
    • Q-Learning alone is slow to converge on large grids.
    Combining them gives: optimal global route + reactive local
    navigation + intelligent escape from traps.

REFERENCE:
    This architecture mirrors the hybrid approach in Paper 3 (3.pdf):
    "global planner + local reactive layer + learned escape policy."
"""

import math
from typing import List, Optional, Tuple, Dict
from grid import Grid, Position
from a_star import a_star, euclidean
from apf import (
    total_potential, APFResult,
)
from quantum_q_learning import (
    train_quantum_q_learning, quantum_q_learning_plan,
    _angle_to_prob, _is_valid_move, ACTIONS,
    QuantumQLearningResult,
)
from utils import path_cost as _path_cost

import numpy as np


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

class HybridResult:
    """Container for hybrid planner output."""

    def __init__(
        self,
        path: Optional[List[Position]],
        cost: float,
        success: bool,
        total_steps: int,
        astar_replans: int,
        apf_steps: int,
        quantum_escapes: int,
        stuck_detected: int,
    ):
        self.path = path
        self.cost = cost
        self.success = success
        self.total_steps = total_steps
        self.astar_replans = astar_replans
        self.apf_steps = apf_steps
        self.quantum_escapes = quantum_escapes
        self.stuck_detected = stuck_detected

    def __repr__(self) -> str:
        if self.success:
            return (
                f"HybridResult(success=True, cost={self.cost:.2f}, "
                f"steps={self.total_steps}, "
                f"A*_replans={self.astar_replans}, "
                f"APF_steps={self.apf_steps}, "
                f"Q_escapes={self.quantum_escapes})"
            )
        return (
            f"HybridResult(success=False, "
            f"steps={self.total_steps}, stuck={self.stuck_detected})"
        )


# ------------------------------------------------------------------
# Hybrid planner
# ------------------------------------------------------------------

def hybrid_plan(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    max_steps: int = 500,
    # APF parameters
    k_att: float = 1.0,
    k_rep: float = 100.0,
    influence_radius: float = 3.0,
    # Stuck detection
    stuck_window: int = 20,
    stuck_radius: float = 3.0,
    # Quantum escape
    escape_steps: int = 20,
    q_episodes: int = 500,
    q_seed: Optional[int] = None,
    # Pre-trained angles (skip training if provided)
    q_angles: Optional[np.ndarray] = None,
    # Dynamic obstacle stepping
    step_dynamic: bool = False,
) -> HybridResult:
    """
    Run the hybrid planner: A* (global) + APF (local) + Quantum QL (escape).

    Algorithm:
        1. Run A* to get a global path from start to goal.
        2. At each step, use APF to choose the next move:
           - APF follows the global path as a waypoint sequence
           - APF also reacts to obstacles near the current position
        3. If stuck is detected (recent positions all within a small
           radius), switch to quantum Q-learning for `escape_steps`:
           - Follow the learned quantum policy to break out of the
             local minimum
           - Then re-run A* from the new position
        4. Repeat until goal reached or max_steps exceeded.

    Parameters
    ----------
    grid : Grid
    start, goal : Position or None
    max_steps : int
    k_att, k_rep, influence_radius : float
        APF tuning parameters.
    stuck_window : int
        How many recent steps to check for stuck detection.
    stuck_radius : float
        If all recent positions are within this radius of their
        centroid, the robot is declared stuck.
    escape_steps : int
        How many quantum QL steps to take when escaping.
    q_episodes : int
        Training episodes for quantum QL (if not pre-trained).
    q_seed : int or None
    q_angles : np.ndarray or None
        Pre-trained rotation angles.  If None, trains on the fly.

    Returns
    -------
    HybridResult
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    # ── Pre-train quantum Q-learning if needed ──
    if q_angles is None:
        q_angles, _ = train_quantum_q_learning(
            grid, start=start, goal=goal,
            episodes=q_episodes, seed=q_seed,
        )

    # ── Get initial A* global path ──
    astar_result = a_star(grid, start=start, goal=goal, eight_connected=True)
    if not astar_result.success:
        return HybridResult(
            path=None, cost=0.0, success=False, total_steps=0,
            astar_replans=0, apf_steps=0, quantum_escapes=0,
            stuck_detected=0,
        )

    global_path = list(astar_result.path)
    waypoint_idx = 1  # index into global_path (0 is start)

    # ── State tracking ──
    current = start
    full_path = [start]
    visited_counts: Dict[Position, int] = {start: 1}

    stats = {
        "astar_replans": 1,  # initial plan counts
        "apf_steps": 0,
        "quantum_escapes": 0,
        "stuck_detected": 0,
    }

    mode = "APF"  # current active layer
    escape_remaining = 0

    for step in range(max_steps):
        # ── Advance dynamic obstacles (if enabled) ──
        if step_dynamic:
            grid.step_dynamic_obstacles()

        # ── Goal check ──
        if current == goal:
            cost = _path_cost(full_path)
            return HybridResult(
                path=full_path, cost=cost, success=True,
                total_steps=step, **stats,
            )

        # ── Stuck detection (only in APF mode) ──
        if mode == "APF" and len(full_path) >= stuck_window:
            recent = full_path[-stuck_window:]
            cr = sum(p[0] for p in recent) / stuck_window
            cc = sum(p[1] for p in recent) / stuck_window
            all_close = all(
                math.sqrt((p[0] - cr) ** 2 + (p[1] - cc) ** 2)
                <= stuck_radius
                for p in recent
            )
            if all_close:
                stats["stuck_detected"] += 1
                mode = "QUANTUM"
                escape_remaining = escape_steps

        # ── QUANTUM ESCAPE MODE ──
        if mode == "QUANTUM":
            if escape_remaining <= 0:
                # Done escaping — re-plan with A*
                mode = "APF"
                stats["quantum_escapes"] += 1
                re = a_star(grid, start=current, goal=goal,
                            eight_connected=True)
                if re.success:
                    global_path = list(re.path)
                    waypoint_idx = 1
                    stats["astar_replans"] += 1
                continue

            # Follow quantum policy for one step
            probs = np.array([
                _angle_to_prob(q_angles[current[0], current[1], a])
                for a in range(len(ACTIONS))
            ])
            # Mask invalid
            for i, (dr, dc) in enumerate(ACTIONS):
                npos = (current[0] + dr, current[1] + dc)
                if not _is_valid_move(grid, npos):
                    probs[i] = -1.0

            if np.all(probs < 0):
                # Completely boxed in
                escape_remaining = 0
                continue

            # Prefer unvisited cells
            best_action = None
            best_prob = -2.0
            for i in range(len(ACTIONS)):
                if probs[i] < 0:
                    continue
                dr, dc = ACTIONS[i]
                npos = (current[0] + dr, current[1] + dc)
                bonus = 0.1 if npos not in visited_counts else 0.0
                adjusted = probs[i] + bonus
                if adjusted > best_prob:
                    best_prob = adjusted
                    best_action = i

            if best_action is not None:
                dr, dc = ACTIONS[best_action]
                current = (current[0] + dr, current[1] + dc)
                full_path.append(current)
                visited_counts[current] = visited_counts.get(current, 0) + 1

            escape_remaining -= 1
            continue

        # ── APF MODE (normal operation) ──
        # Determine local APF waypoint from global path
        if waypoint_idx < len(global_path):
            local_goal = global_path[min(waypoint_idx + 3, len(global_path) - 1)]
        else:
            local_goal = goal

        # Gather nearby obstacles
        obstacles: List[Position] = []
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    if (abs(r - current[0]) <= influence_radius + 1 and
                            abs(c - current[1]) <= influence_radius + 1):
                        obstacles.append((r, c))
        for dpos in grid.get_dynamic_obstacle_positions():
            obstacles.append(dpos)

        # Evaluate neighbors using APF
        neighbors = grid.get_neighbors(current, eight_connected=True)
        if not neighbors:
            break  # boxed in

        best_pos = None
        best_potential = float("inf")
        for nb in neighbors:
            p = total_potential(nb, local_goal, obstacles,
                                k_att, k_rep, influence_radius)
            # Penalise revisits
            p += visited_counts.get(nb, 0) * 0.5
            if p < best_potential:
                best_potential = p
                best_pos = nb

        current = best_pos
        full_path.append(current)
        visited_counts[current] = visited_counts.get(current, 0) + 1
        stats["apf_steps"] += 1

        # Advance waypoint if we're close to the current one
        if waypoint_idx < len(global_path):
            wp = global_path[waypoint_idx]
            dist_to_wp = math.sqrt(
                (current[0] - wp[0]) ** 2 + (current[1] - wp[1]) ** 2
            )
            if dist_to_wp <= 1.5:
                waypoint_idx += 1

    # Ran out of steps
    cost = _path_cost(full_path)
    return HybridResult(
        path=full_path, cost=cost, success=(current == goal),
        total_steps=max_steps, **stats,
    )
