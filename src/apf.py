"""
apf.py  –  Artificial Potential Field Local Planner
=====================================================

WHAT THIS FILE DOES (plain English):
    Instead of computing a full path through the grid upfront (like A*),
    the APF planner makes ONE-STEP-AT-A-TIME decisions by calculating
    "forces" on the robot:

        • ATTRACTIVE FORCE  — pulls the robot toward the goal.
          The further away the goal, the stronger the pull.

        • REPULSIVE FORCE   — pushes the robot away from nearby obstacles
          (both static and dynamic).  Only kicks in when an obstacle is
          within a certain radius (influence_radius).

    Each step, the robot looks at all its neighbours, computes the total
    force (attractive + repulsive) at each, and moves to the neighbour
    with the LOWEST potential (strongest "pull" toward goal, weakest
    "push" from obstacles).

WHY APF IS USEFUL:
    Because it reacts to the LOCAL environment each step, it naturally
    dodges dynamic (moving) obstacles without needing to re-plan the
    whole path.

WHY APF FAILS (local minima):
    Sometimes the attractive and repulsive forces cancel out perfectly
    (e.g. the goal is behind a U-shaped wall — the wall pushes the
    robot away just as hard as the goal pulls it forward).  The robot
    gets STUCK in a "local minimum."  This is the EXACT problem that
    quantum Q-learning will solve in Week 3.

REFERENCE:
    Based on Khatib (1986) potential field method, adapted for our
    discrete 2-D grid.  Paper 3 (3.pdf) uses APF as one layer of
    their hybrid planner.
"""

import math
from typing import List, Optional, Tuple, Dict
from grid import Grid, Position


# ------------------------------------------------------------------
# Potential field functions
# ------------------------------------------------------------------

def attractive_potential(pos: Position, goal: Position, k_att: float = 1.0) -> float:
    """
    Parabolic attractive potential — pulls toward the goal.

    U_att(pos) = 0.5 * k_att * distance(pos, goal)^2

    The further you are from the goal, the stronger the pull.
    """
    dx = pos[0] - goal[0]
    dy = pos[1] - goal[1]
    return 0.5 * k_att * (dx * dx + dy * dy)


def repulsive_potential(
    pos: Position,
    obstacles: List[Position],
    k_rep: float = 100.0,
    influence_radius: float = 3.0,
) -> float:
    """
    Repulsive potential — pushes away from nearby obstacles.

    For each obstacle within influence_radius:
        U_rep = 0.5 * k_rep * (1/d - 1/radius)^2

    If no obstacles are nearby, U_rep = 0.

    Parameters
    ----------
    pos : Position
        Current cell being evaluated.
    obstacles : list of Position
        All obstacle cells (static + dynamic).
    k_rep : float
        Repulsion strength.  Higher = stronger push away.
    influence_radius : float
        Obstacles further than this have NO effect.
    """
    total = 0.0
    for obs in obstacles:
        dx = pos[0] - obs[0]
        dy = pos[1] - obs[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.01:
            # Robot is ON an obstacle — maximum repulsion
            total += 1e6
        elif d <= influence_radius:
            total += 0.5 * k_rep * ((1.0 / d) - (1.0 / influence_radius)) ** 2
    return total


def total_potential(
    pos: Position,
    goal: Position,
    obstacles: List[Position],
    k_att: float = 1.0,
    k_rep: float = 100.0,
    influence_radius: float = 3.0,
) -> float:
    """Sum of attractive and repulsive potentials at *pos*."""
    return (
        attractive_potential(pos, goal, k_att)
        + repulsive_potential(pos, obstacles, k_rep, influence_radius)
    )


# ------------------------------------------------------------------
# APF planner — one complete plan from start to goal
# ------------------------------------------------------------------

class APFResult:
    """Container for APF planning output (mirrors AStarResult style)."""

    def __init__(
        self,
        path: Optional[List[Position]],
        cost: float,
        steps_taken: int,
        success: bool,
        stuck: bool,
    ):
        self.path = path            # list of positions from start→goal, or partial path
        self.cost = cost            # Euclidean path cost
        self.steps_taken = steps_taken  # how many steps the robot took
        self.success = success      # True if goal was reached
        self.stuck = stuck          # True if robot got stuck in a local minimum

    def __repr__(self) -> str:
        if self.success:
            return (
                f"APFResult(success=True, cost={self.cost:.2f}, "
                f"steps={self.steps_taken})"
            )
        status = "STUCK" if self.stuck else "TIMEOUT"
        return f"APFResult(success=False, {status}, steps={self.steps_taken})"


def apf_plan(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    max_steps: int = 500,
    k_att: float = 1.0,
    k_rep: float = 100.0,
    influence_radius: float = 3.0,
    eight_connected: bool = True,
) -> APFResult:
    """
    Plan a path using the Artificial Potential Field method.

    At each step the robot moves to the neighbour with the lowest
    total potential.  If multiple neighbours tie, the one closest to
    the goal wins.

    Parameters
    ----------
    grid : Grid
        The environment (static + dynamic obstacles).
    start, goal : Position or None
        Defaults to grid.start / grid.goal.
    max_steps : int
        Stop after this many steps even if goal not reached.
    k_att : float
        Attractive gain.
    k_rep : float
        Repulsive gain.
    influence_radius : float
        Repulsive field range (cells).
    eight_connected : bool
        Allow diagonal movement?

    Returns
    -------
    APFResult
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    path = [start]
    current = start
    visited_counts: Dict[Position, int] = {start: 1}

    # Stuck detection: track recent positions
    stuck_window = 15       # look at last N positions
    stuck_radius = 2.0      # if all within this radius → stuck
    stuck_threshold = 12    # must stay this many steps to declare stuck

    for step in range(max_steps):
        # --- Goal check ---
        if current == goal:
            cost = _path_cost(path)
            return APFResult(
                path=path, cost=cost, steps_taken=step,
                success=True, stuck=False,
            )

        # --- Gather obstacles (static + dynamic) ---
        obstacles: List[Position] = []
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    # Only include nearby static obstacles (optimisation)
                    if abs(r - current[0]) <= influence_radius + 1 and \
                       abs(c - current[1]) <= influence_radius + 1:
                        obstacles.append((r, c))
        for dpos in grid.get_dynamic_obstacle_positions():
            obstacles.append(dpos)

        # --- Evaluate neighbours ---
        neighbors = grid.get_neighbors(current, eight_connected=eight_connected)
        if not neighbors:
            # Completely boxed in
            cost = _path_cost(path)
            return APFResult(
                path=path, cost=cost, steps_taken=step,
                success=False, stuck=True,
            )

        best_pos = None
        best_potential = float("inf")
        for nb in neighbors:
            p = total_potential(nb, goal, obstacles, k_att, k_rep, influence_radius)
            # Penalise revisited cells slightly to reduce oscillation
            revisit_pen = visited_counts.get(nb, 0) * 0.5
            p += revisit_pen
            if p < best_potential:
                best_potential = p
                best_pos = nb

        # Move
        current = best_pos
        path.append(current)
        visited_counts[current] = visited_counts.get(current, 0) + 1

        # --- Stuck detection ---
        if len(path) >= stuck_window:
            recent = path[-stuck_window:]
            centroid_r = sum(p[0] for p in recent) / stuck_window
            centroid_c = sum(p[1] for p in recent) / stuck_window
            all_close = all(
                math.sqrt((p[0] - centroid_r) ** 2 + (p[1] - centroid_c) ** 2)
                <= stuck_radius
                for p in recent
            )
            if all_close and len(path) > stuck_threshold:
                cost = _path_cost(path)
                return APFResult(
                    path=path, cost=cost, steps_taken=step,
                    success=False, stuck=True,
                )

    # Ran out of steps
    cost = _path_cost(path)
    return APFResult(
        path=path, cost=cost, steps_taken=max_steps,
        success=False, stuck=False,
    )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _path_cost(path: List[Position]) -> float:
    """Euclidean path cost."""
    cost = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i][0] - path[i + 1][0])
        dc = abs(path[i][1] - path[i + 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost
