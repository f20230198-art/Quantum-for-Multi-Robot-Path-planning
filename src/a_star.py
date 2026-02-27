"""
a_star.py  –  A* Pathfinding Algorithm
=======================================

WHAT THIS FILE DOES (plain English):
    Given a Grid (from grid.py), a start cell, and a goal cell,
    this file finds the SHORTEST path between them using the A*
    algorithm — the gold-standard baseline planner referenced in
    all three of our project papers.

HOW A* WORKS (5-sentence version):
    1.  Keep a "to-do list" (priority queue) of cells to explore.
    2.  Each cell has two scores:
            g = actual cost from start to this cell
            f = g + h, where h is a *heuristic* (estimate of remaining
                distance to the goal — we use Euclidean distance).
    3.  Always expand the cell with the lowest f first.
    4.  When we pop the goal cell off the queue, we're done —
        trace backwards through the parent links to recover the path.
    5.  If the queue empties before we reach the goal, no path exists.

WHY IT'S THE BASELINE:
    A* finds the optimal (shortest) path on a static grid, but it
    can't handle moving obstacles or adapt in real-time.  Our quantum
    planner (Week 5+) will beat it on dynamic/complex maps.
"""

import heapq
import math
from typing import List, Optional, Tuple, Dict

# We import Grid only for type hints; no circular dependency issue.
from grid import Grid, Position


# ------------------------------------------------------------------
# Heuristic functions
# ------------------------------------------------------------------

def euclidean(a: Position, b: Position) -> float:
    """Straight-line distance — admissible heuristic for 8-connected grid."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def manhattan(a: Position, b: Position) -> float:
    """Taxi-cab distance — admissible heuristic for 4-connected grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------------
# A* search
# ------------------------------------------------------------------

class AStarResult:
    """Container for the result of an A* search."""

    def __init__(
        self,
        path: Optional[List[Position]],
        cost: float,
        nodes_expanded: int,
        came_from: Dict[Position, Optional[Position]],
        expansion_order: Optional[List[Position]] = None,
        g_scores: Optional[Dict[Position, float]] = None,
        f_scores: Optional[Dict[Position, float]] = None,
    ):
        self.path = path                    # list of (row,col) from start→goal, or None
        self.cost = cost                    # total path cost
        self.nodes_expanded = nodes_expanded  # how many cells we looked at
        self.came_from = came_from          # parent map (useful for visualization)
        self.expansion_order = expansion_order or []  # cells in the order A* explored them
        self.g_scores = g_scores or {}      # actual cost to each explored cell
        self.f_scores = f_scores or {}      # estimated total cost through each cell
        self.success = path is not None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"AStarResult(success=True, cost={self.cost:.2f}, "
                f"path_length={len(self.path)}, expanded={self.nodes_expanded})"
            )
        return f"AStarResult(success=False, expanded={self.nodes_expanded})"


def a_star(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    heuristic=euclidean,
    eight_connected: bool = True,
) -> AStarResult:
    """
    Run A* on *grid* from *start* to *goal*.

    Parameters
    ----------
    grid : Grid
        The environment to search.
    start, goal : Position or None
        If None, uses grid.start / grid.goal.
    heuristic : callable(Position, Position) -> float
        Heuristic function.  Default: Euclidean distance.
    eight_connected : bool
        Allow diagonal movement?

    Returns
    -------
    AStarResult
        Contains the path (list of positions), total cost, and stats.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    # ---- data structures ----
    # open_set is a min-heap of (f_score, counter, position)
    # counter breaks ties so heapq never compares Position tuples
    open_set: list = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[Position, Optional[Position]] = {start: None}
    g_score: Dict[Position, float] = {start: 0.0}
    f_score: Dict[Position, float] = {start: heuristic(start, goal)}

    closed_set = set()
    nodes_expanded = 0
    expansion_order: List[Position] = []   # record the order cells were expanded

    while open_set:
        # Pop cell with lowest f
        current_f, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_expanded += 1
        expansion_order.append(current)

        # ---- goal reached ----
        if current == goal:
            path = _reconstruct_path(came_from, current)
            return AStarResult(
                path=path,
                cost=g_score[current],
                nodes_expanded=nodes_expanded,
                came_from=came_from,
                expansion_order=expansion_order,
                g_scores=dict(g_score),
                f_scores=dict(f_score),
            )

        # ---- expand neighbours ----
        for neighbor in grid.get_neighbors(current, eight_connected=eight_connected):
            if neighbor in closed_set:
                continue

            # Movement cost: √2 for diagonal, 1 for cardinal
            dr = abs(neighbor[0] - current[0])
            dc = abs(neighbor[1] - current[1])
            step_cost = math.sqrt(2) if (dr + dc == 2) else 1.0

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                # This path to neighbor is better
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    # ---- no path found ----
    return AStarResult(
        path=None,
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        came_from=came_from,
        expansion_order=expansion_order,
        g_scores=dict(g_score),
        f_scores=dict(f_score),
    )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _reconstruct_path(
    came_from: Dict[Position, Optional[Position]], current: Position
) -> List[Position]:
    """Walk the parent links backwards to build the path list."""
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
