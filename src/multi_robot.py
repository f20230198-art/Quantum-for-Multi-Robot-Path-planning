"""
multi_robot.py  –  Multi-Robot Path Generation & Conflict Analysis
===================================================================

WHAT THIS FILE DOES (plain English):
    1.  For each robot, generates K candidate paths using A* with
        different cost tweaks (penalty zones, random obstacle
        injections).  This gives QAOA a menu of options to pick from.

    2.  Computes a CONFLICT MATRIX — for every pair of candidate paths
        (across different robots), counts how many timesteps they'd
        collide.  This matrix becomes the input to the QAOA optimizer.

    3.  Also provides classical baselines:
        • Greedy selection (pick shortest path per robot, ignore conflicts)
        • Random selection
        • Brute-force (try all combinations, pick the best)

WHY MULTIPLE CANDIDATE PATHS?
    If each robot only has 1 path (the A* optimal), there's nothing to
    optimize — you're stuck with whatever collisions happen.  By
    generating 4–5 different routes per robot (slightly longer but
    going through different areas), QAOA can choose a combination that
    minimizes total cost AND avoids inter-robot conflicts.

    Example for 3 robots × 4 paths:
        Robot 0 → paths [A, B, C, D]
        Robot 1 → paths [A, B, C, D]
        Robot 2 → paths [A, B, C, D]
        Total combinations = 4 × 4 × 4 = 64
        QAOA searches this space using quantum superposition.
"""

import math
import copy
import itertools
import numpy as np
from typing import List, Tuple, Optional, Dict
from utils import path_cost as _compute_path_cost

from grid import Grid, Position
from a_star import a_star, AStarResult, euclidean


# ------------------------------------------------------------------
# Candidate path generation
# ------------------------------------------------------------------

def generate_candidate_paths(
    grid: Grid,
    start: Position,
    goal: Position,
    num_candidates: int = 4,
    seed: int = 0,
) -> List[Dict]:
    """
    Generate K diverse candidate paths from *start* to *goal* using
    A* with different penalty zones.

    Strategy:
        Path 0 → vanilla A* (the optimal baseline)
        Path 1..K-1 → A* on a modified grid where random "soft penalty"
            zones inflate the cost of certain areas, forcing A* to
            find alternative routes.

    Parameters
    ----------
    grid : Grid
        The base environment.
    start, goal : Position
        Robot's start and goal.
    num_candidates : int
        How many paths to generate (default 4).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict: {"path": list[Position], "cost": float, "variant": str}
        Paths that failed (no route found) are excluded.
    """
    rng = np.random.RandomState(seed)
    candidates = []

    # --- Path 0: vanilla A* (optimal) ---
    result = a_star(grid, start=start, goal=goal, eight_connected=True)
    if result.success:
        candidates.append({
            "path": result.path,
            "cost": result.cost,
            "variant": "optimal",
            "nodes_expanded": result.nodes_expanded,
        })

    # --- Paths 1..K-1: A* with penalty zones ---
    for k in range(1, num_candidates):
        modified = _create_penalty_grid(grid, rng, intensity=k)
        result = a_star(modified, start=start, goal=goal, eight_connected=True)
        if result.success:
            # Recompute true cost on the ORIGINAL grid (penalty grid
            # inflates cost artificially — we want the real cost)
            true_cost = _compute_path_cost(result.path)
            candidates.append({
                "path": result.path,
                "cost": true_cost,
                "variant": f"detour-{k}",
                "nodes_expanded": result.nodes_expanded,
            })

    # Deduplicate paths that ended up identical
    candidates = _deduplicate(candidates)

    return candidates


def _create_penalty_grid(grid: Grid, rng: np.random.RandomState, intensity: int) -> Grid:
    """
    Copy the grid and randomly block a few cells in the middle region
    to force A* to find a different route.

    Higher intensity → more cells blocked → bigger detour.
    """
    modified = Grid(
        size=grid.size,
        obstacle_ratio=0.0,
        start=grid.start,
        goal=grid.goal,
        seed=None,
    )
    # Copy original obstacles
    modified.grid = grid.grid.copy()

    # Add random "penalty" obstacles in the center region
    mid = grid.size // 2
    span = max(3, grid.size // 4)
    num_blocks = intensity * 3 + rng.randint(1, 4)

    for _ in range(num_blocks):
        r = rng.randint(max(0, mid - span), min(grid.size, mid + span))
        c = rng.randint(max(0, mid - span), min(grid.size, mid + span))
        if (r, c) != grid.start and (r, c) != grid.goal:
            modified.grid[r, c] = 1

    return modified



def _deduplicate(candidates: List[Dict]) -> List[Dict]:
    """Remove candidate paths that are identical (same cell sequence)."""
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c["path"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ------------------------------------------------------------------
# Conflict matrix computation
# ------------------------------------------------------------------

def compute_conflict_matrix(
    all_candidates: Dict[int, List[Dict]],
) -> np.ndarray:
    """
    Build a pairwise conflict matrix between ALL candidate paths
    across ALL robots.

    A "conflict" at timestep t means two robots occupy the same cell.
    Paths are padded so the shorter one stays at its last cell (the
    robot waits at the goal).

    Parameters
    ----------
    all_candidates : dict
        robot_id → list of candidate-path dicts.

    Returns
    -------
    conflict_matrix : np.ndarray, shape (total_paths, total_paths)
        conflict_matrix[i][j] = number of timestep collisions between
        flat-index path i and flat-index path j.

    Also returns:
        index_map : list of (robot_id, candidate_idx) for each flat index.
    """
    # Build flat index → (robot_id, candidate_idx, path)
    index_map = []
    paths_flat = []
    for rid in sorted(all_candidates.keys()):
        for k, cand in enumerate(all_candidates[rid]):
            index_map.append((rid, k))
            paths_flat.append(cand["path"])

    n = len(paths_flat)
    max_len = max(len(p) for p in paths_flat)

    # Pad paths: robot stays at last position after reaching goal
    padded = []
    for p in paths_flat:
        padded.append(p + [p[-1]] * (max_len - len(p)))

    # Count conflicts
    matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            ri, _ = index_map[i]
            rj, _ = index_map[j]
            if ri == rj:
                continue  # same robot — not a conflict
            conflicts = sum(
                1 for t in range(max_len) if padded[i][t] == padded[j][t]
            )
            matrix[i, j] = conflicts
            matrix[j, i] = conflicts

    return matrix, index_map


def get_cost_vector(all_candidates: Dict[int, List[Dict]]) -> np.ndarray:
    """
    Flat vector of path costs aligned with the conflict matrix indices.
    """
    costs = []
    for rid in sorted(all_candidates.keys()):
        for cand in all_candidates[rid]:
            costs.append(cand["cost"])
    return np.array(costs)


# ------------------------------------------------------------------
# Classical baseline selectors (to compare against QAOA)
# ------------------------------------------------------------------

def greedy_select(
    all_candidates: Dict[int, List[Dict]],
) -> Dict[int, int]:
    """
    Greedy selection: pick the shortest (lowest-cost) path for each
    robot, completely ignoring inter-robot conflicts.

    Returns dict: robot_id → index of selected candidate.
    """
    selection = {}
    for rid in sorted(all_candidates.keys()):
        best_idx = 0
        best_cost = float("inf")
        for k, cand in enumerate(all_candidates[rid]):
            if cand["cost"] < best_cost:
                best_cost = cand["cost"]
                best_idx = k
        selection[rid] = best_idx
    return selection


def random_select(
    all_candidates: Dict[int, List[Dict]],
    seed: int = 0,
) -> Dict[int, int]:
    """
    Random selection: pick a random path for each robot.
    """
    rng = np.random.RandomState(seed)
    selection = {}
    for rid in sorted(all_candidates.keys()):
        selection[rid] = rng.randint(0, len(all_candidates[rid]))
    return selection


def brute_force_select(
    all_candidates: Dict[int, List[Dict]],
    conflict_penalty: float = 10.0,
) -> Dict[int, int]:
    """
    Brute-force: try every combination of paths, pick the one with
    lowest (total_cost + penalty × total_conflicts).

    This is the classical optimum — QAOA should match or approach it.
    Feasible only for small numbers (e.g. 3 robots × 4 paths = 64 combos).

    Returns dict: robot_id → index of selected candidate.
    """
    rids = sorted(all_candidates.keys())
    ranges = [range(len(all_candidates[rid])) for rid in rids]

    conflict_matrix, index_map = compute_conflict_matrix(all_candidates)
    costs = get_cost_vector(all_candidates)

    # Build a lookup: (robot_id, cand_idx) → flat index
    flat_lookup = {}
    for flat_idx, (rid, k) in enumerate(index_map):
        flat_lookup[(rid, k)] = flat_idx

    best_combo = None
    best_score = float("inf")

    for combo in itertools.product(*ranges):
        # combo is a tuple: (robot0_choice, robot1_choice, ...)
        total_cost = 0.0
        total_conflicts = 0

        flat_indices = []
        for i, rid in enumerate(rids):
            fi = flat_lookup[(rid, combo[i])]
            flat_indices.append(fi)
            total_cost += costs[fi]

        # Count pairwise conflicts in this combination
        for a in range(len(flat_indices)):
            for b in range(a + 1, len(flat_indices)):
                total_conflicts += conflict_matrix[flat_indices[a], flat_indices[b]]

        score = total_cost + conflict_penalty * total_conflicts
        if score < best_score:
            best_score = score
            best_combo = combo

    selection = {}
    for i, rid in enumerate(rids):
        selection[rid] = best_combo[i]
    return selection


# ------------------------------------------------------------------
# Helper to evaluate a selection
# ------------------------------------------------------------------

def evaluate_selection(
    all_candidates: Dict[int, List[Dict]],
    selection: Dict[int, int],
    conflict_penalty: float = 10.0,
) -> Dict:
    """
    Given a selection (robot_id → candidate index), compute:
        • total_cost: sum of path costs
        • total_conflicts: pairwise timestep collisions
        • score: total_cost + penalty * total_conflicts

    Returns a dict with these values.
    """
    conflict_matrix, index_map = compute_conflict_matrix(all_candidates)
    costs = get_cost_vector(all_candidates)

    flat_lookup = {}
    for flat_idx, (rid, k) in enumerate(index_map):
        flat_lookup[(rid, k)] = flat_idx

    rids = sorted(selection.keys())
    flat_indices = [flat_lookup[(rid, selection[rid])] for rid in rids]

    total_cost = sum(costs[fi] for fi in flat_indices)
    total_conflicts = 0
    for a in range(len(flat_indices)):
        for b in range(a + 1, len(flat_indices)):
            total_conflicts += conflict_matrix[flat_indices[a], flat_indices[b]]

    score = total_cost + conflict_penalty * total_conflicts
    return {
        "total_cost": total_cost,
        "total_conflicts": int(total_conflicts),
        "score": score,
        "selection": selection,
    }


# ------------------------------------------------------------------
# Full pipeline: grid → candidates → conflict matrix (convenience)
# ------------------------------------------------------------------

def prepare_qaoa_input(
    grid: Grid,
    num_candidates: int = 4,
    seed: int = 0,
) -> Dict:
    """
    End-to-end: given a multi-robot Grid, generate candidate paths
    for each robot and compute the conflict matrix.

    Returns a dict with everything QAOA needs:
        - all_candidates: robot_id → list of path dicts
        - conflict_matrix: (N×N) pairwise conflicts
        - index_map: flat_idx → (robot_id, cand_idx)
        - cost_vector: flat cost array
        - robot_configs: from the grid
    """
    all_candidates = {}
    for cfg in grid.robot_configs:
        rid = cfg["id"]
        candidates = generate_candidate_paths(
            grid,
            start=cfg["start"],
            goal=cfg["goal"],
            num_candidates=num_candidates,
            seed=seed + rid * 1000,
        )
        all_candidates[rid] = candidates
        costs_str = ", ".join(f"{c['cost']:.1f}" for c in candidates)
        print(f"  Robot {rid}: generated {len(candidates)} candidate paths "
              f"(costs: [{costs_str}])")

    conflict_matrix, index_map = compute_conflict_matrix(all_candidates)
    cost_vector = get_cost_vector(all_candidates)

    return {
        "all_candidates": all_candidates,
        "conflict_matrix": conflict_matrix,
        "index_map": index_map,
        "cost_vector": cost_vector,
        "robot_configs": grid.robot_configs,
    }
