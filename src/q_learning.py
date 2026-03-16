"""
q_learning.py  –  Standard Tabular Q-Learning Planner
=======================================================

WHAT THIS FILE DOES (plain English):
    The robot LEARNS by trial-and-error how to navigate the grid.
    It runs many "episodes" (practice runs), starting from start and
    trying random moves.  Each move gets a REWARD:
        +100  — reached the goal  (great!)
         -10  — hit a wall / obstacle  (bad!)
          -1  — normal step  (small cost to encourage short paths)
          -3  — revisited an already-visited cell  (wasteful)

    Over time, the robot builds a Q-TABLE: a big lookup table that
    says "if I'm at cell (r,c) and go UP, the expected future reward
    is X."  After training, the robot just follows the highest Q-value
    at each cell → that's the learned path.

WHY Q-LEARNING IS USED:
    It's the standard BASELINE reinforcement learning algorithm.
    Your quantum Q-learning (Week 3) will use the same structure but
    replace the Q-table with quantum-inspired probability amplitudes
    (α, β) and rotation gates.  By having standard Q-learning as a
    baseline, you can show that the quantum version converges faster.

KEY CONCEPTS (for your presentation):
    • STATE  = cell position (row, col)
    • ACTION = direction to move (up, down, left, right)
    • REWARD = feedback signal after each action
    • Q-VALUE = estimated total future reward for (state, action)
    • UPDATE RULE:
        Q(s, a) ← Q(s, a) + α * [reward + γ * max_a' Q(s', a') - Q(s, a)]
        where α = learning rate, γ = discount factor

    • ε-GREEDY exploration:
        With probability ε → move randomly (EXPLORE)
        With probability 1-ε → follow best Q-value (EXPLOIT)
        ε starts high (lots of exploring) and decays over episodes.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from grid import Grid, Position


# ------------------------------------------------------------------
# Actions: 4 cardinal directions
# ------------------------------------------------------------------

ACTIONS = [
    (-1,  0),  # UP
    ( 1,  0),  # DOWN
    ( 0, -1),  # LEFT
    ( 0,  1),  # RIGHT
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


# ------------------------------------------------------------------
# Q-Learning Result container
# ------------------------------------------------------------------

class QLearningResult:
    """Container for Q-learning output (mirrors AStarResult / APFResult)."""

    def __init__(
        self,
        path: Optional[List[Position]],
        cost: float,
        episodes_trained: int,
        success: bool,
        q_table: Optional[np.ndarray] = None,
        training_rewards: Optional[List[float]] = None,
    ):
        self.path = path
        self.cost = cost
        self.episodes_trained = episodes_trained
        self.success = success
        self.q_table = q_table
        self.training_rewards = training_rewards or []

    def __repr__(self) -> str:
        if self.success:
            return (
                f"QLearningResult(success=True, cost={self.cost:.2f}, "
                f"path_len={len(self.path)}, episodes={self.episodes_trained})"
            )
        return (
            f"QLearningResult(success=False, "
            f"episodes={self.episodes_trained})"
        )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_q_learning(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Train a Q-table on the given grid.

    Parameters
    ----------
    grid : Grid
        The environment (static obstacles considered as walls).
    start, goal : Position or None
        Defaults to grid.start / grid.goal.
    episodes : int
        Number of training episodes.
    max_steps_per_episode : int
        Max steps per episode before forced reset.
    alpha : float
        Learning rate (0 < α ≤ 1).
    gamma : float
        Discount factor (0 < γ ≤ 1).
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiply ε by this each episode.
    seed : int or None
        Random seed.

    Returns
    -------
    q_table : np.ndarray, shape (grid.size, grid.size, 4)
        Q-values for each (row, col, action).
    episode_rewards : list of float
        Total reward per episode (for plotting learning curve).
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    rng = np.random.RandomState(seed)

    # Initialise Q-table with small random values
    q_table = rng.uniform(-0.01, 0.01, size=(grid.size, grid.size, len(ACTIONS)))

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(episodes):
        state = start
        total_reward = 0.0
        visited = {state}

        for step in range(max_steps_per_episode):
            # ε-greedy action selection
            if rng.random() < epsilon:
                action_idx = rng.randint(0, len(ACTIONS))
            else:
                action_idx = int(np.argmax(q_table[state[0], state[1]]))

            dr, dc = ACTIONS[action_idx]
            next_state = (state[0] + dr, state[1] + dc)

            # --- Compute reward ---
            if next_state == goal:
                reward = 100.0
            elif not _is_valid_move(grid, next_state):
                reward = -10.0
                next_state = state  # stay in place (wall bounce)
            elif next_state in visited:
                reward = -3.0       # discourage revisiting
            else:
                reward = -1.0       # normal step cost

            visited.add(next_state)
            total_reward += reward

            # --- Q-update ---
            old_q = q_table[state[0], state[1], action_idx]
            next_max_q = np.max(q_table[next_state[0], next_state[1]])
            new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
            q_table[state[0], state[1], action_idx] = new_q

            state = next_state

            if state == goal:
                break

        episode_rewards.append(total_reward)

        # Decay ε
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table, episode_rewards


# ------------------------------------------------------------------
# Planning (greedy policy from trained Q-table)
# ------------------------------------------------------------------

def q_learning_plan(
    grid: Grid,
    q_table: np.ndarray,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    max_steps: int = 500,
) -> QLearningResult:
    """
    Follow the greedy policy from a trained Q-table.

    At each cell, pick the action with the highest Q-value.
    No exploration (ε = 0) — pure exploitation.

    Parameters
    ----------
    grid : Grid
    q_table : np.ndarray
        Trained Q-table from train_q_learning().
    start, goal : Position or None
    max_steps : int

    Returns
    -------
    QLearningResult
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    path = [start]
    current = start
    visited = {start}

    for step in range(max_steps):
        if current == goal:
            cost = _path_cost(path)
            return QLearningResult(
                path=path, cost=cost,
                episodes_trained=0,  # filled in by caller
                success=True,
                q_table=q_table,
            )

        # Pick best action
        q_values = q_table[current[0], current[1]].copy()

        # Mask invalid moves (set to -inf so they won't be chosen)
        for i, (dr, dc) in enumerate(ACTIONS):
            next_pos = (current[0] + dr, current[1] + dc)
            if not _is_valid_move(grid, next_pos):
                q_values[i] = -float("inf")

        if np.all(q_values == -float("inf")):
            # All moves are invalid — stuck
            cost = _path_cost(path)
            return QLearningResult(
                path=path, cost=cost,
                episodes_trained=0,
                success=False,
                q_table=q_table,
            )

        action_idx = int(np.argmax(q_values))
        dr, dc = ACTIONS[action_idx]
        next_pos = (current[0] + dr, current[1] + dc)

        # Loop detection: if we've been here before, try second-best
        if next_pos in visited:
            sorted_actions = np.argsort(q_values)[::-1]
            moved = False
            for ai in sorted_actions:
                if q_values[ai] == -float("inf"):
                    break
                dr2, dc2 = ACTIONS[ai]
                candidate = (current[0] + dr2, current[1] + dc2)
                if candidate not in visited and _is_valid_move(grid, candidate):
                    next_pos = candidate
                    moved = True
                    break
            if not moved:
                # All options visited — allow revisit with best Q
                pass

        current = next_pos
        path.append(current)
        visited.add(current)

    # Timeout
    cost = _path_cost(path)
    return QLearningResult(
        path=path, cost=cost,
        episodes_trained=0,
        success=False,
        q_table=q_table,
    )


# ------------------------------------------------------------------
# Convenience: train + plan in one call
# ------------------------------------------------------------------

def q_learning_full(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    seed: Optional[int] = None,
) -> QLearningResult:
    """
    Train Q-learning on the grid, then follow the learned policy.

    This is the main entry point — call this for a complete result.

    Returns
    -------
    QLearningResult
        Contains path, cost, episodes_trained, q_table, training_rewards.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    q_table, rewards = train_q_learning(
        grid, start=start, goal=goal,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        alpha=alpha, gamma=gamma,
        seed=seed,
    )

    result = q_learning_plan(grid, q_table, start=start, goal=goal)
    result.episodes_trained = episodes
    result.training_rewards = rewards
    result.q_table = q_table

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_valid_move(grid: Grid, pos: Position) -> bool:
    """Check if a position is in-bounds and free (static grid only)."""
    r, c = pos
    if not (0 <= r < grid.size and 0 <= c < grid.size):
        return False
    if grid.grid[r, c] == 1:
        return False
    return True


def _path_cost(path: List[Position]) -> float:
    """Euclidean path cost."""
    cost = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i][0] - path[i + 1][0])
        dc = abs(path[i][1] - path[i + 1][1])
        cost += math.sqrt(2) if (dr + dc == 2) else 1.0
    return cost
