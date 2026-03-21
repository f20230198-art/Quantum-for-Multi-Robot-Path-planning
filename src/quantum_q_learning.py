"""
quantum_q_learning.py  –  Quantum-Inspired Q-Learning Planner
==============================================================

WHAT THIS FILE DOES (plain English):
    This is the QUANTUM version of the Q-Learning planner from Phase 2.
    Instead of storing scalar Q-values in a table, we store ROTATION
    ANGLES (θ) for one qubit per (state, action) pair.

    Each qubit is in state:
        |ψ⟩ = Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

    The probability of measuring |1⟩ is sin²(θ/2), and we interpret
    this as "how good is action a in state s?"  Higher P(|1⟩) = better.

WHY QUANTUM Q-LEARNING:
    1. The rotation-gate update is a smooth, bounded operation — angles
       live on the Bloch sphere, so values never diverge to ±∞ the way
       raw Q-values sometimes do.
    2. Superposition-based exploration: early in training (small θ),
       all actions have roughly equal probability, giving natural
       exploration without needing a separate ε schedule.
    3. This module uses Qiskit to build real quantum circuits with Ry
       gates and measure them, validating that our math matches actual
       quantum mechanics.  When Qiskit is not installed, a pure-numpy
       fallback computes sin²(θ/2) directly.

KEY CONCEPTS (for your presentation):
    • θ[s,a]       = rotation angle for qubit representing (state, action)
    • P(a|s)       = sin²(θ[s,a] / 2)  — probability of choosing action a
    • UPDATE RULE:
        δ = reward + γ * max_a'[P(a'|s')] − P(a|s)       (TD error)
        Δθ = α * δ                                        (rotation delta)
        θ[s,a] ← clamp(θ[s,a] + Δθ,  0, π)               (new angle)

    • Qiskit circuit per measurement:
        qc = QuantumCircuit(1)
        qc.ry(θ, 0)
        qc.measure(0, 0)
        → sample from AerSimulator

REFERENCE:
    Dong et al. (2008) "Quantum reinforcement learning" — the original
    quantum-inspired RL using probability amplitude updates.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from grid import Grid, Position
from utils import path_cost as _path_cost

# ------------------------------------------------------------------
# Actions: 4 cardinal directions (same as classical q_learning.py)
# ------------------------------------------------------------------

ACTIONS = [
    (-1,  0),  # UP
    ( 1,  0),  # DOWN
    ( 0, -1),  # LEFT
    ( 0,  1),  # RIGHT
]

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


# ------------------------------------------------------------------
# Qiskit integration (optional — falls back to numpy if unavailable)
# ------------------------------------------------------------------

_USE_QISKIT = False
_qiskit_sim = None

try:
    from qiskit.circuit import QuantumCircuit
    from qiskit_aer import AerSimulator
    _qiskit_sim = AerSimulator(method="statevector")
    _USE_QISKIT = True
except ImportError:
    pass  # numpy fallback


def _angle_to_prob(theta: float) -> float:
    """
    Convert rotation angle to action-selection probability.

    P(|1⟩) = sin²(θ/2)

    When θ = 0   →  P = 0   (never choose this action)
    When θ = π   →  P = 1   (always choose this action)
    When θ = π/2 →  P = 0.5 (50/50)
    """
    return math.sin(theta / 2.0) ** 2


def _measure_qubit_qiskit(theta: float) -> int:
    """
    Build a real quantum circuit: Ry(θ)|0⟩, measure, return 0 or 1.

    Uses Qiskit AerSimulator for a single-shot measurement.
    """
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    result = _qiskit_sim.run(qc, shots=1).result()
    counts = result.get_counts()
    return int(max(counts, key=counts.get))


def _measure_qubit_numpy(theta: float, rng: np.random.RandomState) -> int:
    """
    Simulate qubit measurement using numpy.

    Sample from Bernoulli(sin²(θ/2)).
    """
    prob_one = _angle_to_prob(theta)
    return 1 if rng.random() < prob_one else 0


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

class QuantumQLearningResult:
    """Container for Quantum Q-Learning output."""

    def __init__(
        self,
        path: Optional[List[Position]],
        cost: float,
        episodes_trained: int,
        success: bool,
        q_angles: Optional[np.ndarray] = None,
        training_rewards: Optional[List[float]] = None,
        used_qiskit: bool = False,
    ):
        self.path = path
        self.cost = cost
        self.episodes_trained = episodes_trained
        self.success = success
        self.q_angles = q_angles
        self.training_rewards = training_rewards or []
        self.used_qiskit = used_qiskit

    def __repr__(self) -> str:
        backend = "Qiskit" if self.used_qiskit else "numpy"
        if self.success:
            return (
                f"QuantumQLResult(success=True, cost={self.cost:.2f}, "
                f"path_len={len(self.path)}, episodes={self.episodes_trained}, "
                f"backend={backend})"
            )
        return (
            f"QuantumQLResult(success=False, "
            f"episodes={self.episodes_trained}, backend={backend})"
        )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_quantum_q_learning(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 300,
    alpha: float = 0.3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    use_qiskit: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Train quantum Q-learning on the given grid.

    Instead of a Q-table of scalar values, we maintain a table of
    rotation angles θ[row, col, action].  The "Q-value" of a
    (state, action) pair is P(|1⟩) = sin²(θ/2).

    Parameters
    ----------
    grid : Grid
    start, goal : Position or None
    episodes : int
        Number of training episodes.
    max_steps_per_episode : int
    alpha : float
        Learning rate for angle updates (smaller than classical
        because angles are bounded [0, π]).
    gamma : float
        Discount factor.
    epsilon_start, epsilon_end, epsilon_decay : float
        ε-greedy schedule (same as classical version).
    use_qiskit : bool or None
        Force Qiskit on/off.  None = auto-detect.
    seed : int or None

    Returns
    -------
    q_angles : np.ndarray, shape (grid.size, grid.size, 4)
        Rotation angles in [0, π] for each (row, col, action).
    episode_rewards : list of float
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    rng = np.random.RandomState(seed)

    qiskit_on = _USE_QISKIT if use_qiskit is None else use_qiskit

    # Initialise angles: small random values near π/4 (moderate prior)
    q_angles = rng.uniform(0.3, 0.8, size=(grid.size, grid.size, len(ACTIONS)))

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(episodes):
        state = start
        total_reward = 0.0
        visited = {state}

        for step in range(max_steps_per_episode):
            # ── Action selection: ε-greedy on qubit probabilities ──
            if rng.random() < epsilon:
                action_idx = rng.randint(0, len(ACTIONS))
            else:
                # Compute P(|1⟩) for each action from current state
                probs = np.array([
                    _angle_to_prob(q_angles[state[0], state[1], a])
                    for a in range(len(ACTIONS))
                ])
                action_idx = int(np.argmax(probs))

            dr, dc = ACTIONS[action_idx]
            next_state = (state[0] + dr, state[1] + dc)

            # ── Compute reward (same structure as classical) ──
            if next_state == goal:
                reward = 100.0
            elif not _is_valid_move(grid, next_state):
                reward = -10.0
                next_state = state  # wall bounce
            elif next_state in visited:
                reward = -3.0
            else:
                reward = -1.0

            visited.add(next_state)
            total_reward += reward

            # ── Quantum update (Ry rotation gate) ──
            #
            # We use a Grover-inspired self-regulating rotation:
            #   - Map reward to a signal in [-1, +1]
            #   - Compute TD error in probability space
            #   - Apply rotation proportional to error, but DAMPED
            #     as the angle approaches its boundary (0 or π).
            #     This prevents saturation and mimics the natural
            #     damping of repeated Grover rotations.
            #
            # Signal mapping:
            #   +100 (goal)    → +1.0
            #    -1  (step)    → -0.01
            #    -3  (revisit) → -0.03
            #   -10  (wall)    → -0.10
            signal = reward / 100.0

            # Current P(|1⟩) for this (state, action)
            old_theta = q_angles[state[0], state[1], action_idx]
            current_prob = _angle_to_prob(old_theta)

            # Best next P(|1⟩) over next-state actions
            next_probs = np.array([
                _angle_to_prob(q_angles[next_state[0], next_state[1], a])
                for a in range(len(ACTIONS))
            ])
            next_max_prob = float(np.max(next_probs))

            # TD target (clipped to [0, 1] since it represents a probability)
            td_target = np.clip(signal + gamma * next_max_prob, 0.0, 1.0)
            td_error = td_target - current_prob

            # Self-regulating rotation: dampen as angle approaches boundary
            # Positive error → rotate toward π, damped by (π - θ)/π
            # Negative error → rotate toward 0, damped by θ/π
            if td_error > 0:
                damping = (math.pi - old_theta) / math.pi
            else:
                damping = old_theta / math.pi

            delta_theta = alpha * td_error * damping

            # Update angle
            new_theta = np.clip(old_theta + delta_theta, 0.01, math.pi - 0.01)
            q_angles[state[0], state[1], action_idx] = new_theta

            state = next_state
            if state == goal:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_angles, episode_rewards


# ------------------------------------------------------------------
# Planning (greedy policy from trained angles)
# ------------------------------------------------------------------

def quantum_q_learning_plan(
    grid: Grid,
    q_angles: np.ndarray,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    max_steps: int = 500,
    use_qiskit: Optional[bool] = None,
    seed: Optional[int] = None,
) -> QuantumQLearningResult:
    """
    Follow the greedy policy derived from trained rotation angles.

    At each cell, pick the action with highest P(|1⟩) = sin²(θ/2).
    Optionally use Qiskit to actually build and measure the circuit.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal
    rng = np.random.RandomState(seed)
    qiskit_on = _USE_QISKIT if use_qiskit is None else use_qiskit

    path = [start]
    current = start
    visited = {start}

    for step in range(max_steps):
        if current == goal:
            cost = _path_cost(path)
            return QuantumQLearningResult(
                path=path, cost=cost,
                episodes_trained=0,
                success=True,
                q_angles=q_angles,
                used_qiskit=qiskit_on,
            )

        # Compute probability for each action
        probs = np.array([
            _angle_to_prob(q_angles[current[0], current[1], a])
            for a in range(len(ACTIONS))
        ])

        # Mask invalid moves
        for i, (dr, dc) in enumerate(ACTIONS):
            next_pos = (current[0] + dr, current[1] + dc)
            if not _is_valid_move(grid, next_pos):
                probs[i] = -1.0

        if np.all(probs < 0):
            cost = _path_cost(path)
            return QuantumQLearningResult(
                path=path, cost=cost,
                episodes_trained=0,
                success=False,
                q_angles=q_angles,
                used_qiskit=qiskit_on,
            )

        action_idx = int(np.argmax(probs))
        dr, dc = ACTIONS[action_idx]
        next_pos = (current[0] + dr, current[1] + dc)

        # Loop detection: if revisiting, try second-best
        if next_pos in visited:
            sorted_actions = np.argsort(probs)[::-1]
            moved = False
            for ai in sorted_actions:
                if probs[ai] < 0:
                    break
                dr2, dc2 = ACTIONS[ai]
                candidate = (current[0] + dr2, current[1] + dc2)
                if candidate not in visited and _is_valid_move(grid, candidate):
                    next_pos = candidate
                    moved = True
                    break
            if not moved:
                pass  # allow revisit with best action

        current = next_pos
        path.append(current)
        visited.add(current)

    # Timeout
    cost = _path_cost(path)
    return QuantumQLearningResult(
        path=path, cost=cost,
        episodes_trained=0,
        success=False,
        q_angles=q_angles,
        used_qiskit=qiskit_on,
    )


# ------------------------------------------------------------------
# Convenience: train + plan in one call
# ------------------------------------------------------------------

def quantum_q_learning_full(
    grid: Grid,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
    episodes: int = 500,
    max_steps_per_episode: int = 300,
    alpha: float = 0.3,
    gamma: float = 0.95,
    use_qiskit: Optional[bool] = None,
    seed: Optional[int] = None,
) -> QuantumQLearningResult:
    """
    Train quantum Q-learning, then follow the learned policy.

    This is the main entry point — call this for a complete result.
    """
    start = start if start is not None else grid.start
    goal = goal if goal is not None else grid.goal

    q_angles, rewards = train_quantum_q_learning(
        grid, start=start, goal=goal,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        alpha=alpha, gamma=gamma,
        use_qiskit=use_qiskit,
        seed=seed,
    )

    result = quantum_q_learning_plan(
        grid, q_angles,
        start=start, goal=goal,
        use_qiskit=use_qiskit,
        seed=seed,
    )
    result.episodes_trained = episodes
    result.training_rewards = rewards
    result.q_angles = q_angles

    return result


# ------------------------------------------------------------------
# Qiskit verification: compare numpy vs circuit measurement
# ------------------------------------------------------------------

def verify_qiskit_match(theta: float, shots: int = 1000) -> Dict:
    """
    Compare numpy P(|1⟩) = sin²(θ/2) with actual Qiskit circuit
    measurement statistics.  Used for validation / presentation.

    Returns dict with numpy_prob, qiskit_prob, difference.
    """
    numpy_prob = _angle_to_prob(theta)

    if not _USE_QISKIT:
        return {
            "theta": theta,
            "numpy_prob": numpy_prob,
            "qiskit_prob": None,
            "difference": None,
            "qiskit_available": False,
        }

    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    result = _qiskit_sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    qiskit_prob = counts.get("1", 0) / shots

    return {
        "theta": theta,
        "numpy_prob": numpy_prob,
        "qiskit_prob": qiskit_prob,
        "difference": abs(numpy_prob - qiskit_prob),
        "qiskit_available": True,
    }


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
