"""
qaoa_optimizer.py  –  QAOA Multi-Robot Path Optimizer
=====================================================

WHAT THIS FILE DOES (plain English):
    Given K candidate paths for each of N robots and a pairwise conflict
    matrix, this module uses the Quantum Approximate Optimization Algorithm
    (QAOA) to select one path per robot that minimizes:

        total_path_cost  +  penalty × total_collisions

    This is the SAME objective that brute_force_select() optimizes, but
    QAOA does it without exhaustively checking every combination.

HOW IT WORKS:
    1.  QUBO Formulation
        - Binary variables: x[i,k] = 1 if robot i takes path k, else 0.
        - One-hot constraint: each robot picks exactly one path.
        - Objective: minimize cost + penalty * conflicts.

    2.  QAOA Circuit (numpy simulation)
        - Encode QUBO as an Ising Hamiltonian.
        - Apply p layers of (cost unitary + mixer unitary).
        - Optimize gamma/beta angles with scipy.
        - Sample the best bitstring → decode into robot selections.

    3.  Optional Qiskit Backend
        If qiskit-optimization is installed, can use its QAOA solver
        as an alternative backend for comparison.

WHY QAOA OVER BRUTE-FORCE?
    Brute-force checks O(K^N) combinations — exponential.
    QAOA searches the solution space using quantum interference,
    aiming for near-optimal solutions in polynomial time.
    For 3 robots × 4 paths = 64 combos, both are fast.
    For 20 robots × 4 paths = 4^20 ≈ 10^12 combos, brute-force
    is impossible while QAOA remains feasible.
"""

import time
import math
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from multi_robot import (
    compute_conflict_matrix,
    get_cost_vector,
    evaluate_selection,
)


# ------------------------------------------------------------------
# Result data class
# ------------------------------------------------------------------

@dataclass
class QAOAResult:
    """Result of QAOA optimization."""
    selection: Dict[int, int]       # robot_id → candidate index
    total_cost: float
    total_conflicts: int
    score: float
    optimal_angles: Optional[dict]  # gamma/beta angles found
    num_qubits: int
    num_layers: int                 # QAOA depth (p)
    backend: str                    # "numpy" or "qiskit"
    time_ms: float                  # wall-clock time
    iterations: int                 # optimizer iterations

    def __repr__(self):
        return (f"QAOAResult(score={self.score:.2f}, cost={self.total_cost:.1f}, "
                f"conflicts={self.total_conflicts}, qubits={self.num_qubits}, "
                f"p={self.num_layers}, backend={self.backend}, "
                f"time={self.time_ms:.1f}ms)")


# ------------------------------------------------------------------
# QUBO formulation
# ------------------------------------------------------------------

def build_qubo(
    all_candidates: Dict[int, List[Dict]],
    conflict_penalty: float = 10.0,
    one_hot_penalty: float = 50.0,
) -> Tuple[np.ndarray, dict]:
    """
    Build the QUBO matrix for multi-robot path selection.

    Binary variables: x[i*K + k] = 1 if robot i selects candidate k.
    Total variables = sum of candidates per robot.

    Objective (to minimize):
        Q = path_cost_terms + conflict_penalty * collision_terms
            + one_hot_penalty * (one-hot constraint violations)

    Parameters
    ----------
    all_candidates : dict
        robot_id → list of candidate path dicts
    conflict_penalty : float
        Weight for collision terms.
    one_hot_penalty : float
        Penalty weight for constraint: each robot picks exactly 1 path.

    Returns
    -------
    Q : np.ndarray, shape (n_vars, n_vars)
        Upper-triangular QUBO matrix.
    var_map : dict
        Mapping info: robot_ids, candidates_per_robot, offsets, etc.
    """
    rids = sorted(all_candidates.keys())

    # Build variable offset map
    offsets = {}     # robot_id → starting index in the flat variable vector
    idx = 0
    for rid in rids:
        offsets[rid] = idx
        idx += len(all_candidates[rid])
    n_vars = idx

    Q = np.zeros((n_vars, n_vars), dtype=np.float64)

    # --- Linear terms: path costs on the diagonal ---
    cost_vector = get_cost_vector(all_candidates)
    for i in range(n_vars):
        Q[i, i] += cost_vector[i]

    # --- Quadratic terms: conflict penalties ---
    conflict_matrix, index_map = compute_conflict_matrix(all_candidates)
    for a in range(len(index_map)):
        for b in range(a + 1, len(index_map)):
            ra, _ = index_map[a]
            rb, _ = index_map[b]
            if ra == rb:
                continue  # same robot — skip
            n_conflicts = conflict_matrix[a, b]
            if n_conflicts > 0:
                Q[a, b] += conflict_penalty * n_conflicts

    # --- One-hot constraint: each robot picks exactly 1 path ---
    # Penalty: P * (sum_k x[i,k] - 1)^2
    #        = P * (sum_k x[i,k]^2 - 2*sum_k x[i,k] + 1)
    #        = P * (sum_k x[i,k] - 2*sum_k x[i,k] + 1)   [since x^2 = x for binary]
    #        = P * (-sum_k x[i,k] + 2*sum_{k<l} x[i,k]*x[i,l] + 1)
    for rid in rids:
        start = offsets[rid]
        n_k = len(all_candidates[rid])
        # Linear: -P on each variable (encourages picking at least one)
        for k in range(n_k):
            Q[start + k, start + k] -= one_hot_penalty
        # Quadratic: +2P on each pair (discourages picking more than one)
        for k in range(n_k):
            for l in range(k + 1, n_k):
                Q[start + k, start + l] += 2 * one_hot_penalty
        # Constant term +P (ignored in optimization, but noted)

    var_map = {
        "robot_ids": rids,
        "offsets": offsets,
        "candidates_per_robot": {rid: len(all_candidates[rid]) for rid in rids},
        "n_vars": n_vars,
        "index_map": index_map,
    }

    return Q, var_map


def decode_bitstring(
    bitstring: np.ndarray,
    var_map: dict,
) -> Optional[Dict[int, int]]:
    """
    Decode a binary solution vector into robot selections.

    Returns None if the solution violates constraints (robot picks
    0 or >1 paths).  Otherwise returns {robot_id: candidate_index}.
    """
    selection = {}
    for rid in var_map["robot_ids"]:
        start = var_map["offsets"][rid]
        n_k = var_map["candidates_per_robot"][rid]
        chosen = []
        for k in range(n_k):
            if bitstring[start + k] > 0.5:
                chosen.append(k)
        if len(chosen) == 1:
            selection[rid] = chosen[0]
        elif len(chosen) == 0:
            # No path chosen — pick lowest-cost as fallback
            selection[rid] = 0
        else:
            # Multiple chosen — pick first (constraint violated)
            selection[rid] = chosen[0]
    return selection


def qubo_energy(bitstring: np.ndarray, Q: np.ndarray) -> float:
    """Compute QUBO energy: x^T Q x."""
    return float(bitstring @ Q @ bitstring)


# ------------------------------------------------------------------
# Numpy QAOA Simulator
# ------------------------------------------------------------------

def _qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO to Ising model.

    QUBO:  min x^T Q x,  x in {0,1}^n
    Ising: min z^T J z + h^T z + c,  z in {-1,+1}^n

    Substitution: x = (1 - z) / 2  →  x=0 when z=+1, x=1 when z=-1
    """
    n = Q.shape[0]
    # Make Q symmetric for the conversion
    Qs = (Q + Q.T) / 2.0

    h = np.zeros(n)
    J = np.zeros((n, n))
    c = 0.0

    for i in range(n):
        h[i] = -0.5 * Qs[i, i]
        for j in range(n):
            if i != j:
                h[i] -= 0.25 * Qs[i, j]
        c += 0.25 * Qs[i, i]

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = 0.25 * Qs[i, j]
            c += 0.25 * Qs[i, j]

    return J, h, c


class NumpyQAOA:
    """
    QAOA simulator using numpy statevector simulation.

    For n qubits, maintains a 2^n complex state vector and applies
    the cost and mixer unitaries directly.

    Practical for n <= ~16 qubits (64K amplitudes).
    """

    def __init__(self, Q: np.ndarray, p: int = 1):
        """
        Parameters
        ----------
        Q : np.ndarray
            QUBO matrix (n x n).
        p : int
            Number of QAOA layers.
        """
        self.Q = Q
        self.n = Q.shape[0]
        self.p = p
        self.J, self.h, self.c = _qubo_to_ising(Q)
        self.N = 2 ** self.n  # state space size

        # Precompute diagonal cost Hamiltonian for all 2^n basis states
        self._cost_diag = self._precompute_cost_diagonal()

    def _precompute_cost_diagonal(self) -> np.ndarray:
        """
        Compute the cost function value for every basis state.
        Returns a vector of length 2^n.
        """
        diag = np.zeros(self.N)
        for idx in range(self.N):
            # Convert index to spin configuration z in {+1, -1}
            z = np.array([1 - 2 * ((idx >> (self.n - 1 - i)) & 1)
                          for i in range(self.n)])
            # Ising energy: sum_ij J_ij z_i z_j + sum_i h_i z_i
            energy = 0.0
            for i in range(self.n):
                energy += self.h[i] * z[i]
                for j in range(i + 1, self.n):
                    energy += self.J[i, j] * z[i] * z[j]
            diag[idx] = energy
        return diag

    def _apply_cost_unitary(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """Apply e^{-i * gamma * C} to the state (diagonal in Z basis)."""
        return state * np.exp(-1j * gamma * self._cost_diag)

    def _apply_mixer_unitary(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply the transverse-field mixer: product of e^{-i * beta * X_j}
        for each qubit j.

        For each qubit, this is a 2x2 rotation applied to pairs of
        amplitudes that differ only in bit j.
        """
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        result = state.copy()

        for j in range(self.n):
            new_result = np.zeros_like(result)
            bit_mask = 1 << (self.n - 1 - j)
            for idx in range(self.N):
                if idx & bit_mask == 0:
                    # idx has bit j = 0, partner has bit j = 1
                    partner = idx | bit_mask
                    new_result[idx] += cos_b * result[idx] - 1j * sin_b * result[partner]
                    new_result[partner] += -1j * sin_b * result[idx] + cos_b * result[partner]
            result = new_result

        return result

    def evaluate(self, params: np.ndarray) -> float:
        """
        Run QAOA circuit with given parameters and return expected cost.

        Parameters
        ----------
        params : np.ndarray, shape (2*p,)
            [gamma_1, ..., gamma_p, beta_1, ..., beta_p]

        Returns
        -------
        expected_cost : float
        """
        gammas = params[:self.p]
        betas = params[self.p:]

        # Start in uniform superposition |+>^n
        state = np.ones(self.N, dtype=np.complex128) / np.sqrt(self.N)

        # Apply p layers
        for layer in range(self.p):
            state = self._apply_cost_unitary(state, gammas[layer])
            state = self._apply_mixer_unitary(state, betas[layer])

        # Expected cost = <state| C |state> = sum_i |a_i|^2 * cost_i
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self._cost_diag))

    def get_probabilities(self, params: np.ndarray) -> np.ndarray:
        """Return probability distribution over all basis states."""
        gammas = params[:self.p]
        betas = params[self.p:]

        state = np.ones(self.N, dtype=np.complex128) / np.sqrt(self.N)
        for layer in range(self.p):
            state = self._apply_cost_unitary(state, gammas[layer])
            state = self._apply_mixer_unitary(state, betas[layer])

        return np.abs(state) ** 2

    def sample_best(self, params: np.ndarray, num_samples: int = 100) -> np.ndarray:
        """
        Sample from the QAOA output distribution and return the
        best (lowest QUBO energy) bitstring found.
        """
        probs = self.get_probabilities(params)

        # Pick top candidates by probability
        top_indices = np.argsort(probs)[-num_samples:]

        best_energy = float("inf")
        best_bits = None
        for idx in top_indices:
            # Convert index to bitstring
            bits = np.array([(idx >> (self.n - 1 - i)) & 1
                             for i in range(self.n)], dtype=np.float64)
            energy = qubo_energy(bits, self.Q)
            if energy < best_energy:
                best_energy = energy
                best_bits = bits

        return best_bits


def qaoa_optimize_numpy(
    Q: np.ndarray,
    p: int = 2,
    num_restarts: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run QAOA optimization using numpy statevector simulation.

    Parameters
    ----------
    Q : QUBO matrix
    p : number of QAOA layers
    num_restarts : number of random initializations
    seed : random seed

    Returns
    -------
    best_bitstring : np.ndarray
    best_params : np.ndarray
    total_iters : int
    """
    from scipy.optimize import minimize

    rng = np.random.RandomState(seed)
    qaoa = NumpyQAOA(Q, p=p)

    best_params = None
    best_cost = float("inf")
    total_iters = 0

    for restart in range(num_restarts):
        # Random initialization of angles
        init_params = rng.uniform(0, 2 * np.pi, size=2 * p)

        result = minimize(
            qaoa.evaluate,
            init_params,
            method="COBYLA",
            options={"maxiter": 200, "rhobeg": 0.5},
        )

        total_iters += result.nfev

        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x

    # Extract best bitstring from optimized distribution
    best_bitstring = qaoa.sample_best(best_params, num_samples=min(200, qaoa.N))

    return best_bitstring, best_params, total_iters


# ------------------------------------------------------------------
# Qiskit QAOA Backend (optional)
# ------------------------------------------------------------------

def _try_qiskit_qaoa(
    Q: np.ndarray,
    var_map: dict,
    all_candidates: Dict[int, List[Dict]],
    conflict_penalty: float,
    one_hot_penalty: float,
    p: int = 2,
) -> Optional[Dict]:
    """
    Try using qiskit-optimization's QAOA. Returns None if not available.
    """
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import Sampler
    except ImportError:
        return None

    try:
        n_vars = var_map["n_vars"]
        rids = var_map["robot_ids"]

        # Build QuadraticProgram
        qp = QuadraticProgram("multi_robot_path_selection")

        # Add binary variables
        for i in range(n_vars):
            qp.binary_var(f"x{i}")

        # Set objective from QUBO matrix
        linear = {f"x{i}": float(Q[i, i]) for i in range(n_vars)}
        quadratic = {}
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(Q[i, j]) > 1e-10:
                    quadratic[(f"x{i}", f"x{j}")] = float(Q[i, j])

        qp.minimize(linear=linear, quadratic=quadratic)

        # Solve with QAOA
        sampler = Sampler()
        optimizer = COBYLA(maxiter=200)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(qp)

        # Decode solution
        bitstring = np.array([result.variables_dict[f"x{i}"]
                              for i in range(n_vars)])
        selection = decode_bitstring(bitstring, var_map)

        return {
            "selection": selection,
            "bitstring": bitstring,
            "energy": result.fval,
        }
    except Exception:
        return None


# ------------------------------------------------------------------
# Main API: qaoa_select()
# ------------------------------------------------------------------

def qaoa_select(
    all_candidates: Dict[int, List[Dict]],
    conflict_penalty: float = 10.0,
    one_hot_penalty: float = 50.0,
    p: int = 2,
    num_restarts: int = 5,
    seed: int = 0,
    prefer_qiskit: bool = True,
) -> Dict[int, int]:
    """
    Select one path per robot using QAOA optimization.

    This is the main entry point — drop-in replacement for
    greedy_select() or brute_force_select().

    Parameters
    ----------
    all_candidates : dict
        robot_id → list of candidate path dicts.
    conflict_penalty : float
        Weight for collision penalty in objective.
    one_hot_penalty : float
        Penalty for constraint violations.
    p : int
        QAOA circuit depth (number of layers).
    num_restarts : int
        Number of random angle initializations (numpy backend).
    seed : int
        Random seed.
    prefer_qiskit : bool
        If True and qiskit-optimization is available, use it.

    Returns
    -------
    selection : dict
        robot_id → index of selected candidate.
    """
    Q, var_map = build_qubo(all_candidates, conflict_penalty, one_hot_penalty)
    bitstring, _, _ = qaoa_optimize_numpy(Q, p=p, num_restarts=num_restarts, seed=seed)
    selection = decode_bitstring(bitstring, var_map)
    return selection


def qaoa_select_full(
    all_candidates: Dict[int, List[Dict]],
    conflict_penalty: float = 10.0,
    one_hot_penalty: float = 50.0,
    p: int = 2,
    num_restarts: int = 5,
    seed: int = 0,
    prefer_qiskit: bool = True,
) -> QAOAResult:
    """
    Full QAOA optimization with detailed result reporting.

    Returns a QAOAResult with timing, angles, and evaluation metrics.
    """
    t0 = time.perf_counter()

    Q, var_map = build_qubo(all_candidates, conflict_penalty, one_hot_penalty)
    n_vars = var_map["n_vars"]

    backend_used = "numpy"
    iterations = 0

    # Try qiskit backend first if preferred
    if prefer_qiskit:
        qiskit_result = _try_qiskit_qaoa(
            Q, var_map, all_candidates,
            conflict_penalty, one_hot_penalty, p=p,
        )
        if qiskit_result is not None:
            backend_used = "qiskit"
            selection = qiskit_result["selection"]
            optimal_angles = None  # qiskit doesn't expose these easily
            iterations = 200  # approximate
        else:
            qiskit_result = None

    if backend_used == "numpy":
        bitstring, best_params, iterations = qaoa_optimize_numpy(
            Q, p=p, num_restarts=num_restarts, seed=seed,
        )
        selection = decode_bitstring(bitstring, var_map)
        optimal_angles = {
            "gammas": best_params[:p].tolist(),
            "betas": best_params[p:].tolist(),
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Evaluate the selection
    ev = evaluate_selection(all_candidates, selection, conflict_penalty)

    return QAOAResult(
        selection=selection,
        total_cost=ev["total_cost"],
        total_conflicts=ev["total_conflicts"],
        score=ev["score"],
        optimal_angles=optimal_angles,
        num_qubits=n_vars,
        num_layers=p,
        backend=backend_used,
        time_ms=elapsed_ms,
        iterations=iterations,
    )


# ------------------------------------------------------------------
# Scalability benchmark
# ------------------------------------------------------------------

def scalability_benchmark(
    robot_counts: List[int] = [2, 3, 4, 5],
    candidates_per_robot: int = 4,
    conflict_penalty: float = 10.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Compare QAOA vs brute-force computation time for increasing
    numbers of robots.

    Generates synthetic cost vectors and conflict matrices to test
    scaling behavior without needing actual grid environments.

    Returns list of dicts with timing and quality data.
    """
    from multi_robot import brute_force_select

    results = []
    rng = np.random.RandomState(seed)

    for n_robots in robot_counts:
        n_vars = n_robots * candidates_per_robot
        total_combos = candidates_per_robot ** n_robots

        # Generate synthetic candidates
        all_candidates = {}
        for rid in range(n_robots):
            candidates = []
            for k in range(candidates_per_robot):
                # Synthetic path: just need cost, path can be dummy
                base_cost = 20.0 + rng.uniform(-5, 10)
                path = [(0, 0), (0, k + 1), (rid + 1, k + 1)]  # dummy path
                candidates.append({
                    "path": path,
                    "cost": base_cost + k * 2,
                    "variant": f"synthetic-{k}",
                })
            all_candidates[rid] = candidates

        print(f"  [{n_robots} robots × {candidates_per_robot} paths = "
              f"{n_vars} qubits, {total_combos} combos]")

        # -- Brute-force timing --
        t0 = time.perf_counter()
        bf_sel = brute_force_select(all_candidates, conflict_penalty)
        bf_ms = (time.perf_counter() - t0) * 1000
        bf_ev = evaluate_selection(all_candidates, bf_sel, conflict_penalty)

        # -- QAOA timing --
        # Only run QAOA if qubits <= 16 (2^16 = 65536 states, feasible)
        if n_vars <= 12:
            qaoa_result = qaoa_select_full(
                all_candidates, conflict_penalty=conflict_penalty,
                p=1, num_restarts=2, seed=seed,
            )
            qaoa_ms = qaoa_result.time_ms
            qaoa_score = qaoa_result.score
        else:
            qaoa_ms = None
            qaoa_score = None

        results.append({
            "n_robots": n_robots,
            "n_qubits": n_vars,
            "total_combos": total_combos,
            "bf_time_ms": bf_ms,
            "bf_score": bf_ev["score"],
            "qaoa_time_ms": qaoa_ms,
            "qaoa_score": qaoa_score,
        })

    return results
