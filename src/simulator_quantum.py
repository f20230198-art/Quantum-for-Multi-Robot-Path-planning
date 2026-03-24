"""
simulator_quantum.py  –  Quantum Planners Interactive Demo
============================================================

Shows quantum-inspired and hybrid algorithms in action:
    Quantum QL  — Rotation-gate Q-learning (bounded angles, no divergence)
    Hybrid      — A* + APF + Quantum QL (3-layer adaptive planner)
    QAOA        — Quantum Approximate Optimization for multi-robot coordination

RUN:
    python simulator_quantum.py
    python simulator_quantum.py --robots 5 --size 25
    python simulator_quantum.py --dynamic 8

CONTROLS:
    SPACE       — Pause / Resume
    P           — Cycle planner (Quantum QL → Hybrid → QAOA → ...)
    LEFT CLICK  — Place / remove wall
    MIDDLE CLICK— Spawn moving obstacle
    R           — Reset
    Q / ESC     — Quit

WHAT TO DEMO:
    1. Start with Quantum QL — watch rotation-angle policy navigate
    2. Build a U-shaped trap with walls → Quantum QL escapes it
    3. Switch to Hybrid (press P) — see A*/APF/QL layers cooperate
    4. Add moving obstacles → Hybrid adapts in real-time (APF layer)
    5. Switch to QAOA — coordinated multi-robot path selection
    6. Compare with classical simulator to show the difference!
"""

import sys
import os
import argparse
import copy

sys.path.insert(0, os.path.dirname(__file__))

from grid import Grid
from quantum_q_learning import train_quantum_q_learning
from simulator_base import RealtimeSimulator


class QuantumSimulator(RealtimeSimulator):
    """Interactive simulator with quantum planners only."""

    PLANNER_NAMES = ["Quantum QL", "Hybrid", "QAOA"]

    def _initial_plan(self):
        if self.planner_name in ("Quantum QL", "Hybrid"):
            self._train_quantum_angles()
        elif self.planner_name == "QAOA":
            self._plan_qaoa()
            return
        super()._initial_plan()

    def _on_cycle_planner(self):
        if self.planner_name in ("Quantum QL", "Hybrid"):
            self._train_quantum_angles()
        elif self.planner_name == "QAOA":
            self._plan_qaoa()

    def _train_quantum_angles(self):
        self._log("Training Quantum Q-tables (500 episodes per robot)...")
        for agent in self.robots:
            q_angles, _ = train_quantum_q_learning(
                self.grid,
                start=agent.start,
                goal=agent.goal,
                episodes=500,
                seed=42 + agent.rid,
            )
            agent.q_angles = q_angles
        self._log("Quantum Q-tables trained.")


def launch_quantum(
    size=20, num_robots=3, num_dynamic=4,
    obstacle_ratio=0.12, seed=42, cell_size=32,
):
    print("=" * 55)
    print("  QUANTUM PLANNERS — Interactive Simulator")
    print("=" * 55)
    print(f"\n  Planners: Quantum QL, Hybrid, QAOA")
    print(f"  Grid: {size}x{size}, {num_robots} robots, "
          f"{num_dynamic} dynamic obstacles")
    print(f"\n  Press SPACE to start, P to cycle planners")
    print(f"  Middle-click to add moving obstacles\n")

    grid = Grid.create_multi_robot_env(
        size=size, obstacle_ratio=obstacle_ratio,
        num_robots=num_robots, num_dynamic=num_dynamic, seed=seed,
    )
    for cfg in grid.robot_configs:
        print(f"  Robot {cfg['id']}: {cfg['start']} -> {cfg['goal']}")

    sim = QuantumSimulator(
        grid=grid, cell_size=cell_size, ticks_per_sec=3.0,
        title=f"Quantum Planners ({num_robots} robots, {size}x{size})",
    )
    sim.run()
    print("\n[sim]  Simulator closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum planners interactive simulator")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--dynamic", type=int, default=4)
    parser.add_argument("--obstacles", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=32)
    args = parser.parse_args()

    launch_quantum(
        size=args.size, num_robots=args.robots,
        num_dynamic=args.dynamic, obstacle_ratio=args.obstacles,
        seed=args.seed, cell_size=args.cell_size,
    )
