"""
simulator_classical.py  –  Classical Planners Interactive Demo
===============================================================

Shows how traditional algorithms handle multi-robot navigation:
    A*          — Global optimal search (plans once, can replan)
    APF         — Local reactive potential fields (no global view)
    Q-Learning  — Reinforcement learning baseline (trained offline)

RUN:
    python simulator_classical.py
    python simulator_classical.py --robots 5 --size 25
    python simulator_classical.py --dynamic 8

CONTROLS:
    SPACE       — Pause / Resume
    P           — Cycle planner (A* → APF → Q-Learning → A* ...)
    LEFT CLICK  — Place / remove wall
    MIDDLE CLICK— Spawn moving obstacle
    R           — Reset
    Q / ESC     — Quit

WHAT TO DEMO:
    1. Start with A* — watch robots find optimal paths
    2. Middle-click to add moving obstacles — A* replans but is slow
    3. Switch to APF (press P) — fast reactions but gets stuck in traps
    4. Switch to Q-Learning (press P) — learned policy, no replanning
    5. Place a U-shaped wall → APF gets stuck, A* finds the way around
"""

import sys
import os
import argparse
import copy

sys.path.insert(0, os.path.dirname(__file__))

from grid import Grid
from q_learning import train_q_learning
from simulator_base import RealtimeSimulator


class ClassicalSimulator(RealtimeSimulator):
    """Interactive simulator with classical planners only."""

    PLANNER_NAMES = ["A*", "APF", "Q-Learning"]

    def _initial_plan(self):
        if self.planner_name == "Q-Learning":
            self._train_q_tables()
        super()._initial_plan()

    def _on_cycle_planner(self):
        if self.planner_name == "Q-Learning":
            self._train_q_tables()

    def _train_q_tables(self):
        self._log("Training Q-tables (500 episodes per robot)...")
        for agent in self.robots:
            q_table, _ = train_q_learning(
                self.grid,
                start=agent.start,
                goal=agent.goal,
                episodes=500,
                seed=42 + agent.rid,
            )
            agent.q_table = q_table
        self._log("Q-tables trained.")


def launch_classical(
    size=20, num_robots=3, num_dynamic=4,
    obstacle_ratio=0.12, seed=42, cell_size=32,
):
    print("=" * 55)
    print("  CLASSICAL PLANNERS — Interactive Simulator")
    print("=" * 55)
    print(f"\n  Planners: A*, APF, Q-Learning")
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

    sim = ClassicalSimulator(
        grid=grid, cell_size=cell_size, ticks_per_sec=3.0,
        title=f"Classical Planners ({num_robots} robots, {size}x{size})",
    )
    sim.run()
    print("\n[sim]  Simulator closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classical planners interactive simulator")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--dynamic", type=int, default=4)
    parser.add_argument("--obstacles", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=32)
    args = parser.parse_args()

    launch_classical(
        size=args.size, num_robots=args.robots,
        num_dynamic=args.dynamic, obstacle_ratio=args.obstacles,
        seed=args.seed, cell_size=args.cell_size,
    )
