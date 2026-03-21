"""
dynamic_env.py  –  Time-Stepped Simulation Environment
=======================================================

WHAT THIS FILE DOES (plain English):
    Runs a simulation loop where:
        1. Multiple robots each take one step per tick
        2. Dynamic obstacles move each tick
        3. Collisions are detected and recorded
        4. The simulation ends when all robots reach their goals
           (or a max-time limit is hit)

    This is the "game engine" that ties the grid, robots, and
    obstacles together into one coherent simulation.

HOW IT WORKS:
    The Simulation class takes a Grid (with dynamic obstacles already
    added) and a list of robot paths.  Each tick:
        • Dynamic obstacles advance one step
        • Each robot advances one step along its path
        • Collision checks run (robot-robot and robot-obstacle)
    The output is a SimulationResult with per-tick snapshots you
    can feed into the visualizer.
"""

import copy
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from grid import Grid, Position


# ------------------------------------------------------------------
# Data classes for clean output
# ------------------------------------------------------------------

@dataclass
class RobotState:
    """Snapshot of one robot at one point in time."""
    id: int
    pos: Position
    reached_goal: bool = False


@dataclass
class TickSnapshot:
    """Full state of the simulation at one tick."""
    tick: int
    robot_states: List[RobotState]
    dynamic_obstacle_positions: List[Position]
    collisions: List[Dict]          # list of {"type": ..., "agents": ..., "pos": ...}


@dataclass
class SimulationResult:
    """Complete record of a simulation run."""
    ticks: List[TickSnapshot]
    total_ticks: int
    all_reached_goal: bool
    total_collisions: int
    per_robot_stats: Dict[int, Dict]   # id → {path_length, time_to_goal, collisions}


# ------------------------------------------------------------------
# Simulation engine
# ------------------------------------------------------------------

class Simulation:
    """
    Runs a time-stepped multi-robot simulation on a Grid.

    Parameters
    ----------
    grid : Grid
        The environment (with dynamic obstacles already added).
    robot_paths : dict
        Maps robot_id (int) → list of Positions (the planned path).
        The robot follows this path step by step.
    max_ticks : int
        Safety limit to prevent infinite loops.
    """

    def __init__(
        self,
        grid: Grid,
        robot_paths: Dict[int, List[Position]],
        max_ticks: int = 200,
    ):
        self.grid = copy.deepcopy(grid)     # work on a copy so we don't mutate the original
        self.robot_paths = robot_paths
        self.max_ticks = max_ticks

        # Robot cursors — which step of the path each robot is on
        self.cursors: Dict[int, int] = {rid: 0 for rid in robot_paths}
        self.reached_goal: Dict[int, bool] = {rid: False for rid in robot_paths}
        self.goal_tick: Dict[int, int] = {}
        self.collision_counts: Dict[int, int] = {rid: 0 for rid in robot_paths}

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the full simulation and return results."""
        snapshots: List[TickSnapshot] = []
        total_collisions = 0

        for tick in range(self.max_ticks):
            # 1. Move dynamic obstacles
            self.grid.step_dynamic_obstacles()

            # 2. Advance each robot one step along its path
            current_positions: Dict[int, Position] = {}
            for rid, path in self.robot_paths.items():
                if self.reached_goal[rid]:
                    # Robot already at goal — stay there
                    current_positions[rid] = path[-1]
                    continue

                cursor = self.cursors[rid]
                next_cursor = min(cursor + 1, len(path) - 1)
                self.cursors[rid] = next_cursor
                pos = path[next_cursor]
                current_positions[rid] = pos

                # Check if reached goal
                goal = self._get_goal(rid)
                if pos == goal:
                    self.reached_goal[rid] = True
                    self.goal_tick[rid] = tick

            # 3. Collision detection
            tick_collisions = self._detect_collisions(current_positions, tick)
            total_collisions += len(tick_collisions)

            # 4. Record snapshot
            robot_states = [
                RobotState(id=rid, pos=current_positions[rid],
                           reached_goal=self.reached_goal[rid])
                for rid in sorted(current_positions)
            ]
            snap = TickSnapshot(
                tick=tick,
                robot_states=robot_states,
                dynamic_obstacle_positions=self.grid.get_dynamic_obstacle_positions(),
                collisions=tick_collisions,
            )
            snapshots.append(snap)

            # 5. Early exit if all robots reached goal
            if all(self.reached_goal.values()):
                break

        # Build per-robot stats
        per_robot = {}
        for rid in self.robot_paths:
            per_robot[rid] = {
                "path_length": len(self.robot_paths[rid]),
                "time_to_goal": self.goal_tick.get(rid, self.max_ticks),
                "reached_goal": self.reached_goal[rid],
                "collisions": self.collision_counts[rid],
            }

        return SimulationResult(
            ticks=snapshots,
            total_ticks=len(snapshots),
            all_reached_goal=all(self.reached_goal.values()),
            total_collisions=total_collisions,
            per_robot_stats=per_robot,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_goal(self, rid: int) -> Position:
        """Look up a robot's goal from grid.robot_configs."""
        for cfg in self.grid.robot_configs:
            if cfg["id"] == rid:
                return cfg["goal"]
        # Fallback to the grid's default goal
        return self.grid.goal

    def _detect_collisions(
        self,
        positions: Dict[int, Position],
        tick: int,
    ) -> List[Dict]:
        """
        Check for:
            1. Robot-robot collisions (two robots at same cell)
            2. Robot-dynamic obstacle collisions
        """
        collisions = []
        rids = sorted(positions.keys())

        # Robot-robot
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                ri, rj = rids[i], rids[j]
                if positions[ri] == positions[rj]:
                    collisions.append({
                        "type": "robot-robot",
                        "agents": (ri, rj),
                        "pos": positions[ri],
                        "tick": tick,
                    })
                    self.collision_counts[ri] += 1
                    self.collision_counts[rj] += 1

        # Robot-dynamic obstacle
        dyn_positions = set(self.grid.get_dynamic_obstacle_positions())
        for rid in rids:
            if positions[rid] in dyn_positions:
                collisions.append({
                    "type": "robot-obstacle",
                    "agents": (rid,),
                    "pos": positions[rid],
                    "tick": tick,
                })
                self.collision_counts[rid] += 1

        return collisions

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Simulation(robots={len(self.robot_paths)}, "
            f"dynamic_obs={len(self.grid.dynamic_obstacles)}, "
            f"max_ticks={self.max_ticks})"
        )
