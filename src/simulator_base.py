"""
simulator_base.py  –  Shared Base for Interactive Simulators
=============================================================

Contains the core RealtimeSimulator class and RobotAgent that both
simulator_classical.py and simulator_quantum.py build on.

DO NOT RUN THIS FILE DIRECTLY. Use:
    python simulator_classical.py   — A*, APF, Q-Learning
    python simulator_quantum.py     — Quantum QL, Hybrid, QAOA
"""

import sys
import os
import time
import copy
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required.  Install with:  pip install pygame")
    sys.exit(1)

from grid import Grid, Position
from a_star import a_star, euclidean
from apf import apf_plan
from q_learning import train_q_learning, q_learning_plan, ACTIONS
from quantum_q_learning import (
    train_quantum_q_learning, quantum_q_learning_plan,
    ACTIONS as Q_ACTIONS,
)
from hybrid_planner import hybrid_plan
from multi_robot import generate_candidate_paths, compute_conflict_matrix
try:
    from qaoa_optimizer import qaoa_select
    QAOA_AVAILABLE = True
except ImportError:
    QAOA_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────────

WHITE       = (255, 255, 255)
WHITE_WARM  = (252, 250, 245)
BLACK       = (30,  30,  30)
GREY_LIGHT  = (210, 210, 210)
GREY_MED    = (160, 160, 160)
GREY_DARK   = (70,  70,  70)
RED         = (220, 55,  55)
RED_LIGHT   = (255, 120, 120)
GREEN       = (50,  200, 80)
GREEN_DARK  = (30,  150, 60)
YELLOW      = (255, 220, 50)
ORANGE      = (255, 165, 30)
PANEL_BG    = (240, 240, 245)

ROBOT_COLORS = [
    (51,  153, 255),  # blue
    (255, 102, 51),   # orange
    (51,  204, 102),  # green
    (204, 51,  255),  # purple
    (255, 204, 0),    # yellow
]

ROBOT_TRAIL_COLORS = [
    (140, 190, 255, 120),
    (255, 175, 130, 120),
    (130, 235, 165, 120),
    (230, 150, 255, 120),
    (255, 235, 110, 120),
]

ROBOT_GHOST_COLORS = [
    (51,  153, 255, 60),
    (255, 102, 51,  60),
    (51,  204, 102, 60),
    (204, 51,  255, 60),
    (255, 204, 0,   60),
]


# ──────────────────────────────────────────────────────────────────
# Robot agent (each robot is an independent agent)
# ──────────────────────────────────────────────────────────────────

class RobotAgent:
    """
    An individual robot that moves toward its goal and can re-plan
    when blocked.  Robots are REACTIVE — they adapt every tick.
    """

    def __init__(self, rid: int, start: Position, goal: Position):
        self.rid = rid
        self.start = start
        self.goal = goal
        self.pos = start
        self.path = []
        self.trail = [start]
        self.reached_goal = False
        self.stuck_counter = 0
        self.replan_count = 0
        self.total_steps = 0
        self.collisions = 0
        self.q_table = None
        self.q_angles = None
        self.hybrid_stats = None

    def plan(self, grid, other_robot_positions=None, planner="A*"):
        """Compute a path from current position to goal on the LIVE grid."""
        blocked = set()
        if other_robot_positions:
            for p in other_robot_positions:
                if p != self.pos and p != self.goal and grid.grid[p[0], p[1]] == 0:
                    grid.grid[p[0], p[1]] = 1
                    blocked.add(p)

        success = False

        if planner == "APF":
            result = apf_plan(
                grid, start=self.pos, goal=self.goal,
                max_steps=200, eight_connected=True,
                k_rep=50.0, influence_radius=2.5,
            )
            if result.path and len(result.path) > 1:
                self.path = result.path[1:]
                self.replan_count += 1
                success = True
            else:
                self.path = []

        elif planner == "Q-Learning":
            if self.q_table is not None:
                result = q_learning_plan(
                    grid, self.q_table,
                    start=self.pos, goal=self.goal,
                    max_steps=200,
                )
                if result.path and len(result.path) > 1:
                    self.path = result.path[1:]
                    self.replan_count += 1
                    success = True
                else:
                    self.path = []
            else:
                result = a_star(grid, start=self.pos, goal=self.goal,
                                eight_connected=True)
                if result.success:
                    self.path = result.path[1:]
                    self.replan_count += 1
                    success = True
                else:
                    self.path = []

        elif planner == "Quantum QL":
            if self.q_angles is not None:
                result = quantum_q_learning_plan(
                    grid, self.q_angles,
                    start=self.pos, goal=self.goal,
                    max_steps=200,
                )
                if result.path and len(result.path) > 1:
                    self.path = result.path[1:]
                    self.replan_count += 1
                    success = True
                else:
                    self.path = []
            else:
                result = a_star(grid, start=self.pos, goal=self.goal,
                                eight_connected=True)
                if result.success:
                    self.path = result.path[1:]
                    self.replan_count += 1
                    success = True
                else:
                    self.path = []

        elif planner == "Hybrid":
            result = hybrid_plan(
                grid, start=self.pos, goal=self.goal,
                max_steps=300, q_episodes=300, q_seed=42 + self.rid,
                q_angles=self.q_angles,
            )
            if result.path and len(result.path) > 1:
                self.path = result.path[1:]
                self.replan_count += 1
                self.hybrid_stats = {
                    "astar_replans": result.astar_replans,
                    "apf_steps": result.apf_steps,
                    "quantum_escapes": result.quantum_escapes,
                    "stuck_detected": result.stuck_detected,
                }
                success = True
            else:
                self.path = []

        else:  # Default: A*
            result = a_star(grid, start=self.pos, goal=self.goal,
                            eight_connected=True)
            if result.success:
                self.path = result.path[1:]
                self.replan_count += 1
                success = True
            else:
                self.path = []

        for p in blocked:
            grid.grid[p[0], p[1]] = 0
        return success

    def next_desired_pos(self):
        if self.path:
            return self.path[0]
        return self.pos

    def step(self, grid, occupied_cells):
        if self.reached_goal:
            return True
        if not self.path:
            self.stuck_counter += 1
            return False

        next_pos = self.path[0]
        if grid.is_free(next_pos) and next_pos not in occupied_cells:
            self.pos = next_pos
            self.path.pop(0)
            self.trail.append(self.pos)
            self.total_steps += 1
            self.stuck_counter = 0
            if self.pos == self.goal:
                self.reached_goal = True
            return True
        else:
            self.stuck_counter += 1
            return False

    def needs_replan(self, grid, occupied_cells):
        if self.reached_goal:
            return False
        if not self.path:
            return True
        lookahead = min(5, len(self.path))
        for i in range(lookahead):
            p = self.path[i]
            if not grid.is_free(p):
                return True
            if p in occupied_cells and p != self.goal:
                return True
        if self.stuck_counter >= 3:
            return True
        return False


# ──────────────────────────────────────────────────────────────────
# Base Simulator
# ──────────────────────────────────────────────────────────────────

class RealtimeSimulator:
    """
    Interactive real-time simulator base class.
    Subclasses define which planners are available.
    """

    # Subclasses MUST override this
    PLANNER_NAMES = ["A*"]

    def __init__(
        self,
        grid,
        cell_size=32,
        ticks_per_sec=3.0,
        title="Multi-Robot Simulator",
    ):
        self.base_grid = grid
        self.grid = copy.deepcopy(grid)
        self.cell_size = cell_size
        self.ticks_per_sec = ticks_per_sec
        self.title = title

        self.grid_px_w = grid.size * cell_size
        self.grid_px_h = grid.size * cell_size
        self.panel_width = 290
        self.win_w = self.grid_px_w + self.panel_width
        self.win_h = max(self.grid_px_h, 580)

        self.robots = []
        for cfg in grid.robot_configs:
            agent = RobotAgent(cfg["id"], cfg["start"], cfg["goal"])
            self.robots.append(agent)

        self.tick = 0
        self.paused = True
        self.finished = False
        self.show_ghost = True
        self.show_trail = True
        self.collision_flash_cells = []
        self.total_collisions = 0
        self.total_replans = 0
        self.log_lines = []

        self.planner_idx = 0
        self.planner_name = self.PLANNER_NAMES[0]

        self.dragging_obs_idx = None

        self._log("Simulation ready. Press SPACE to start.")
        self._log("Left-click to place walls.")
        self._log("Middle-click to spawn moving obstacles.")
        self._log(f"Planner: {self.planner_name} (press P to cycle)")
        self._initial_plan()

    # ── Planner hooks (override in subclasses) ──────────────────

    def _initial_plan(self):
        """Plan paths for all robots at the start. Override in subclass."""
        other_positions = []
        for agent in self.robots:
            success = agent.plan(self.grid, other_positions,
                                planner=self.planner_name)
            if success:
                self._log(f"R{agent.rid}: planned ({len(agent.path)} steps)")
                other_positions.append(agent.pos)
            else:
                self._log(f"R{agent.rid}: NO PATH FOUND!")

    def _on_cycle_planner(self):
        """Called after planner name changes. Override for training."""
        pass

    def _cycle_planner(self):
        """Cycle through available planners."""
        self.planner_idx = (self.planner_idx + 1) % len(self.PLANNER_NAMES)
        self.planner_name = self.PLANNER_NAMES[self.planner_idx]
        self._log(f"Switched planner to: {self.planner_name}")
        self._on_cycle_planner()
        for agent in self.robots:
            if not agent.reached_goal:
                other_pos = [a.pos for a in self.robots
                             if a.rid != agent.rid and not a.reached_goal]
                agent.plan(self.grid, other_pos, planner=self.planner_name)

    # ── Replanning ──────────────────────────────────────────────

    def _replan_if_needed(self):
        occupied = set()
        for a in self.robots:
            if not a.reached_goal:
                occupied.add(a.pos)

        needs_replan = False
        for agent in self.robots:
            if agent.needs_replan(self.grid, occupied - {agent.pos}):
                needs_replan = True
                break

        if needs_replan and self.planner_name == "QAOA":
            self.total_replans += 1
            self._plan_qaoa()
            return

        for agent in self.robots:
            if agent.needs_replan(self.grid, occupied - {agent.pos}):
                other_pos = [a.pos for a in self.robots
                             if a.rid != agent.rid and not a.reached_goal]
                old_count = agent.replan_count
                success = agent.plan(self.grid, other_pos,
                                     planner=self.planner_name)
                if success and agent.replan_count > old_count:
                    self.total_replans += 1
                    self._log(f"R{agent.rid}: REPLANNED ({len(agent.path)} steps)")
                elif not success:
                    self._log(f"R{agent.rid}: stuck, no path!")

    def _plan_qaoa(self):
        """QAOA coordinated replanning (used by quantum simulator)."""
        if not QAOA_AVAILABLE:
            self._log("QAOA: scipy not installed, falling back to A*")
            for agent in self.robots:
                if not agent.reached_goal:
                    agent.plan(self.grid, [], planner="A*")
            return
        self._log("QAOA: generating candidate paths...")
        all_candidates = {}
        for agent in self.robots:
            cands = generate_candidate_paths(
                self.grid, start=agent.pos, goal=agent.goal,
                num_candidates=4, seed=42 + agent.rid,
            )
            all_candidates[agent.rid] = cands
            self._log(f"  R{agent.rid}: {len(cands)} candidates")

        if not all(len(c) > 0 for c in all_candidates.values()):
            self._log("QAOA: some robots have no paths, falling back to A*")
            for agent in self.robots:
                agent.plan(self.grid, [], planner="A*")
            return

        self._log("QAOA: optimizing path selection...")
        selection = qaoa_select(
            all_candidates, conflict_penalty=10.0,
            p=1, num_restarts=2, seed=42,
        )
        for agent in self.robots:
            rid = agent.rid
            chosen_idx = selection.get(rid, 0)
            path = all_candidates[rid][chosen_idx]["path"]
            if path and path[0] == agent.pos:
                agent.path = path[1:]
            else:
                agent.path = path
            agent.replan_count += 1
            self._log(f"  R{rid}: QAOA picked path {chosen_idx} "
                      f"({len(agent.path)} steps)")
        self._log("QAOA: selection complete.")

    # ── Tick logic ──────────────────────────────────────────────

    def _advance_tick(self):
        if self.finished:
            return
        self.tick += 1
        self.grid.step_dynamic_obstacles()
        self._replan_if_needed()

        occupied = set()
        for agent in sorted(self.robots, key=lambda a: a.rid):
            if agent.reached_goal:
                occupied.add(agent.pos)
                continue
            agent.step(self.grid, occupied)
            occupied.add(agent.pos)
            if agent.reached_goal:
                self._log(f"R{agent.rid}: REACHED GOAL at tick {self.tick}!")

        self._detect_collisions()
        self.collision_flash_cells = [
            (pos, cd - 1) for pos, cd in self.collision_flash_cells if cd > 1
        ]
        if all(a.reached_goal for a in self.robots):
            self.finished = True
            self._log("ALL ROBOTS REACHED GOALS!")

    def _detect_collisions(self):
        positions = {a.rid: a.pos for a in self.robots if not a.reached_goal}
        rids = list(positions.keys())
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                if positions[rids[i]] == positions[rids[j]]:
                    self.total_collisions += 1
                    self.collision_flash_cells.append((positions[rids[i]], 10))
                    for agent in self.robots:
                        if agent.rid in (rids[i], rids[j]):
                            agent.collisions += 1
                    self._log(f"COLLISION: R{rids[i]} & R{rids[j]} at {positions[rids[i]]}")

        dyn_set = set(self.grid.get_dynamic_obstacle_positions())
        for rid, pos in positions.items():
            if pos in dyn_set:
                self.total_collisions += 1
                self.collision_flash_cells.append((pos, 10))
                for agent in self.robots:
                    if agent.rid == rid:
                        agent.collisions += 1
                self._log(f"COLLISION: R{rid} hit obstacle at {pos}")

    # ── Mouse interaction ───────────────────────────────────────

    def _handle_mouse(self, event):
        mx, my = event.pos
        if mx >= self.grid_px_w:
            return
        c = mx // self.cell_size
        r = my // self.cell_size
        if not (0 <= r < self.grid.size and 0 <= c < self.grid.size):
            return
        pos = (r, c)

        special = set()
        for cfg in self.grid.robot_configs:
            special.add(tuple(cfg["start"]))
            special.add(tuple(cfg["goal"]))
        if pos in special:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # LEFT CLICK — toggle wall
                if self.grid.grid[r, c] == 1:
                    self.grid.grid[r, c] = 0
                    self.base_grid.grid[r, c] = 0
                    self._log(f"Removed wall at {pos}")
                else:
                    self.grid.grid[r, c] = 1
                    self.base_grid.grid[r, c] = 1
                    self._log(f"Placed wall at {pos}")

            elif event.button == 3:  # RIGHT CLICK — drag/remove dynamic obs
                for i, obs in enumerate(self.grid.dynamic_obstacles):
                    if tuple(obs["pos"]) == pos:
                        self.dragging_obs_idx = i
                        return
                for i, obs in enumerate(self.grid.dynamic_obstacles):
                    or_, oc_ = obs["pos"]
                    if abs(or_ - r) <= 1 and abs(oc_ - c) <= 1:
                        self.grid.dynamic_obstacles.pop(i)
                        self.base_grid.dynamic_obstacles = copy.deepcopy(
                            self.grid.dynamic_obstacles)
                        self._log(f"Removed dynamic obstacle near {pos}")
                        return

            elif event.button == 2:  # MIDDLE CLICK — spawn dynamic obstacle
                import random
                d = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                self.grid.add_dynamic_obstacle(pos, direction=d, pattern="bounce")
                self.base_grid.add_dynamic_obstacle(pos, direction=d, pattern="bounce")
                self._log(f"Spawned dynamic obstacle at {pos}")

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_obs_idx = None

        elif event.type == pygame.MOUSEMOTION and self.dragging_obs_idx is not None:
            idx = self.dragging_obs_idx
            if idx < len(self.grid.dynamic_obstacles):
                self.grid.dynamic_obstacles[idx]["pos"] = [r, c]

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE or key == pygame.K_q:
            return False
        elif key == pygame.K_SPACE:
            if self.finished:
                self._reset()
            else:
                self.paused = not self.paused
                self._log("PAUSED" if self.paused else "RESUMED")
        elif key == pygame.K_RIGHT and self.paused:
            self._advance_tick()
        elif key == pygame.K_UP:
            self.ticks_per_sec = min(30, self.ticks_per_sec * 1.3)
        elif key == pygame.K_DOWN:
            self.ticks_per_sec = max(0.5, self.ticks_per_sec / 1.3)
        elif key == pygame.K_r:
            self._reset()
        elif key == pygame.K_p:
            self._cycle_planner()
        elif key == pygame.K_g:
            self.show_ghost = not self.show_ghost
        elif key == pygame.K_t:
            self.show_trail = not self.show_trail
        return True

    def _reset(self):
        self.grid = copy.deepcopy(self.base_grid)
        self.tick = 0
        self.paused = True
        self.finished = False
        self.total_collisions = 0
        self.total_replans = 0
        self.collision_flash_cells = []
        self.log_lines = []
        for agent in self.robots:
            agent.pos = agent.start
            agent.path = []
            agent.trail = [agent.start]
            agent.reached_goal = False
            agent.stuck_counter = 0
            agent.replan_count = 0
            agent.total_steps = 0
            agent.collisions = 0
            agent.q_table = None
            agent.q_angles = None
            agent.hybrid_stats = None
        self._log("RESET. Press SPACE to start.")
        self._initial_plan()

    # ── Drawing ─────────────────────────────────────────────────

    def _draw_grid(self, screen):
        cs = self.cell_size
        for r in range(self.grid.size):
            for c in range(self.grid.size):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                if self.grid.grid[r, c] == 1:
                    pygame.draw.rect(screen, GREY_DARK, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, GREY_LIGHT, rect, 1)

    def _draw_dynamic_obstacles(self, screen):
        cs = self.cell_size
        for obs in self.grid.dynamic_obstacles:
            r, c = obs["pos"]
            if 0 <= r < self.grid.size and 0 <= c < self.grid.size:
                rect = pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4)
                pygame.draw.rect(screen, RED, rect, border_radius=4)
                dr, dc = obs["direction"]
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2
                ex = cx + dc * (cs // 3)
                ey = cy + dr * (cs // 3)
                pygame.draw.line(screen, WHITE, (cx, cy), (int(ex), int(ey)), 2)
                s = 4
                pygame.draw.line(screen, WHITE,
                                 (cx - s, cy - s), (cx + s, cy + s), 2)
                pygame.draw.line(screen, WHITE,
                                 (cx - s, cy + s), (cx + s, cy - s), 2)

    def _draw_goals(self, screen):
        cs = self.cell_size
        for i, cfg in enumerate(self.grid.robot_configs):
            r, c = cfg["goal"]
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            rect = pygame.Rect(c * cs + 3, r * cs + 3, cs - 6, cs - 6)
            pygame.draw.rect(screen, color, rect, 3)
            self._label(screen, "G", c * cs + cs // 2, r * cs + cs // 2,
                        color, size=11)

    def _draw_trails(self, screen):
        cs = self.cell_size
        for agent in self.robots:
            if len(agent.trail) < 2:
                continue
            tc = ROBOT_TRAIL_COLORS[agent.rid % len(ROBOT_TRAIL_COLORS)]
            trail_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            trail_surf.fill(tc)
            for pos in agent.trail[:-1]:
                screen.blit(trail_surf, (pos[1] * cs, pos[0] * cs))

    def _draw_ghost_paths(self, screen):
        cs = self.cell_size
        for agent in self.robots:
            if not agent.path:
                continue
            gc = ROBOT_GHOST_COLORS[agent.rid % len(ROBOT_GHOST_COLORS)]
            ghost_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            ghost_surf.fill(gc)
            for pos in agent.path:
                screen.blit(ghost_surf, (pos[1] * cs, pos[0] * cs))

    def _draw_robots(self, screen):
        cs = self.cell_size
        for agent in self.robots:
            color = ROBOT_COLORS[agent.rid % len(ROBOT_COLORS)]
            cx = agent.pos[1] * cs + cs // 2
            cy = agent.pos[0] * cs + cs // 2
            radius = max(cs // 3, 6)
            if agent.reached_goal:
                pygame.draw.circle(screen, GREEN_DARK, (cx, cy), radius + 2)
            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, BLACK, (cx, cy), radius, 2)
            self._label(screen, str(agent.rid), cx, cy, WHITE, size=11)

    def _draw_collision_flash(self, screen):
        cs = self.cell_size
        for (r, c), cd in self.collision_flash_cells:
            alpha = min(180, cd * 20)
            flash_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            flash_surf.fill((255, 0, 0, alpha))
            screen.blit(flash_surf, (c * cs, r * cs))

    def _draw_hover_highlight(self, screen):
        mx, my = pygame.mouse.get_pos()
        if mx >= self.grid_px_w:
            return
        c = mx // self.cell_size
        r = my // self.cell_size
        if 0 <= r < self.grid.size and 0 <= c < self.grid.size:
            cs = self.cell_size
            hover_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            hover_surf.fill((100, 100, 255, 40))
            screen.blit(hover_surf, (c * cs, r * cs))

    # ── Info panel ──────────────────────────────────────────────

    def _draw_panel(self, screen, font, font_bold, font_big, font_log):
        x0 = self.grid_px_w + 12
        y = 10

        pygame.draw.rect(screen, PANEL_BG,
                         (self.grid_px_w, 0, self.panel_width, self.win_h))
        pygame.draw.line(screen, GREY_MED,
                         (self.grid_px_w, 0), (self.grid_px_w, self.win_h), 2)

        screen.blit(font_big.render("MULTI-ROBOT SIM", True, BLACK), (x0, y)); y += 25

        if self.finished:
            status, scol = "COMPLETE", GREEN_DARK
        elif self.paused:
            status, scol = "PAUSED", ORANGE
        else:
            status, scol = "RUNNING", GREEN
        screen.blit(font_bold.render(f"Status: {status}", True, scol), (x0, y)); y += 20
        screen.blit(font_bold.render(f"Planner: {self.planner_name}", True, (0, 100, 180)), (x0, y)); y += 20
        screen.blit(font.render(f"Tick: {self.tick}", True, BLACK), (x0, y)); y += 18
        screen.blit(font.render(f"Speed: {self.ticks_per_sec:.1f} t/s", True, BLACK), (x0, y)); y += 18
        screen.blit(font.render(f"Replans: {self.total_replans}", True, (0, 100, 180)), (x0, y)); y += 18
        col_color = RED if self.total_collisions > 0 else BLACK
        screen.blit(font.render(f"Collisions: {self.total_collisions}", True, col_color), (x0, y)); y += 18
        screen.blit(font.render(f"Dyn obstacles: {len(self.grid.dynamic_obstacles)}", True, BLACK), (x0, y)); y += 25

        pygame.draw.line(screen, GREY_LIGHT, (x0, y), (x0 + self.panel_width - 24, y)); y += 8

        screen.blit(font_big.render("ROBOTS", True, BLACK), (x0, y)); y += 22
        for agent in self.robots:
            color = ROBOT_COLORS[agent.rid % len(ROBOT_COLORS)]
            pygame.draw.circle(screen, color, (x0 + 7, y + 7), 6)

            if agent.reached_goal:
                state_str = f"DONE (t={agent.total_steps})"
            elif agent.stuck_counter >= 3:
                state_str = "STUCK!"
            else:
                state_str = f"({agent.pos[0]:2d},{agent.pos[1]:2d})"

            screen.blit(font.render(
                f"R{agent.rid}: {state_str}", True, BLACK), (x0 + 18, y))
            y += 16
            extra = f"    plans:{agent.replan_count} col:{agent.collisions}"
            if agent.hybrid_stats and self.planner_name == "Hybrid":
                hs = agent.hybrid_stats
                extra += f" esc:{hs['quantum_escapes']}"
            screen.blit(font.render(extra, True, GREY_DARK), (x0 + 18, y))
            y += 20

        y += 8
        pygame.draw.line(screen, GREY_LIGHT, (x0, y), (x0 + self.panel_width - 24, y)); y += 8

        screen.blit(font_big.render("CONTROLS", True, BLACK), (x0, y)); y += 22
        ctrls = [
            ("SPACE",       "Pause / Resume"),
            ("RIGHT",       "Step (paused)"),
            ("UP / DOWN",   "Speed up / down"),
            ("R",           "Reset all"),
            ("P",           "Cycle planner"),
            ("G",           "Ghost paths on/off"),
            ("T",           "Trails on/off"),
            ("L-Click",     "Place / remove wall"),
            ("M-Click",     "Spawn moving obs"),
            ("R-Click",     "Remove / drag obs"),
            ("Q / ESC",     "Quit"),
        ]
        for key, desc in ctrls:
            screen.blit(font_bold.render(f"{key:12s}", True, GREY_DARK), (x0, y))
            screen.blit(font.render(desc, True, BLACK), (x0 + 100, y))
            y += 16

        y += 10
        pygame.draw.line(screen, GREY_LIGHT, (x0, y), (x0 + self.panel_width - 24, y)); y += 8

        screen.blit(font_big.render("EVENT LOG", True, BLACK), (x0, y)); y += 20
        max_log_lines = max(1, (self.win_h - y - 10) // 14)
        visible = self.log_lines[-max_log_lines:]
        for line in visible:
            color = RED if "COLLISION" in line else (
                (0, 100, 180) if "REPLAN" in line else BLACK
            )
            screen.blit(font_log.render(line, True, color), (x0, y))
            y += 14

    # ── Main loop ───────────────────────────────────────────────

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("consolas", 13)
        font_bold = pygame.font.SysFont("consolas", 13, bold=True)
        font_big = pygame.font.SysFont("consolas", 15, bold=True)
        font_log = pygame.font.SysFont("consolas", 11)

        last_tick_time = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type in (pygame.MOUSEBUTTONDOWN,
                                    pygame.MOUSEBUTTONUP,
                                    pygame.MOUSEMOTION):
                    self._handle_mouse(event)
                elif event.type == pygame.KEYDOWN:
                    if not self._handle_key(event.key):
                        running = False

            # Auto-advance
            now = time.time()
            if not self.paused and not self.finished:
                dt = 1.0 / self.ticks_per_sec
                if now - last_tick_time >= dt:
                    self._advance_tick()
                    last_tick_time = now

            # Draw
            screen.fill(WHITE_WARM)
            self._draw_grid(screen)
            self._draw_dynamic_obstacles(screen)
            self._draw_goals(screen)
            if self.show_trail:
                self._draw_trails(screen)
            if self.show_ghost:
                self._draw_ghost_paths(screen)
            self._draw_robots(screen)
            self._draw_collision_flash(screen)
            self._draw_hover_highlight(screen)
            self._draw_panel(screen, font, font_bold, font_big, font_log)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    # ── Utilities ───────────────────────────────────────────────

    def _label(self, screen, text, cx, cy, color, size=12):
        font = pygame.font.SysFont("consolas", size, bold=True)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)

    def _log(self, msg):
        prefix = f"[{self.tick:4d}] " if self.tick > 0 else "[init] "
        self.log_lines.append(prefix + msg)
        if len(self.log_lines) > 100:
            self.log_lines = self.log_lines[-100:]
