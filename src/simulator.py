"""
simulator.py  –  Interactive Real-Time Multi-Robot Simulator
=============================================================

TRULY DYNAMIC SIMULATION:
    • Robots re-plan paths IN REAL TIME when obstacles block them
    • You can CLICK on the grid to place/remove obstacles mid-simulation
    • Dynamic obstacles move autonomously and force robots to adapt
    • Replanning uses A* (baseline) — Phase 2 will add QAOA replanning
    • Side panel shows live stats: replans triggered, collisions, costs

CONTROLS:
    LEFT CLICK  — Place a static obstacle on a cell
    RIGHT CLICK — Remove a static obstacle / drag dynamic obstacle
    MIDDLE CLICK— Spawn a new dynamic obstacle (moves in random dir)

    SPACE       — Pause / Resume
    RIGHT ARROW — Step forward one tick (when paused)
    UP ARROW    — Speed up simulation
    DOWN ARROW  — Slow down
    R           — Reset (fresh grid, re-plan everything)
    G           — Toggle showing planned-ahead path ghost lines
    T           — Toggle trail visibility
    Q / ESC     — Quit

RUN IT:
    python simulator.py                       (default: 3 robots, 20×20)
    python simulator.py --robots 5 --size 25  (5 robots, 25×25 grid)
    python simulator.py --dynamic 8           (8 bouncing obstacles)

HOW REPLANNING WORKS:
    Each tick, every robot checks: "Is my next cell still free?"
    If blocked (obstacle moved in, or user placed one), the robot:
        1. Stops at current position
        2. Runs A* from current position -> goal on the LIVE grid
        3. Switches to the new path immediately
    This is REACTIVE planning — the robot adapts every tick.
    Phase 2 will replace step 2 with QAOA-based multi-robot replanning.

DEPENDENCIES:
    pip install pygame numpy
"""

import sys
import os
import time
import copy
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required.  Install with:  pip install pygame")
    sys.exit(1)

from grid import Grid, Position
from a_star import a_star, euclidean

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
    when blocked. This is the key difference from the old simulator:
    robots are REACTIVE, not blindly following a pre-baked path.
    """

    def __init__(self, rid: int, start: Position, goal: Position):
        self.rid = rid
        self.start = start
        self.goal = goal
        self.pos = start
        self.path = []                     # planned path from current pos -> goal
        self.trail = [start]               # positions visited so far
        self.reached_goal = False
        self.stuck_counter = 0             # ticks spent unable to move
        self.replan_count = 0              # how many times path was re-computed
        self.total_steps = 0
        self.collisions = 0

    def plan(self, grid, other_robot_positions=None):
        """
        Compute a path from current position to goal using A*.
        The grid is the LIVE grid (with current obstacle state).
        """
        # Temporarily mark other robots as obstacles to avoid them
        blocked = set()
        if other_robot_positions:
            for p in other_robot_positions:
                if p != self.pos and p != self.goal and grid.grid[p[0], p[1]] == 0:
                    grid.grid[p[0], p[1]] = 1
                    blocked.add(p)

        result = a_star(grid, start=self.pos, goal=self.goal, eight_connected=True)

        # Unblock
        for p in blocked:
            grid.grid[p[0], p[1]] = 0

        if result.success:
            self.path = result.path[1:]    # exclude current position
            self.replan_count += 1
            return True
        else:
            self.path = []
            return False

    def next_desired_pos(self):
        """What cell does the robot WANT to move to next?"""
        if self.path:
            return self.path[0]
        return self.pos

    def step(self, grid, occupied_cells):
        """
        Attempt to move one cell along the planned path.
        Returns True if moved, False if blocked/stuck.
        """
        if self.reached_goal:
            return True

        if not self.path:
            self.stuck_counter += 1
            return False

        next_pos = self.path[0]

        # Check if next cell is free RIGHT NOW
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
            # BLOCKED — need to replan
            self.stuck_counter += 1
            return False

    def needs_replan(self, grid, occupied_cells):
        """
        Does this robot need to recompute its path?
        Yes if: no path, or next cell is blocked, or stuck too long.
        """
        if self.reached_goal:
            return False
        if not self.path:
            return True
        # Check if any cell in the first few steps is now blocked
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
# Main Simulator
# ──────────────────────────────────────────────────────────────────

class RealtimeSimulator:
    """
    Interactive real-time simulator with:
        - Click-to-place obstacles
        - Dynamic obstacles that move autonomously
        - Robots that re-plan when blocked
    """

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

        # Window dimensions
        self.grid_px_w = grid.size * cell_size
        self.grid_px_h = grid.size * cell_size
        self.panel_width = 290
        self.win_w = self.grid_px_w + self.panel_width
        self.win_h = max(self.grid_px_h, 580)

        # Create robot agents from grid config
        self.robots = []
        for cfg in grid.robot_configs:
            agent = RobotAgent(cfg["id"], cfg["start"], cfg["goal"])
            self.robots.append(agent)

        # Sim state
        self.tick = 0
        self.paused = True
        self.finished = False
        self.show_ghost = True
        self.show_trail = True
        self.collision_flash_cells = []
        self.total_collisions = 0
        self.total_replans = 0
        self.log_lines = []

        # Mouse state for obstacle interaction
        self.dragging_obs_idx = None

        # Initial planning
        self._log("Simulation ready. Press SPACE to start.")
        self._log("Left-click to place walls.")
        self._log("Middle-click to spawn moving obstacles.")
        self._initial_plan()

    # ──────────────────────────────────────────────────────────
    # Planning
    # ──────────────────────────────────────────────────────────

    def _initial_plan(self):
        """Plan paths for all robots at the start."""
        other_positions = []
        for agent in self.robots:
            success = agent.plan(self.grid, other_positions)
            if success:
                self._log(f"R{agent.rid}: planned ({len(agent.path)} steps)")
                other_positions.append(agent.pos)
            else:
                self._log(f"R{agent.rid}: NO PATH FOUND!")

    def _replan_if_needed(self):
        """Check every robot and re-plan those that need it."""
        occupied = set()
        for a in self.robots:
            if not a.reached_goal:
                occupied.add(a.pos)

        for agent in self.robots:
            if agent.needs_replan(self.grid, occupied - {agent.pos}):
                other_pos = [a.pos for a in self.robots
                             if a.rid != agent.rid and not a.reached_goal]
                old_count = agent.replan_count
                success = agent.plan(self.grid, other_pos)
                if success and agent.replan_count > old_count:
                    self.total_replans += 1
                    self._log(f"R{agent.rid}: REPLANNED ({len(agent.path)} steps)")
                elif not success:
                    self._log(f"R{agent.rid}: stuck, no path!")

    # ──────────────────────────────────────────────────────────
    # Simulation tick
    # ──────────────────────────────────────────────────────────

    def _advance_tick(self):
        """One simulation timestep."""
        if self.finished:
            return
        self.tick += 1

        # 1. Move dynamic obstacles
        self.grid.step_dynamic_obstacles()

        # 2. Check & replan
        self._replan_if_needed()

        # 3. Move robots (priority order — lower ID goes first)
        occupied = set()
        for agent in sorted(self.robots, key=lambda a: a.rid):
            if agent.reached_goal:
                occupied.add(agent.pos)
                continue
            moved = agent.step(self.grid, occupied)
            occupied.add(agent.pos)

            if agent.reached_goal:
                self._log(f"R{agent.rid}: REACHED GOAL at tick {self.tick}!")

        # 4. Collision detection
        self._detect_collisions()

        # 5. Update flash countdowns
        self.collision_flash_cells = [
            (pos, cd - 1) for pos, cd in self.collision_flash_cells if cd > 1
        ]

        # 6. Check if all done
        if all(a.reached_goal for a in self.robots):
            self.finished = True
            self._log("ALL ROBOTS REACHED GOALS!")

    def _detect_collisions(self):
        """Check robot-robot and robot-dynamic_obstacle collisions."""
        positions = {a.rid: a.pos for a in self.robots if not a.reached_goal}
        rids = list(positions.keys())

        # Robot-robot
        for i in range(len(rids)):
            for j in range(i + 1, len(rids)):
                if positions[rids[i]] == positions[rids[j]]:
                    self.total_collisions += 1
                    self.collision_flash_cells.append((positions[rids[i]], 10))
                    for agent in self.robots:
                        if agent.rid in (rids[i], rids[j]):
                            agent.collisions += 1
                    self._log(f"COLLISION: R{rids[i]} & R{rids[j]} at {positions[rids[i]]}")

        # Robot-dynamic obstacle
        dyn_set = set(self.grid.get_dynamic_obstacle_positions())
        for rid, pos in positions.items():
            if pos in dyn_set:
                self.total_collisions += 1
                self.collision_flash_cells.append((pos, 10))
                for agent in self.robots:
                    if agent.rid == rid:
                        agent.collisions += 1
                self._log(f"COLLISION: R{rid} hit obstacle at {pos}")

    # ──────────────────────────────────────────────────────────
    # Mouse interaction
    # ──────────────────────────────────────────────────────────

    def _handle_mouse(self, event):
        """Handle mouse clicks on the grid."""
        mx, my = event.pos
        if mx >= self.grid_px_w:
            return

        c = mx // self.cell_size
        r = my // self.cell_size
        if not (0 <= r < self.grid.size and 0 <= c < self.grid.size):
            return

        pos = (r, c)

        # Don't allow placing on robot positions or start/goal
        protected = set()
        for a in self.robots:
            protected.add(a.pos)
            protected.add(a.start)
            protected.add(a.goal)
        if pos in protected:
            return

        if event.button == 1:  # LEFT CLICK — toggle static obstacle
            if self.grid.grid[r, c] == 0:
                self.grid.grid[r, c] = 1
                self._log(f"Placed wall at ({r},{c})")
            else:
                self.grid.grid[r, c] = 0
                self._log(f"Removed wall at ({r},{c})")

        elif event.button == 3:  # RIGHT CLICK — remove or drag
            if self.grid.grid[r, c] == 1:
                self.grid.grid[r, c] = 0
                self._log(f"Removed wall at ({r},{c})")
            else:
                for i, obs in enumerate(self.grid.dynamic_obstacles):
                    if tuple(obs["pos"]) == pos:
                        self.dragging_obs_idx = i
                        self._log(f"Dragging obstacle {i}")
                        return

        elif event.button == 2:  # MIDDLE CLICK — spawn dynamic obstacle
            if self.grid.grid[r, c] == 0:
                rng = np.random.RandomState()
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                d = dirs[rng.randint(0, 4)]
                self.grid.add_dynamic_obstacle(pos, direction=d, pattern="bounce")
                self._log(f"Spawned moving obstacle at ({r},{c})")

    def _handle_mouse_motion(self, event):
        if self.dragging_obs_idx is None:
            return
        mx, my = event.pos
        c = mx // self.cell_size
        r = my // self.cell_size
        if 0 <= r < self.grid.size and 0 <= c < self.grid.size:
            if self.grid.grid[r, c] == 0:
                self.grid.dynamic_obstacles[self.dragging_obs_idx]["pos"] = [r, c]

    def _handle_mouse_up(self, event):
        if self.dragging_obs_idx is not None:
            self._log(f"Dropped obstacle {self.dragging_obs_idx}")
            self.dragging_obs_idx = None

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    def run(self):
        """Open window and run."""
        pygame.init()
        screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("consolas", 13)
        font_bold = pygame.font.SysFont("consolas", 14, bold=True)
        font_big = pygame.font.SysFont("consolas", 17, bold=True)
        font_log = pygame.font.SysFont("consolas", 11)

        accumulator = 0.0
        last_time = time.time()

        running = True
        while running:
            now = time.time()
            dt = now - last_time
            last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event)

            # Tick logic
            if not self.paused and not self.finished:
                accumulator += dt
                interval = 1.0 / self.ticks_per_sec
                while accumulator >= interval:
                    accumulator -= interval
                    self._advance_tick()

            # Draw
            screen.fill(WHITE_WARM)
            self._draw_grid(screen)
            self._draw_dynamic_obstacles(screen)
            if self.show_trail:
                self._draw_trails(screen)
            if self.show_ghost:
                self._draw_ghost_paths(screen)
            self._draw_start_goal(screen)
            self._draw_robots(screen)
            self._draw_collision_flash(screen)
            self._draw_hover_highlight(screen)
            self._draw_panel(screen, font, font_bold, font_big, font_log)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    # ──────────────────────────────────────────────────────────
    # Key handling
    # ──────────────────────────────────────────────────────────

    def _handle_key(self, key):
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            self._log("PAUSED" if self.paused else "RESUMED")
        elif key == pygame.K_RIGHT and self.paused:
            self._advance_tick()
        elif key == pygame.K_UP:
            self.ticks_per_sec = min(30, self.ticks_per_sec + 1)
        elif key == pygame.K_DOWN:
            self.ticks_per_sec = max(0.5, self.ticks_per_sec - 1)
        elif key == pygame.K_r:
            self._reset()
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
        self._log("RESET. Press SPACE to start.")
        self._initial_plan()

    # ──────────────────────────────────────────────────────────
    # Drawing functions
    # ──────────────────────────────────────────────────────────

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
                # Direction arrow
                dr, dc = obs["direction"]
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2
                ex = cx + dc * (cs // 3)
                ey = cy + dr * (cs // 3)
                pygame.draw.line(screen, WHITE, (cx, cy), (int(ex), int(ey)), 2)
                # X mark to distinguish from static
                s = 4
                pygame.draw.line(screen, WHITE, (cx-s, cy-s), (cx+s, cy+s), 1)
                pygame.draw.line(screen, WHITE, (cx+s, cy-s), (cx-s, cy+s), 1)

    def _draw_trails(self, screen):
        cs = self.cell_size
        trail_surf = pygame.Surface((self.grid_px_w, self.grid_px_h), pygame.SRCALPHA)
        for agent in self.robots:
            if len(agent.trail) < 2:
                continue
            color = ROBOT_TRAIL_COLORS[agent.rid % len(ROBOT_TRAIL_COLORS)]
            points = [(c * cs + cs//2, r * cs + cs//2) for r, c in agent.trail]
            if len(points) >= 2:
                pygame.draw.lines(trail_surf, color, False, points, 3)
        screen.blit(trail_surf, (0, 0))

    def _draw_ghost_paths(self, screen):
        """Draw semi-transparent planned-ahead paths."""
        cs = self.cell_size
        ghost_surf = pygame.Surface((self.grid_px_w, self.grid_px_h), pygame.SRCALPHA)
        for agent in self.robots:
            if agent.reached_goal or not agent.path:
                continue
            color = ROBOT_GHOST_COLORS[agent.rid % len(ROBOT_GHOST_COLORS)]
            full_path = [agent.pos] + agent.path
            points = [(c * cs + cs//2, r * cs + cs//2) for r, c in full_path]
            if len(points) >= 2:
                pygame.draw.lines(ghost_surf, color, False, points, 4)
            # Small dot at planned destination
            if agent.path:
                gr, gc = agent.path[-1]
                pygame.draw.circle(ghost_surf, color, (gc*cs+cs//2, gr*cs+cs//2), 5)
        screen.blit(ghost_surf, (0, 0))

    def _draw_start_goal(self, screen):
        cs = self.cell_size
        for agent in self.robots:
            color = ROBOT_COLORS[agent.rid % len(ROBOT_COLORS)]

            # Start — hollow circle
            sr, sc = agent.start
            cx, cy = sc * cs + cs//2, sr * cs + cs//2
            pygame.draw.circle(screen, color, (cx, cy), cs//3, 2)

            # Goal — filled diamond
            gr, gc = agent.goal
            gx, gy = gc * cs + cs//2, gr * cs + cs//2
            s = cs // 3
            diamond = [(gx, gy-s), (gx+s, gy), (gx, gy+s), (gx-s, gy)]
            pygame.draw.polygon(screen, color, diamond)
            pygame.draw.polygon(screen, BLACK, diamond, 2)

            # Labels
            self._label(screen, f"S{agent.rid}", cx, cy, BLACK, 10)
            self._label(screen, f"G{agent.rid}", gx, gy, WHITE, 10)

    def _draw_robots(self, screen):
        cs = self.cell_size
        for agent in self.robots:
            r, c = agent.pos
            color = ROBOT_COLORS[agent.rid % len(ROBOT_COLORS)]
            cx, cy = c * cs + cs//2, r * cs + cs//2
            radius = cs // 2 - 3

            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, BLACK, (cx, cy), radius, 2)
            self._label(screen, str(agent.rid), cx, cy, WHITE, 13)

            if agent.reached_goal:
                pygame.draw.circle(screen, GREEN, (cx, cy), radius + 4, 3)

            if agent.stuck_counter >= 3 and not agent.reached_goal:
                pulse = 2 + abs(math.sin(time.time() * 5)) * 3
                pygame.draw.circle(screen, ORANGE, (cx, cy), int(radius + pulse), 2)

    def _draw_collision_flash(self, screen):
        cs = self.cell_size
        for pos, cd in self.collision_flash_cells:
            r, c = pos
            flash_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            alpha = min(180, cd * 20)
            flash_surf.fill((255, 0, 0, alpha))
            screen.blit(flash_surf, (c * cs, r * cs))

    def _draw_hover_highlight(self, screen):
        """Highlight cell under mouse cursor."""
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

    # ──────────────────────────────────────────────────────────
    # Info panel
    # ──────────────────────────────────────────────────────────

    def _draw_panel(self, screen, font, font_bold, font_big, font_log):
        x0 = self.grid_px_w + 12
        y = 10

        # Panel background
        pygame.draw.rect(screen, PANEL_BG,
                         (self.grid_px_w, 0, self.panel_width, self.win_h))
        pygame.draw.line(screen, GREY_MED,
                         (self.grid_px_w, 0), (self.grid_px_w, self.win_h), 2)

        # Title
        screen.blit(font_big.render("MULTI-ROBOT SIM", True, BLACK), (x0, y)); y += 25

        # Status
        if self.finished:
            status, scol = "COMPLETE", GREEN_DARK
        elif self.paused:
            status, scol = "PAUSED", ORANGE
        else:
            status, scol = "RUNNING", GREEN
        screen.blit(font_bold.render(f"Status: {status}", True, scol), (x0, y)); y += 20
        screen.blit(font.render(f"Tick: {self.tick}", True, BLACK), (x0, y)); y += 18
        screen.blit(font.render(f"Speed: {self.ticks_per_sec:.1f} t/s", True, BLACK), (x0, y)); y += 18
        screen.blit(font.render(f"Replans: {self.total_replans}", True, (0, 100, 180)), (x0, y)); y += 18
        col_color = RED if self.total_collisions > 0 else BLACK
        screen.blit(font.render(f"Collisions: {self.total_collisions}", True, col_color), (x0, y)); y += 18
        screen.blit(font.render(f"Dyn obstacles: {len(self.grid.dynamic_obstacles)}", True, BLACK), (x0, y)); y += 25

        # Separator
        pygame.draw.line(screen, GREY_LIGHT, (x0, y), (x0 + self.panel_width - 24, y)); y += 8

        # Robots
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
            screen.blit(font.render(
                f"    plans:{agent.replan_count} col:{agent.collisions}",
                True, GREY_DARK), (x0 + 18, y))
            y += 20

        y += 8
        pygame.draw.line(screen, GREY_LIGHT, (x0, y), (x0 + self.panel_width - 24, y)); y += 8

        # Controls
        screen.blit(font_big.render("CONTROLS", True, BLACK), (x0, y)); y += 22
        ctrls = [
            ("SPACE",       "Pause / Resume"),
            ("RIGHT",       "Step (paused)"),
            ("UP / DOWN",   "Speed up / down"),
            ("R",           "Reset all"),
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

        # Event log
        screen.blit(font_big.render("EVENT LOG", True, BLACK), (x0, y)); y += 20
        max_log_lines = max(1, (self.win_h - y - 10) // 14)
        visible = self.log_lines[-max_log_lines:]
        for line in visible:
            color = RED if "COLLISION" in line else (
                (0, 100, 180) if "REPLAN" in line else BLACK
            )
            screen.blit(font_log.render(line, True, color), (x0, y))
            y += 14

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# Launcher
# ──────────────────────────────────────────────────────────────────

def launch_simulator(
    size=20,
    num_robots=3,
    num_dynamic=4,
    obstacle_ratio=0.12,
    seed=42,
    cell_size=32,
):
    """Create environment and launch the interactive simulator."""
    print("=" * 55)
    print("  INTERACTIVE MULTI-ROBOT SIMULATOR")
    print("=" * 55)

    print(f"\n[env]  {size}x{size} grid, {num_robots} robots, "
          f"{num_dynamic} dynamic obstacles, seed={seed}")

    grid = Grid.create_multi_robot_env(
        size=size,
        obstacle_ratio=obstacle_ratio,
        num_robots=num_robots,
        num_dynamic=num_dynamic,
        seed=seed,
    )

    for cfg in grid.robot_configs:
        print(f"  Robot {cfg['id']}: {cfg['start']} -> {cfg['goal']}")
    print(f"  Dynamic obstacles: {len(grid.dynamic_obstacles)}")

    print("\n[sim]  Launching interactive simulator...")
    print("       Left-click to place walls, middle-click for dynamic obstacles")
    print("       Press SPACE to start, Q/ESC to exit\n")

    sim = RealtimeSimulator(
        grid=grid,
        cell_size=cell_size,
        ticks_per_sec=3.0,
        title=f"Interactive Simulator ({num_robots} robots, {size}x{size})",
    )
    sim.run()
    print("\n[sim]  Simulator closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive multi-robot simulator")
    parser.add_argument("--size", type=int, default=20, help="Grid size (default: 20)")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots (2-5)")
    parser.add_argument("--dynamic", type=int, default=4, help="Dynamic obstacles (default: 4)")
    parser.add_argument("--obstacles", type=float, default=0.12, help="Static obstacle ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cell-size", type=int, default=32, help="Pixels per cell")
    args = parser.parse_args()

    launch_simulator(
        size=args.size,
        num_robots=args.robots,
        num_dynamic=args.dynamic,
        obstacle_ratio=args.obstacles,
        seed=args.seed,
        cell_size=args.cell_size,
    )
