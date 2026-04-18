"""Realtime A* + QAOA demo simulator for paper presentation.

This interactive simulator focuses only on the paper novelty:
1) A* candidate path generation per robot
2) QAOA-based multi-robot path selection
3) Dynamic obstacle-aware replanning in realtime

Controls:
  SPACE        Pause/Resume
  N            Single-step (when paused)
  P            Force replanning now
  C            Toggle candidate path visualization
  T            Toggle robot trails
  O            Toggle planning mode (FAST/QUALITY)
  G            Add random dynamic obstacle
  R            Reset scene
  UP/DOWN      Speed up / slow down simulation
  Left Click   Toggle static wall at cursor
  Middle Click Add dynamic obstacle at cursor
  Right Click  Remove dynamic obstacle at cursor
  ESC / Q      Quit
"""

import math
import random
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame

from grid import Grid, Position
from multi_robot import generate_candidate_paths, greedy_select, random_select
from qaoa_optimizer import qaoa_select_full


Color = Tuple[int, int, int]


@dataclass
class RobotRuntime:
    rid: int
    pos: Position
    goal: Position
    color: Color
    path: List[Position]
    path_idx: int
    reached_goal: bool
    trail: List[Position]


class PaperDemoSimulator:
    BG: Color = (18, 22, 30)
    GRID_BG: Color = (28, 34, 46)
    FREE: Color = (34, 42, 56)
    WALL: Color = (130, 140, 156)
    DYN_OBS: Color = (244, 114, 182)
    GOAL: Color = (250, 204, 21)
    TXT: Color = (226, 232, 240)
    MUTE: Color = (148, 163, 184)
    PANEL: Color = (20, 28, 40)
    ALERT: Color = (251, 113, 133)

    ROBOT_COLORS: List[Color] = [
        (59, 130, 246),
        (16, 185, 129),
        (249, 115, 22),
        (168, 85, 247),
        (244, 63, 94),
    ]

    def __init__(self, size: int = 20, num_robots: int = 3, num_dynamic: int = 4, seed: int = 42):
        pygame.init()
        pygame.display.set_caption("Paper Demo: A* Candidate Generation + QAOA Coordination")

        self.size = size
        self.num_robots = num_robots
        self.num_dynamic = num_dynamic
        self.seed = seed

        self.cell = 34
        self.margin = 14
        self.sidebar_w = 380
        self.grid_px = self.size * self.cell
        self.w = self.grid_px + self.sidebar_w + self.margin * 3
        self.h = self.grid_px + self.margin * 2

        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 15)

        self.running = True
        self.paused = False
        self.show_candidates = True
        self.show_trails = True
        self.sim_hz = 3.0
        self.tick = 0

        self.plan_interval = 8
        self.plan_due = True
        self.stagnation_ticks = 0
        self.last_plan_ms = 0.0
        self.last_backend = "-"
        self.last_score = 0.0
        self.last_conflicts = 0
        self.status_text = ""

        # FAST mode keeps replanning responsive for live demo.
        self.mode = "FAST"
        self.num_candidates = 3
        self.qaoa_depth = 1
        self.qaoa_restarts = 1

        self.grid = None
        self.robots: Dict[int, RobotRuntime] = {}
        self.all_candidates: Dict[int, List[Dict]] = {}
        self.selection: Dict[int, int] = {}

        self.reset_scene()

    def reset_scene(self) -> None:
        self.grid = Grid.create_multi_robot_env(
            size=self.size,
            obstacle_ratio=0.12,
            num_robots=self.num_robots,
            num_dynamic=self.num_dynamic,
            seed=self.seed,
        )

        self.robots.clear()
        for cfg in self.grid.robot_configs:
            rid = cfg["id"]
            start = cfg["start"]
            goal = cfg["goal"]

            # Keep local start/goal neighborhoods open so every robot can move.
            for center in (start, goal):
                cr, cc = center
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc2 = cr + dr, cc + dc
                        if 0 <= rr < self.size and 0 <= cc2 < self.size:
                            self.grid.grid[rr, cc2] = 0

            self.robots[rid] = RobotRuntime(
                rid=rid,
                pos=start,
                goal=goal,
                color=self.ROBOT_COLORS[rid % len(self.ROBOT_COLORS)],
                path=[start],
                path_idx=0,
                reached_goal=False,
                trail=[start],
            )

        self.tick = 0
        self.plan_due = True
        self.stagnation_ticks = 0
        self.status_text = "Scene reset"
        self.replan_paths(force=True)

    def set_mode(self, mode: str) -> None:
        if mode == "FAST":
            self.mode = "FAST"
            self.num_candidates = 3
            self.qaoa_depth = 1
            self.qaoa_restarts = 1
        else:
            self.mode = "QUALITY"
            self.num_candidates = 4
            self.qaoa_depth = 2
            self.qaoa_restarts = 3
        self.plan_due = True

    def _robot_cells(self) -> set:
        return {r.pos for r in self.robots.values()}

    def _goal_cells(self) -> set:
        return {r.goal for r in self.robots.values()}

    def _build_candidates(self) -> Dict[int, List[Dict]]:
        # Plan on static map only. Dynamic obstacles are transient and handled
        # by runtime collision checks + periodic replanning.
        planning_grid = copy.deepcopy(self.grid)
        planning_grid.dynamic_obstacles = []

        all_candidates: Dict[int, List[Dict]] = {}
        for rid in sorted(self.robots.keys()):
            robot = self.robots[rid]
            if robot.reached_goal:
                all_candidates[rid] = [{"path": [robot.pos], "cost": 0.0, "variant": "at-goal"}]
                continue

            cands = generate_candidate_paths(
                planning_grid,
                start=robot.pos,
                goal=robot.goal,
                num_candidates=self.num_candidates,
                seed=self.seed + rid * 1009 + self.tick,
            )

            if not cands:
                # Fallback to a wait action if path is blocked currently.
                cands = [{"path": [robot.pos], "cost": 9999.0, "variant": "wait"}]

            all_candidates[rid] = cands
        return all_candidates

    def replan_paths(self, force: bool = False) -> None:
        if not force and not self.plan_due:
            return

        t0 = time.perf_counter()
        self.all_candidates = self._build_candidates()

        emergency = self.stagnation_ticks >= 3

        try:
            qaoa_result = qaoa_select_full(
                self.all_candidates,
                conflict_penalty=10.0,
                one_hot_penalty=50.0,
                p=self.qaoa_depth,
                num_restarts=self.qaoa_restarts,
                seed=self.seed + self.tick,
                prefer_qiskit=False,
            )
            self.selection = qaoa_result.selection
            self.last_backend = qaoa_result.backend
            self.last_score = qaoa_result.score
            self.last_conflicts = qaoa_result.total_conflicts

            if emergency:
                # If everyone has been stuck for several ticks, perturb selection
                # to break deadlocks near chokepoints.
                self.selection = random_select(self.all_candidates, seed=self.seed + self.tick)
                self.last_backend = "random-escape"
        except Exception:
            # Safe fallback for live demos.
            self.selection = greedy_select(self.all_candidates)
            self.last_backend = "greedy-fallback"
            self.last_score = 0.0
            self.last_conflicts = 0

        for rid in sorted(self.robots.keys()):
            robot = self.robots[rid]
            sel_idx = self.selection.get(rid, 0)
            selected = self.all_candidates[rid][sel_idx]["path"]
            robot.path = selected if selected else [robot.pos]
            robot.path_idx = 0

        self.last_plan_ms = (time.perf_counter() - t0) * 1000.0
        self.plan_due = False
        if emergency:
            self.stagnation_ticks = 0
        self.status_text = (
            f"Replanned via A* candidates + QAOA ({self.mode}) | "
            f"{self.last_plan_ms:.0f} ms"
        )

    def _step_robots(self) -> int:
        occupied = self._robot_cells()
        next_positions: Dict[int, Position] = {}
        moved = 0

        for rid in sorted(self.robots.keys()):
            robot = self.robots[rid]
            if robot.reached_goal:
                next_positions[rid] = robot.pos
                continue

            if robot.path_idx + 1 >= len(robot.path):
                next_positions[rid] = robot.pos
                continue

            nxt = robot.path[robot.path_idx + 1]
            blocked = (not self.grid.is_free(nxt))
            conflict = (nxt in next_positions.values())

            if blocked or conflict:
                next_positions[rid] = robot.pos
                self.plan_due = True
            else:
                next_positions[rid] = nxt

        for rid in sorted(self.robots.keys()):
            robot = self.robots[rid]
            new_pos = next_positions[rid]
            if new_pos != robot.pos:
                robot.pos = new_pos
                robot.path_idx = min(robot.path_idx + 1, len(robot.path) - 1)
                robot.trail.append(new_pos)
                moved += 1

            if robot.pos == robot.goal:
                robot.reached_goal = True

        return moved

    def advance_one_tick(self) -> None:
        self.grid.step_dynamic_obstacles()
        moved = self._step_robots()
        self.tick += 1

        if moved == 0 and not all(r.reached_goal for r in self.robots.values()):
            self.stagnation_ticks += 1
            self.plan_due = True
        else:
            self.stagnation_ticks = 0

        if self.tick % self.plan_interval == 0:
            self.plan_due = True

        if self.plan_due:
            self.replan_paths()

    def grid_cell_from_mouse(self, mx: int, my: int):
        gx = mx - self.margin
        gy = my - self.margin
        if gx < 0 or gy < 0:
            return None
        c = gx // self.cell
        r = gy // self.cell
        if 0 <= r < self.size and 0 <= c < self.size:
            return int(r), int(c)
        return None

    def add_dynamic_obstacle(self, pos: Position) -> None:
        if pos in self._robot_cells() or pos in self._goal_cells():
            return
        if self.grid.grid[pos[0], pos[1]] == 1:
            return
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        d = random.choice(directions)
        self.grid.add_dynamic_obstacle(pos, direction=d, pattern="bounce")
        self.plan_due = True

    def remove_dynamic_obstacle_at(self, pos: Position) -> bool:
        removed = False
        keep = []
        for obs in self.grid.dynamic_obstacles:
            if tuple(obs["pos"]) == pos and not removed:
                removed = True
                continue
            keep.append(obs)
        if removed:
            self.grid.dynamic_obstacles = keep
            self.plan_due = True
        return removed

    def toggle_wall(self, pos: Position) -> None:
        if pos in self._robot_cells() or pos in self._goal_cells():
            return
        r, c = pos
        self.grid.grid[r, c] = 0 if self.grid.grid[r, c] == 1 else 1
        self.plan_due = True

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif key == pygame.K_n and self.paused:
                    self.advance_one_tick()
                elif key == pygame.K_p:
                    self.plan_due = True
                    self.replan_paths(force=True)
                elif key == pygame.K_c:
                    self.show_candidates = not self.show_candidates
                elif key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif key == pygame.K_g:
                    # Spawn random dynamic obstacle.
                    for _ in range(100):
                        r = random.randint(0, self.size - 1)
                        c = random.randint(0, self.size - 1)
                        pos = (r, c)
                        if (
                            self.grid.grid[r, c] == 0
                            and pos not in self._robot_cells()
                            and pos not in self._goal_cells()
                        ):
                            self.add_dynamic_obstacle(pos)
                            break
                elif key == pygame.K_r:
                    self.reset_scene()
                elif key == pygame.K_UP:
                    self.sim_hz = min(20.0, self.sim_hz + 1.0)
                elif key == pygame.K_DOWN:
                    self.sim_hz = max(1.0, self.sim_hz - 1.0)
                elif key == pygame.K_o:
                    self.set_mode("QUALITY" if self.mode == "FAST" else "FAST")
                    self.replan_paths(force=True)

            if event.type == pygame.MOUSEBUTTONDOWN:
                cell = self.grid_cell_from_mouse(*event.pos)
                if cell is None:
                    continue

                if event.button == 1:
                    self.toggle_wall(cell)
                elif event.button == 2:
                    self.add_dynamic_obstacle(cell)
                elif event.button == 3:
                    removed = self.remove_dynamic_obstacle_at(cell)
                    if not removed and cell not in self._robot_cells() and cell not in self._goal_cells():
                        self.grid.grid[cell[0], cell[1]] = 0
                        self.plan_due = True

    def draw_grid(self) -> None:
        surf = self.screen
        surf.fill(self.BG)

        gx0, gy0 = self.margin, self.margin
        pygame.draw.rect(surf, self.GRID_BG, (gx0 - 2, gy0 - 2, self.grid_px + 4, self.grid_px + 4), border_radius=8)

        # Cells
        for r in range(self.size):
            for c in range(self.size):
                x = gx0 + c * self.cell
                y = gy0 + r * self.cell
                color = self.WALL if self.grid.grid[r, c] == 1 else self.FREE
                pygame.draw.rect(surf, color, (x, y, self.cell - 1, self.cell - 1))

        # Goals
        for robot in self.robots.values():
            x = gx0 + robot.goal[1] * self.cell + self.cell // 2
            y = gy0 + robot.goal[0] * self.cell + self.cell // 2
            s = self.cell // 3
            pygame.draw.rect(surf, self.GOAL, (x - s, y - s, 2 * s, 2 * s), width=2)

        # Candidate paths (faint)
        if self.show_candidates:
            layer = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
            for rid in sorted(self.all_candidates.keys()):
                robot = self.robots[rid]
                base = robot.color
                for idx, cand in enumerate(self.all_candidates[rid]):
                    pts = [
                        (
                            p[1] * self.cell + self.cell // 2,
                            p[0] * self.cell + self.cell // 2,
                        )
                        for p in cand["path"]
                    ]
                    if len(pts) < 2:
                        continue
                    alpha = 80 if self.selection.get(rid, -1) != idx else 170
                    pygame.draw.lines(layer, (*base, alpha), False, pts, 2)
            surf.blit(layer, (gx0, gy0))

        # Dynamic obstacles
        for obs in self.grid.dynamic_obstacles:
            r, c = obs["pos"]
            x = gx0 + c * self.cell + self.cell // 2
            y = gy0 + r * self.cell + self.cell // 2
            pygame.draw.circle(surf, self.DYN_OBS, (x, y), self.cell // 3)

        # Trails
        if self.show_trails:
            for robot in self.robots.values():
                if len(robot.trail) < 2:
                    continue
                pts = [
                    (
                        gx0 + p[1] * self.cell + self.cell // 2,
                        gy0 + p[0] * self.cell + self.cell // 2,
                    )
                    for p in robot.trail
                ]
                pygame.draw.lines(surf, robot.color, False, pts, 2)

        # Robots
        for robot in self.robots.values():
            x = gx0 + robot.pos[1] * self.cell + self.cell // 2
            y = gy0 + robot.pos[0] * self.cell + self.cell // 2
            pygame.draw.circle(surf, robot.color, (x, y), self.cell // 3)
            label = self.small.render(str(robot.rid), True, (255, 255, 255))
            surf.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))

        # Sidebar
        sx = gx0 + self.grid_px + self.margin
        pygame.draw.rect(surf, self.PANEL, (sx, gy0, self.sidebar_w, self.grid_px), border_radius=10)

        lines = [
            "Realtime Paper Demo",
            "Novelty: A* candidates + QAOA",
            "",
            f"Tick: {self.tick}",
            f"Speed: {self.sim_hz:.1f} ticks/sec",
            f"Mode: {self.mode}  (O toggle)",
            f"QAOA p={self.qaoa_depth}, restarts={self.qaoa_restarts}",
            f"Candidates/robot: {self.num_candidates}",
            f"Last plan: {self.last_plan_ms:.0f} ms",
            f"Backend: {self.last_backend}",
            f"Score: {self.last_score:.1f}",
            f"Conflicts: {self.last_conflicts}",
            f"Dynamic obstacles: {len(self.grid.dynamic_obstacles)}",
            "",
            "Controls",
            "SPACE pause | N step | P replan",
            "Left click: wall toggle",
            "Middle click: add dynamic obstacle",
            "Right click: remove dynamic obstacle",
            "C: candidate lines  T: trails",
            "UP/DOWN: speed  R: reset",
            "G: random dynamic obstacle",
            "ESC/Q: quit",
            "",
            self.status_text,
        ]

        y = gy0 + 14
        for i, line in enumerate(lines):
            if not line:
                y += 10
                continue
            color = self.TXT
            if i in (0, 1, 14):
                color = (196, 230, 253)
            if line.startswith("Last plan") and self.last_plan_ms > 1200:
                color = self.ALERT
            text = self.small.render(line, True, color)
            surf.blit(text, (sx + 14, y))
            y += 21

    def run(self) -> None:
        accum = 0.0
        prev = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            self.handle_events()

            if not self.paused:
                accum += dt
                step_period = 1.0 / max(1.0, self.sim_hz)
                while accum >= step_period:
                    self.advance_one_tick()
                    accum -= step_period

            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    sim = PaperDemoSimulator(size=20, num_robots=3, num_dynamic=4, seed=42)
    sim.run()


if __name__ == "__main__":
    main()
