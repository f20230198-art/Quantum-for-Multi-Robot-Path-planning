"""
grid.py  –  2-D Grid Environment for Multi-Robot Path Planning
================================================================

WHAT THIS FILE DOES (plain English):
    Creates a 2-D "world" represented as an N×N grid of cells.
    Each cell is either FREE (a robot can walk on it) or an OBSTACLE
    (a wall the robot must avoid).

    The grid can also contain DYNAMIC obstacles that move each time-step
    (added in Week 2).

HOW IT WORKS:
    • The grid is stored as a 2-D NumPy array of 0s and 1s.
        0 = free cell
        1 = static obstacle
    • You choose the grid size (e.g. 20×20) and an obstacle ratio
      (e.g. 0.15 = 15 % of cells are walls).
    • Obstacles are placed randomly but the start and goal cells are
      always kept free.
    • Helper methods let other files ask "is cell (r,c) walkable?" and
      "what are the neighbours of cell (r,c)?".
"""

import numpy as np
import copy
from typing import List, Tuple, Optional, Dict

# Type alias for a grid position  (row, col)
Position = Tuple[int, int]


class Grid:
    """
    A square 2-D grid world.

    Attributes
    ----------
    size : int
        Number of rows (and columns) – the grid is size × size.
    grid : np.ndarray
        2-D array where 0 = free, 1 = static obstacle.
    start : Position
        Starting cell (row, col).
    goal : Position
        Goal cell (row, col).
    dynamic_obstacles : list[dict]
        Moving obstacles added later (Week 2).  Each dict holds
        ``pos``, ``direction``, ``speed``.
    """

    # Eight possible movement directions: up, down, left, right + diagonals
    DIRECTIONS_8 = [
        (-1,  0),  # up
        ( 1,  0),  # down
        ( 0, -1),  # left
        ( 0,  1),  # right
        (-1, -1),  # up-left
        (-1,  1),  # up-right
        ( 1, -1),  # down-left
        ( 1,  1),  # down-right
    ]

    # Four cardinal directions only (no diagonals)
    DIRECTIONS_4 = [
        (-1,  0),
        ( 1,  0),
        ( 0, -1),
        ( 0,  1),
    ]

    def __init__(
        self,
        size: int = 20,
        obstacle_ratio: float = 0.15,
        start: Optional[Position] = None,
        goal: Optional[Position] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        size : int
            Grid width and height.
        obstacle_ratio : float
            Fraction of cells that become walls (0.0 – 0.4 recommended).
        start : Position or None
            If None, defaults to top-left corner (0, 0).
        goal : Position or None
            If None, defaults to bottom-right corner (size-1, size-1).
        seed : int or None
            Random seed for reproducibility.
        """
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.rng = np.random.RandomState(seed)

        # Default start / goal
        self.start: Position = start if start is not None else (0, 0)
        self.goal: Position = goal if goal is not None else (size - 1, size - 1)

        # Build the grid
        self.grid = self._generate_grid()

        # Dynamic obstacles list
        self.dynamic_obstacles: List[dict] = []

        # Multiple robot start/goal pairs (for multi-robot scenarios)
        # Each entry: {"id": int, "start": Position, "goal": Position}
        self.robot_configs: List[Dict] = []

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------
    def _generate_grid(self) -> np.ndarray:
        """Create a random grid ensuring start and goal are free and a
        path between them exists."""
        while True:
            g = np.zeros((self.size, self.size), dtype=np.int8)
            num_obstacles = int(self.size * self.size * self.obstacle_ratio)
            # Pick random obstacle positions
            all_positions = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if (r, c) != self.start and (r, c) != self.goal
            ]
            chosen = self.rng.choice(len(all_positions), size=min(num_obstacles, len(all_positions)), replace=False)
            for idx in chosen:
                r, c = all_positions[idx]
                g[r, c] = 1

            # Quick BFS check: make sure start→goal is reachable
            if self._is_reachable(g):
                return g

    def _is_reachable(self, g: np.ndarray) -> bool:
        """BFS to verify a path exists from start to goal."""
        from collections import deque
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in self.DIRECTIONS_8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited and g[nr, nc] == 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    # ------------------------------------------------------------------
    # Queries other files use
    # ------------------------------------------------------------------
    def is_free(self, pos: Position) -> bool:
        """Return True if *pos* is inside the grid and not an obstacle."""
        r, c = pos
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.grid[r, c] == 1:
            return False
        # Also check dynamic obstacles
        for obs in self.dynamic_obstacles:
            if tuple(obs["pos"]) == (r, c):
                return False
        return True

    def get_neighbors(self, pos: Position, eight_connected: bool = True) -> List[Position]:
        """Return walkable neighbours of *pos*.
        
        For diagonal moves, BOTH adjacent cardinal cells must also be
        free — this prevents "corner cutting" through diagonal walls.
        Example: to move from (r,c) to (r+1,c+1), cells (r+1,c) AND
        (r,c+1) must both be free, otherwise the robot would clip
        through a wall corner.
        """
        directions = self.DIRECTIONS_8 if eight_connected else self.DIRECTIONS_4
        neighbors = []
        r, c = pos
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not self.is_free((nr, nc)):
                continue
            # Diagonal move — check corner-cutting
            if dr != 0 and dc != 0:
                # Both adjacent cardinal cells must be free
                if not self.is_free((r + dr, c)) or not self.is_free((r, c + dc)):
                    continue  # blocked — would clip through wall corner
            neighbors.append((nr, nc))
        return neighbors

    def get_obstacle_positions(self) -> List[Position]:
        """Return list of all static obstacle cells."""
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 1:
                    positions.append((r, c))
        return positions

    # ------------------------------------------------------------------
    # Dynamic obstacle management
    # ------------------------------------------------------------------

    def add_dynamic_obstacle(
        self,
        pos: Position,
        direction: Tuple[int, int] = (0, 1),
        speed: int = 1,
        pattern: str = "bounce",
    ) -> None:
        """
        Add a moving obstacle to the grid.

        Parameters
        ----------
        pos : Position
            Initial (row, col).
        direction : tuple
            Movement direction (dr, dc), e.g. (0,1)=right, (1,0)=down.
        speed : int
            Cells per timestep (usually 1).
        pattern : str
            'bounce' — reverses direction at walls.
            'loop'   — wraps around the grid edges.
        """
        self.dynamic_obstacles.append({
            "pos": list(pos),
            "direction": list(direction),
            "speed": speed,
            "pattern": pattern,
        })

    def step_dynamic_obstacles(self) -> None:
        """
        Advance all dynamic obstacles by one timestep.

        Bounce pattern: reverse direction when hitting a wall or grid edge.
        Loop pattern:  wrap around to the opposite side.
        """
        for obs in self.dynamic_obstacles:
            r, c = obs["pos"]
            dr, dc = obs["direction"]
            nr, nc = r + dr * obs["speed"], c + dc * obs["speed"]

            if obs["pattern"] == "bounce":
                # Reverse if next position is out-of-bounds or a static wall
                if not (0 <= nr < self.size and 0 <= nc < self.size) or \
                   self.grid[nr, nc] == 1:
                    obs["direction"] = [-dr, -dc]
                    nr, nc = r + (-dr) * obs["speed"], c + (-dc) * obs["speed"]
                    # If reverse is also blocked, stay put
                    if not (0 <= nr < self.size and 0 <= nc < self.size) or \
                       self.grid[nr, nc] == 1:
                        nr, nc = r, c
            elif obs["pattern"] == "loop":
                nr = nr % self.size
                nc = nc % self.size
                # Skip if wrapped into a static wall
                if self.grid[nr, nc] == 1:
                    nr, nc = r, c

            obs["pos"] = [nr, nc]

    def get_dynamic_obstacle_positions(self) -> List[Position]:
        """Return current positions of all dynamic obstacles."""
        return [tuple(obs["pos"]) for obs in self.dynamic_obstacles]

    def snapshot(self) -> np.ndarray:
        """
        Return a 2-D array where 0=free, 1=static obstacle,
        2=dynamic obstacle.  Useful for visualization and
        collision checking at a single point in time.
        """
        s = self.grid.copy()
        for obs in self.dynamic_obstacles:
            r, c = obs["pos"]
            if 0 <= r < self.size and 0 <= c < self.size:
                s[r, c] = 2
        return s

    def is_free_at_time(self, pos: Position, dynamic_positions: List[Position]) -> bool:
        """
        Check if *pos* is free considering a specific set of dynamic
        obstacle positions (used when simulating future timesteps).
        """
        r, c = pos
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.grid[r, c] == 1:
            return False
        if pos in dynamic_positions:
            return False
        return True

    # ------------------------------------------------------------------
    # Multi-robot environment factories
    # ------------------------------------------------------------------

    @classmethod
    def create_multi_robot_env(
        cls,
        size: int = 20,
        obstacle_ratio: float = 0.12,
        num_robots: int = 3,
        num_dynamic: int = 4,
        seed: int = 42,
    ) -> "Grid":
        """
        Create an environment with multiple robots and dynamic obstacles.

        Robot start/goal pairs are placed in different corners / edges
        so that their paths are likely to conflict — this is where QAOA
        adds value.

        Parameters
        ----------
        size : int
            Grid dimensions (size x size).
        obstacle_ratio : float
            Fraction of cells that are static walls.
        num_robots : int
            Number of robots (2–5 supported).
        num_dynamic : int
            Number of moving obstacles.
        seed : int
            Random seed.
        """
        # Pre-defined start/goal positions that force path conflicts
        # (robots must cross each other's natural routes)
        presets = [
            {"start": (0,          0),           "goal": (size-1, size-1)},  # top-left  → bottom-right
            {"start": (0,          size-1),      "goal": (size-1, 0)},       # top-right → bottom-left
            {"start": (size-1,     0),           "goal": (0,      size-1)},  # bottom-left → top-right
            {"start": (size-1,     size-1),      "goal": (0,      0)},       # bottom-right → top-left
            {"start": (size//2,    0),           "goal": (size//2, size-1)},  # mid-left → mid-right
        ]

        # Use first robot's start/goal as the grid's default
        g = cls(
            size=size,
            obstacle_ratio=obstacle_ratio,
            start=presets[0]["start"],
            goal=presets[0]["goal"],
            seed=seed,
        )

        # Make sure all robot start/goal cells are free
        for i in range(min(num_robots, len(presets))):
            sr, sc = presets[i]["start"]
            gr, gc = presets[i]["goal"]
            g.grid[sr, sc] = 0
            g.grid[gr, gc] = 0

        # Store robot configurations
        g.robot_configs = []
        for i in range(min(num_robots, len(presets))):
            g.robot_configs.append({
                "id": i,
                "start": presets[i]["start"],
                "goal": presets[i]["goal"],
            })

        # Add dynamic obstacles in the middle region
        rng = np.random.RandomState(seed + 100)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        mid = size // 2
        placed = 0
        attempts = 0
        while placed < num_dynamic and attempts < 200:
            r = rng.randint(mid - size // 4, mid + size // 4)
            c = rng.randint(mid - size // 4, mid + size // 4)
            attempts += 1
            if g.grid[r, c] == 0 and (r, c) not in [
                cfg["start"] for cfg in g.robot_configs
            ] and (r, c) not in [
                cfg["goal"] for cfg in g.robot_configs
            ]:
                d = directions[rng.randint(0, len(directions))]
                g.add_dynamic_obstacle((r, c), direction=d, pattern="bounce")
                placed += 1

        return g

    # ------------------------------------------------------------------
    # Pre-built scenario grids (used in experiments later)
    # ------------------------------------------------------------------
    @classmethod
    def create_trap_environment(cls, size: int = 20) -> "Grid":
        """
        Create a grid with a U-shaped trap obstacle.

        The trap is designed so that APF will pull the robot INTO the
        U-shape and get it stuck (local minimum).  This is the scenario
        where quantum Q-learning will shine.
        """
        g = cls(size=size, obstacle_ratio=0.0, seed=42)
        # Build U-shape in the middle of the grid
        mid = size // 2
        wall_len = size // 3
        # Left wall of U
        for r in range(mid - wall_len, mid + wall_len):
            g.grid[r, mid - 3] = 1
        # Right wall of U
        for r in range(mid - wall_len, mid + wall_len):
            g.grid[r, mid + 3] = 1
        # Bottom wall of U (connecting left and right)
        for c in range(mid - 3, mid + 4):
            g.grid[mid + wall_len - 1, c] = 1
        # Place goal inside the U so APF pulls robot in
        g.goal = (mid, mid)
        g.start = (1, 1)
        return g

    @classmethod
    def create_narrow_passage(cls, size: int = 20) -> "Grid":
        """
        Create a grid with a narrow corridor the robot must pass through.
        """
        g = cls(size=size, obstacle_ratio=0.0, seed=42)
        mid = size // 2
        # Horizontal wall across the middle with a 1-cell gap
        for c in range(size):
            if c != mid:  # leave one gap
                g.grid[mid, c] = 1
        g.start = (1, 1)
        g.goal = (size - 2, size - 2)
        return g

    # ------------------------------------------------------------------
    # String representation (for debugging)
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.start:
                    row += "S "
                elif (r, c) == self.goal:
                    row += "G "
                elif self.grid[r, c] == 1:
                    row += "█ "
                else:
                    row += ". "
            lines.append(row)
        return "\n".join(lines)
