"""
visualize.py  –  Matplotlib Visualization for Grid & Paths
==========================================================

WHAT THIS FILE DOES (plain English):
    Draws the grid as a colour-coded image using Matplotlib.
        • White   = free cell
        • Black   = static obstacle
        • Red     = dynamic obstacle
        • Green ● = start
        • Red  ★  = goal
        • Blue —  = planned path

    It can also save the image to a PNG file so you can put it
    in your report / presentation.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves PNGs without blocking
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Optional, Tuple, Dict

from grid import Grid, Position


# Colour constants (RGB floats)
_FREE    = (1.0, 1.0, 1.0)   # white
_STATIC  = (0.15, 0.15, 0.15)  # dark grey / black
_DYNAMIC = (0.9, 0.2, 0.2)   # red
_PATH    = (0.2, 0.5, 1.0)   # blue

# Robot colour palette (for multi-robot views)
ROBOT_COLORS = ["#3399FF", "#FF6633", "#33CC66", "#CC33FF", "#FFCC00"]
ROBOT_COLORS_RGB = [
    (0.2, 0.6, 1.0),   # blue
    (1.0, 0.4, 0.2),   # orange
    (0.2, 0.8, 0.4),   # green
    (0.8, 0.2, 1.0),   # purple
    (1.0, 0.8, 0.0),   # yellow
]


def draw_grid(
    grid: Grid,
    paths: Optional[List[List[Position]]] = None,
    path_colors: Optional[List[str]] = None,
    title: str = "Grid World",
    show: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Render the grid, obstacles, start/goal markers, and optional path(s).

    Parameters
    ----------
    grid : Grid
        The environment to draw.
    paths : list of paths, optional
        Each path is a list of (row, col) positions.
        Multiple paths can be drawn (e.g. one per robot).
    path_colors : list of colour strings, optional
        One colour per path.  Defaults to a built-in palette.
    title : str
        Figure title.
    show : bool
        Call plt.show() at the end?
    save_path : str or None
        If given, save the figure to this file path.
    figsize : tuple
        Figure size in inches.
    ax : matplotlib Axes or None
        If given, draw on this axes instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---- build RGB image ----
    img = np.ones((grid.size, grid.size, 3), dtype=np.float32)

    # Static obstacles
    for r in range(grid.size):
        for c in range(grid.size):
            if grid.grid[r, c] == 1:
                img[r, c] = _STATIC

    # Dynamic obstacles
    for obs in grid.dynamic_obstacles:
        pr, pc = obs["pos"]
        if 0 <= pr < grid.size and 0 <= pc < grid.size:
            img[pr, pc] = _DYNAMIC

    # ---- create figure ----
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(img, origin="upper", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3)
    ax.tick_params(which="minor", size=0)

    # Major ticks every 5
    ax.set_xticks(np.arange(0, grid.size, 5))
    ax.set_yticks(np.arange(0, grid.size, 5))

    # ---- draw path(s) ----
    default_colors = ["#3399FF", "#FF6633", "#33CC66", "#CC33FF", "#FFCC00"]
    if paths:
        if path_colors is None:
            path_colors = default_colors
        for i, path in enumerate(paths):
            color = path_colors[i % len(path_colors)]
            if len(path) >= 2:
                cols = [p[1] for p in path]
                rows = [p[0] for p in path]
                ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.85,
                        label=f"Path {i+1}" if len(paths) > 1 else "Path")

    # ---- start & goal markers ----
    sr, sc = grid.start
    gr, gc = grid.goal
    ax.plot(sc, sr, marker="o", markersize=14, color="limegreen",
            markeredgecolor="black", markeredgewidth=1.5, zorder=5,
            label="Start")
    ax.plot(gc, gr, marker="*", markersize=18, color="red",
            markeredgecolor="black", markeredgewidth=1.0, zorder=5,
            label="Goal")

    # ---- legend ----
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved figure to {save_path}")

    if show:
        plt.show()

    return fig


def draw_comparison(
    grid: Grid,
    paths_dict: dict,
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
):
    """
    Draw the SAME grid side-by-side with different algorithm paths.

    Parameters
    ----------
    grid : Grid
        The shared environment.
    paths_dict : dict
        Keys = algorithm names (str), Values = list of (row,col).
    title : str
        Overall figure title.
    save_path : str or None
        If given, save figure.
    """
    n = len(paths_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, path) in zip(axes, paths_dict.items()):
        draw_grid(grid, paths=[path], title=name, show=False, ax=ax)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved comparison to {save_path}")

    if show:
        plt.show()
    return fig


# ==================================================================
# Step-by-step A* search process visualization (image strip)
# ==================================================================

def draw_search_steps(
    grid: Grid,
    expansion_order: List[Position],
    final_path: Optional[List[Position]],
    num_snapshots: int = 6,
    title: str = "A* Search Process — Step by Step",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a horizontal strip of images showing A* progressively
    exploring the grid and finally revealing the path.

    Each snapshot shows:
        • Light-blue cells  = already explored (closed set)
        • Yellow cell       = cell being expanded RIGHT NOW
        • Final panel       = the discovered path in blue

    Parameters
    ----------
    grid : Grid
    expansion_order : list of Position
        Cells in the order A* expanded them (from AStarResult).
    final_path : list of Position or None
    num_snapshots : int
        How many intermediate frames to show (default 6).
    save_path : str or None
    """
    total = len(expansion_order)
    # Pick evenly-spaced snapshot indices + the final step
    if total <= num_snapshots:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_snapshots - 1))
                   for i in range(num_snapshots)]
    # Always include the last expansion step
    if indices[-1] != total - 1:
        indices.append(total - 1)

    n_panels = len(indices) + 1          # +1 for the "final path" panel
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    for panel_idx, snap_idx in enumerate(indices):
        ax = axes[panel_idx]
        explored = set(expansion_order[:snap_idx + 1])
        current_cell = expansion_order[snap_idx]
        _draw_search_snapshot(
            ax, grid, explored, current_cell,
            label=f"Step {snap_idx + 1}/{total}")

    # Final panel: show the path
    ax_final = axes[-1]
    _draw_search_snapshot(
        ax_final, grid,
        explored=set(expansion_order),
        current_cell=None,
        path=final_path,
        label="Final Path")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved search steps to {save_path}")
    return fig


def _draw_search_snapshot(
    ax: plt.Axes,
    grid: Grid,
    explored: set,
    current_cell: Optional[Position],
    path: Optional[List[Position]] = None,
    label: str = "",
):
    """Draw one snapshot panel for the step-by-step strip."""
    img = np.ones((grid.size, grid.size, 3), dtype=np.float32)

    # Static obstacles
    for r in range(grid.size):
        for c in range(grid.size):
            if grid.grid[r, c] == 1:
                img[r, c] = _STATIC

    # Explored cells — light blue tint
    for (r, c) in explored:
        if grid.grid[r, c] == 0:
            img[r, c] = (0.78, 0.90, 1.0)  # light blue

    # Current cell — yellow highlight
    if current_cell is not None:
        r, c = current_cell
        img[r, c] = (1.0, 0.92, 0.23)  # bright yellow

    ax.imshow(img, origin="upper", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, grid.size, 5))
    ax.set_yticks(np.arange(0, grid.size, 5))

    # Path overlay
    if path and len(path) >= 2:
        cols = [p[1] for p in path]
        rows = [p[0] for p in path]
        ax.plot(cols, rows, color="#3399FF", linewidth=2.5, alpha=0.9)

    # Start & Goal
    sr, sc = grid.start
    gr, gc = grid.goal
    ax.plot(sc, sr, marker="o", markersize=10, color="limegreen",
            markeredgecolor="black", markeredgewidth=1.2, zorder=5)
    ax.plot(gc, gr, marker="*", markersize=14, color="red",
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_title(label, fontsize=10, fontweight="bold")


# ==================================================================
# Animated GIF of the robot walking the final path
# ==================================================================

def draw_robot_walk_gif(
    grid: Grid,
    path: List[Position],
    save_path: str,
    fps: int = 4,
    title: str = "Robot Walking A* Path",
) -> None:
    """
    Save an animated GIF of the robot (blue dot) walking along *path*.

    Parameters
    ----------
    grid : Grid
    path : list of Position
    save_path : str   – must end with .gif
    fps : int         – frames per second
    title : str
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[visualize] ⚠  Could not import animation — skipping GIF.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    def _base_frame():
        """Draw the static grid once."""
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        ax.imshow(img, origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))
        # Start & Goal
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker="o", markersize=12, color="limegreen",
                markeredgecolor="black", markeredgewidth=1.2, zorder=4)
        ax.plot(gc, gr, marker="*", markersize=16, color="red",
                markeredgecolor="black", markeredgewidth=0.8, zorder=4)
        ax.set_title(title, fontsize=12, fontweight="bold")

    _base_frame()

    # Animated elements
    trail_line, = ax.plot([], [], color="#3399FF", linewidth=2, alpha=0.6)
    robot_dot, = ax.plot([], [], marker="o", markersize=14,
                         color="#1155CC", markeredgecolor="white",
                         markeredgewidth=2, zorder=6)

    def _update(frame_idx):
        # Trail up to current position
        trail = path[:frame_idx + 1]
        cols = [p[1] for p in trail]
        rows = [p[0] for p in trail]
        trail_line.set_data(cols, rows)
        # Robot position
        r, c = path[frame_idx]
        robot_dot.set_data([c], [r])
        return trail_line, robot_dot

    anim = FuncAnimation(fig, _update, frames=len(path),
                         interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[visualize] Saved walk GIF to {save_path}")


# ==================================================================
# Greedy BFS (for comparison — a "non-optimal" traditional planner)
# ==================================================================

def greedy_bfs(grid: Grid, heuristic=None) -> dict:
    """
    Greedy Best-First Search — expands cells closest to the goal first
    (ignores actual cost g).  Finds A path but NOT necessarily the
    shortest one.  Used as the 'non-smart' traditional baseline to
    contrast with A* and later with quantum planners.

    Returns dict with keys: path, cost, nodes_expanded, expansion_order.
    """
    import heapq, math
    if heuristic is None:
        heuristic = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    start, goal = grid.start, grid.goal
    open_set = []
    counter = 0
    heapq.heappush(open_set, (heuristic(start, goal), counter, start))
    came_from = {start: None}
    closed = set()
    expansion_order = []

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expansion_order.append(current)

        if current == goal:
            # reconstruct
            path = [current]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            path.reverse()
            cost = sum(
                math.sqrt(2) if abs(path[i][0]-path[i+1][0])+abs(path[i][1]-path[i+1][1])==2 else 1.0
                for i in range(len(path)-1)
            )
            return {"path": path, "cost": cost,
                    "nodes_expanded": len(expansion_order),
                    "expansion_order": expansion_order}

        for nb in grid.get_neighbors(current, eight_connected=True):
            if nb not in closed and nb not in came_from:
                came_from[nb] = current
                counter += 1
                heapq.heappush(open_set, (heuristic(nb, goal), counter, nb))

    return {"path": None, "cost": float("inf"),
            "nodes_expanded": len(expansion_order),
            "expansion_order": expansion_order}


def draw_algorithm_comparison(
    grid: Grid,
    results: Dict[str, dict],
    title: str = "Algorithm Comparison — Traditional Planners",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side panels comparing multiple algorithms on the SAME grid.

    Parameters
    ----------
    grid : Grid
    results : dict
        Keys = algorithm name, Values = dict with keys
        'path', 'cost', 'nodes_expanded', 'expansion_order'.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        # Draw explored cells
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        for (r, c) in res.get("expansion_order", []):
            if grid.grid[r, c] == 0:
                img[r, c] = (0.78, 0.90, 1.0)

        ax.imshow(img, origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))

        # Path
        path = res.get("path")
        if path and len(path) >= 2:
            cols = [p[1] for p in path]
            rows = [p[0] for p in path]
            ax.plot(cols, rows, color="#3399FF", linewidth=2.5, alpha=0.9)

        # Start & Goal
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker="o", markersize=12, color="limegreen",
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(gc, gr, marker="*", markersize=16, color="red",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        cost_str = f"{res['cost']:.2f}" if res.get('path') else "FAIL"
        subtitle = (f"{name}\n"
                    f"Cost: {cost_str}  |  "
                    f"Explored: {res['nodes_expanded']} cells")
        ax.set_title(subtitle, fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved algorithm comparison to {save_path}")
    return fig


# ==================================================================
# Multi-Robot Visualization
# ==================================================================

def draw_multi_robot_grid(
    grid: Grid,
    robot_paths: Optional[Dict[int, List[Position]]] = None,
    title: str = "Multi-Robot Environment",
    show: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Draw the grid with multiple robots' start/goal and their paths.

    Parameters
    ----------
    grid : Grid
        Environment with robot_configs and dynamic_obstacles set.
    robot_paths : dict or None
        robot_id → list of positions.  If None, only start/goal shown.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Base image
    img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
    for r in range(grid.size):
        for c in range(grid.size):
            if grid.grid[r, c] == 1:
                img[r, c] = _STATIC

    # Dynamic obstacles
    for obs in grid.dynamic_obstacles:
        pr, pc = obs["pos"]
        if 0 <= pr < grid.size and 0 <= pc < grid.size:
            img[pr, pc] = _DYNAMIC

    ax.imshow(img, origin="upper", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, grid.size, 5))
    ax.set_yticks(np.arange(0, grid.size, 5))

    # Plot each robot's path, start, and goal
    for cfg in grid.robot_configs:
        rid = cfg["id"]
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        sr, sc = cfg["start"]
        gr, gc = cfg["goal"]

        # Path
        if robot_paths and rid in robot_paths:
            path = robot_paths[rid]
            if len(path) >= 2:
                cols = [p[1] for p in path]
                rows = [p[0] for p in path]
                ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.8,
                        label=f"Robot {rid}")

        # Start marker (circle)
        ax.plot(sc, sr, marker="o", markersize=14, color=color,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.text(sc, sr, f"S{rid}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)

        # Goal marker (star)
        ax.plot(gc, gr, marker="*", markersize=18, color=color,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        ax.text(gc + 0.5, gr, f"G{rid}", ha="left", va="center",
                fontsize=7, fontweight="bold", color=color, zorder=6)

    # Dynamic obstacle label
    if grid.dynamic_obstacles:
        ax.plot([], [], marker="s", color=_DYNAMIC, linestyle="None",
                markersize=8, label=f"Dynamic obs ({len(grid.dynamic_obstacles)})")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved multi-robot grid to {save_path}")
    if show:
        plt.show()
    return fig


def draw_candidate_paths(
    grid: Grid,
    all_candidates: Dict[int, List[Dict]],
    title: str = "Candidate Paths per Robot",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a panel per robot showing all K candidate paths overlaid.
    Used to visualize what QAOA is choosing from.
    """
    rids = sorted(all_candidates.keys())
    n = len(rids)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    variant_colors = ["#3399FF", "#FF9933", "#33CC99", "#FF3366", "#9966FF"]

    for ax, rid in zip(axes, rids):
        # Base grid
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        ax.imshow(img, origin="upper", interpolation="nearest")

        # Grid lines
        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))

        # Draw each candidate path
        for k, cand in enumerate(all_candidates[rid]):
            path = cand["path"]
            color = variant_colors[k % len(variant_colors)]
            if len(path) >= 2:
                cols = [p[1] for p in path]
                rows = [p[0] for p in path]
                ax.plot(cols, rows, color=color, linewidth=2.0, alpha=0.7,
                        label=f"{cand['variant']} (cost={cand['cost']:.1f})")

        # Start & Goal
        cfg = [c for c in grid.robot_configs if c["id"] == rid][0]
        sr, sc = cfg["start"]
        gr, gc = cfg["goal"]
        robot_color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        ax.plot(sc, sr, marker="o", markersize=14, color=robot_color,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.plot(gc, gr, marker="*", markersize=18, color=robot_color,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Robot {rid}  ({len(all_candidates[rid])} candidates)",
                      fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved candidate paths to {save_path}")
    return fig


def draw_simulation_gif(
    grid: Grid,
    simulation_result,
    save_path: str,
    fps: int = 4,
    title: str = "Multi-Robot Simulation",
) -> None:
    """
    Animated GIF showing all robots moving simultaneously with
    dynamic obstacles.

    Parameters
    ----------
    grid : Grid
    simulation_result : SimulationResult (from dynamic_env.py)
    save_path : str    (must end with .gif)
    fps : int
    title : str
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[visualize] Could not import animation — skipping GIF.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    snapshots = simulation_result.ticks

    # Base image (static obstacles only)
    base_img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
    for r in range(grid.size):
        for c in range(grid.size):
            if grid.grid[r, c] == 1:
                base_img[r, c] = _STATIC

    im_obj = ax.imshow(base_img, origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, grid.size, 5))
    ax.set_yticks(np.arange(0, grid.size, 5))

    # Start/Goal markers for all robots
    for cfg in grid.robot_configs:
        rid = cfg["id"]
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        sr, sc = cfg["start"]
        gr, gc = cfg["goal"]
        ax.plot(sc, sr, marker="o", markersize=10, color=color,
                markeredgecolor="black", markeredgewidth=1, zorder=4, alpha=0.5)
        ax.plot(gc, gr, marker="*", markersize=14, color=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=4, alpha=0.5)

    # Robot dots (animated)
    robot_dots = {}
    for cfg in grid.robot_configs:
        rid = cfg["id"]
        color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
        dot, = ax.plot([], [], marker="o", markersize=14, color=color,
                       markeredgecolor="white", markeredgewidth=2, zorder=6)
        robot_dots[rid] = dot

    title_text = ax.set_title(title, fontsize=12, fontweight="bold")

    def _update(frame_idx):
        if frame_idx >= len(snapshots):
            return list(robot_dots.values()) + [im_obj]

        snap = snapshots[frame_idx]

        # Update image with dynamic obstacles
        frame_img = base_img.copy()
        for dpos in snap.dynamic_obstacle_positions:
            dr, dc = dpos
            if 0 <= dr < grid.size and 0 <= dc < grid.size:
                frame_img[dr, dc] = _DYNAMIC
        im_obj.set_data(frame_img)

        # Update robot positions
        for rs in snap.robot_states:
            rid = rs.id
            r, c = rs.pos
            robot_dots[rid].set_data([c], [r])

        # Show collisions in title
        n_col = len(snap.collisions)
        col_str = f"  [COLLISION x{n_col}]" if n_col > 0 else ""
        title_text.set_text(f"{title}  —  Tick {snap.tick}{col_str}")

        return list(robot_dots.values()) + [im_obj]

    anim = FuncAnimation(fig, _update, frames=len(snapshots),
                         interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[visualize] Saved simulation GIF to {save_path}")


def draw_selection_comparison(
    grid: Grid,
    all_candidates: Dict[int, List[Dict]],
    selections: Dict[str, Dict[int, int]],
    eval_results: Dict[str, Dict],
    title: str = "Path Selection Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side panels comparing different selection methods
    (greedy, random, brute-force, QAOA).

    Parameters
    ----------
    selections : dict
        method_name → {robot_id: candidate_index}
    eval_results : dict
        method_name → output of evaluate_selection()
    """
    methods = list(selections.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sel = selections[method]
        ev = eval_results[method]

        # Base grid
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        for obs in grid.dynamic_obstacles:
            pr, pc = obs["pos"]
            if 0 <= pr < grid.size and 0 <= pc < grid.size:
                img[pr, pc] = _DYNAMIC
        ax.imshow(img, origin="upper", interpolation="nearest")

        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))

        # Draw selected path for each robot
        for cfg in grid.robot_configs:
            rid = cfg["id"]
            color = ROBOT_COLORS[rid % len(ROBOT_COLORS)]
            cand_idx = sel[rid]
            path = all_candidates[rid][cand_idx]["path"]

            if len(path) >= 2:
                cols = [p[1] for p in path]
                rows = [p[0] for p in path]
                ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.8,
                        label=f"R{rid}")

            sr, sc = cfg["start"]
            gr, gc = cfg["goal"]
            ax.plot(sc, sr, marker="o", markersize=12, color=color,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=5)
            ax.plot(gc, gr, marker="*", markersize=16, color=color,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        ax.legend(loc="upper right", fontsize=8)
        subtitle = (f"{method}\n"
                    f"Cost: {ev['total_cost']:.1f}  |  "
                    f"Conflicts: {ev['total_conflicts']}  |  "
                    f"Score: {ev['score']:.1f}")
        ax.set_title(subtitle, fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved selection comparison to {save_path}")
    return fig


# ==================================================================
# Phase 3 — Quantum Q-Learning & Hybrid Planner Visualization
# ==================================================================

def draw_quantum_comparison(
    grid: Grid,
    results: Dict[str, dict],
    title: str = "Classical vs Quantum Q-Learning",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side panels comparing planners on the same grid.

    Parameters
    ----------
    grid : Grid
    results : dict
        Keys = planner name, Values = dict with keys:
            'path': list of Position (or None)
            'color': str (hex color)
            'info': str (subtitle text)
    title : str
    save_path : str or None
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        # Base grid
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        for obs in grid.dynamic_obstacles:
            pr, pc = obs["pos"]
            if 0 <= pr < grid.size and 0 <= pc < grid.size:
                img[pr, pc] = _DYNAMIC
        ax.imshow(img, origin="upper", interpolation="nearest")

        # Grid lines
        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))

        # Path
        path = res.get("path")
        color = res.get("color", "#3399FF")
        if path and len(path) >= 2:
            cols = [p[1] for p in path]
            rows = [p[0] for p in path]
            ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.85)

        # Start & Goal
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker="o", markersize=12, color="limegreen",
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(gc, gr, marker="*", markersize=16, color="red",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        info = res.get("info", "")
        ax.set_title(f"{name}\n{info}", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved quantum comparison to {save_path}")
    return fig


def draw_learning_curves(
    curves: Dict[str, List[float]],
    title: str = "Training Reward Curves",
    save_path: Optional[str] = None,
    window: int = 20,
) -> plt.Figure:
    """
    Plot smoothed learning curves (reward per episode) for
    classical vs quantum Q-learning.

    Parameters
    ----------
    curves : dict
        Keys = label, Values = list of episode rewards.
    title : str
    save_path : str or None
    window : int
        Moving-average window for smoothing.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3399FF", "#FF6633", "#33CC66", "#CC33FF"]

    for i, (label, rewards) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        # Raw (faint)
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)
        # Smoothed
        if len(rewards) >= window:
            smoothed = np.convolve(rewards,
                                   np.ones(window) / window,
                                   mode="valid")
            ax.plot(range(window - 1, len(rewards)),
                    smoothed, color=color, linewidth=2, label=label)
        else:
            ax.plot(rewards, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved learning curves to {save_path}")
    return fig


# ==================================================================
# Full Model Comparison — Bar Charts across obstacle densities
# ==================================================================

def draw_full_comparison(
    comparison_data: Dict[str, Dict[str, dict]],
    title: str = "All Planners — Performance vs Obstacle Density",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw a multi-panel bar chart comparing ALL planners across
    different obstacle densities.

    Parameters
    ----------
    comparison_data : dict
        Structure:
            { density_label: {
                planner_name: {
                    "cost": float,
                    "success": bool,
                    "time_ms": float,
                    "steps": int,   # path length or steps taken
                },
                ...
              },
              ...
            }
    title : str
    save_path : str or None
    """
    densities = list(comparison_data.keys())
    # Collect all planner names (union across densities)
    all_planners = []
    for d in densities:
        for p in comparison_data[d]:
            if p not in all_planners:
                all_planners.append(p)

    planner_colors = {
        "A*": "#3399FF",
        "APF": "#FF6633",
        "Classical QL": "#33CC66",
        "Quantum QL": "#CC33FF",
        "Hybrid": "#FF9933",
    }
    default_colors = ["#3399FF", "#FF6633", "#33CC66", "#CC33FF", "#FF9933",
                      "#FFCC00", "#66CCCC"]

    def _color(name):
        return planner_colors.get(name,
               default_colors[all_planners.index(name) % len(default_colors)])

    n_densities = len(densities)
    n_planners = len(all_planners)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    bar_width = 0.8 / n_planners
    x = np.arange(n_densities)

    # ---- Panel 1: Path Cost ----
    ax1 = axes[0]
    for i, pname in enumerate(all_planners):
        costs = []
        for d in densities:
            info = comparison_data[d].get(pname, {})
            c = info.get("cost", 0)
            if not info.get("success", False):
                c = 0
            costs.append(c)
        bars = ax1.bar(x + i * bar_width, costs, bar_width,
                       label=pname, color=_color(pname), edgecolor="white")
        # Mark failures with an X
        for j, d in enumerate(densities):
            info = comparison_data[d].get(pname, {})
            if not info.get("success", False):
                ax1.text(x[j] + i * bar_width, 1, "X",
                         ha="center", va="bottom", fontsize=12,
                         fontweight="bold", color="red")
    ax1.set_xlabel("Obstacle Density", fontsize=11)
    ax1.set_ylabel("Path Cost", fontsize=11)
    ax1.set_title("Path Cost (lower = better)\nX = failed", fontsize=12, fontweight="bold")
    ax1.set_xticks(x + bar_width * (n_planners - 1) / 2)
    ax1.set_xticklabels(densities, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Panel 2: Computation Time ----
    ax2 = axes[1]
    for i, pname in enumerate(all_planners):
        times = []
        for d in densities:
            info = comparison_data[d].get(pname, {})
            times.append(info.get("time_ms", 0))
        ax2.bar(x + i * bar_width, times, bar_width,
                label=pname, color=_color(pname), edgecolor="white")
    ax2.set_xlabel("Obstacle Density", fontsize=11)
    ax2.set_ylabel("Time (ms)", fontsize=11)
    ax2.set_title("Computation Time", fontsize=12, fontweight="bold")
    ax2.set_xticks(x + bar_width * (n_planners - 1) / 2)
    ax2.set_xticklabels(densities, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ---- Panel 3: Path Length (steps) ----
    ax3 = axes[2]
    for i, pname in enumerate(all_planners):
        steps = []
        for d in densities:
            info = comparison_data[d].get(pname, {})
            s = info.get("steps", 0)
            if not info.get("success", False):
                s = 0
            steps.append(s)
        bars = ax3.bar(x + i * bar_width, steps, bar_width,
                       label=pname, color=_color(pname), edgecolor="white")
        for j, d in enumerate(densities):
            info = comparison_data[d].get(pname, {})
            if not info.get("success", False):
                ax3.text(x[j] + i * bar_width, 1, "X",
                         ha="center", va="bottom", fontsize=12,
                         fontweight="bold", color="red")
    ax3.set_xlabel("Obstacle Density", fontsize=11)
    ax3.set_ylabel("Path Length (cells)", fontsize=11)
    ax3.set_title("Path Length (lower = better)\nX = failed", fontsize=12, fontweight="bold")
    ax3.set_xticks(x + bar_width * (n_planners - 1) / 2)
    ax3.set_xticklabels(densities, fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved full comparison to {save_path}")
    return fig


def draw_varied_obstacle_grids(
    grids: Dict[str, "Grid"],
    title: str = "Environments with Varying Obstacle Density",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show side-by-side grid snapshots at different obstacle densities.

    Parameters
    ----------
    grids : dict
        density_label → Grid object
    """
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (label, grid) in zip(axes, grids.items()):
        img = np.ones((grid.size, grid.size, 3), dtype=np.float32)
        for r in range(grid.size):
            for c in range(grid.size):
                if grid.grid[r, c] == 1:
                    img[r, c] = _STATIC
        ax.imshow(img, origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.size, 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(0, grid.size, 5))
        ax.set_yticks(np.arange(0, grid.size, 5))

        # Start & Goal
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker="o", markersize=12, color="limegreen",
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(gc, gr, marker="*", markersize=16, color="red",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        obs_count = int(np.sum(grid.grid == 1))
        total = grid.size * grid.size
        actual_pct = obs_count / total * 100
        ax.set_title(f"{label}\n({obs_count} cells, {actual_pct:.0f}%)",
                     fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Saved varied obstacle grids to {save_path}")
    return fig
