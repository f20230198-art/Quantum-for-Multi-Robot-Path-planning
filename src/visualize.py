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
