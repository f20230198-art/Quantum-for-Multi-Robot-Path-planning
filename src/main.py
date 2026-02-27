"""
main.py  –  One-Click Demo (Week 1)
====================================

WHAT THIS FILE DOES:
    Run with:   python main.py
    It will:
        1. Create a RANDOM 20×20 grid (different every run).
        2. Run A* to find the shortest path.
        3. Show the A* search process step-by-step (image strip).
        4. Create an animated GIF of the robot walking the path.
        5. Compare A* vs Greedy BFS side-by-side (proof that
           smart heuristic-informed search matters — preview of
           why quantum-enhanced search will matter even more).
        6. Also run the trap & narrow-passage scenarios.
        7. Save everything to  experiments/results/

    Each run produces a DIFFERENT random grid.
"""

import sys
import os
import time
import random

# Use non-interactive backend so plt.show() won't block the terminal.
import matplotlib
matplotlib.use("Agg")

# Make sure Python can find our src/ modules when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from grid import Grid
from a_star import a_star, euclidean
from visualize import (draw_grid, draw_search_steps, draw_robot_walk_gif,
                       greedy_bfs, draw_algorithm_comparison)


def main():
    print("=" * 60)
    print("  QUANTUM MULTI-ROBOT PATH PLANNING  –  Week 1 Demo")
    print("=" * 60)

    # ---- output folder ----
    save_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. Create a RANDOM grid (no fixed seed) ----
    seed = random.randint(0, 999_999)
    grid = Grid(size=20, obstacle_ratio=0.15, seed=seed)
    print(f"\n[grid]  Random seed = {seed}")
    print(f"[grid]  Created {grid.size}×{grid.size} grid  "
          f"(obstacle ratio = {grid.obstacle_ratio})")
    print(f"[grid]  Start = {grid.start}   Goal = {grid.goal}")
    print(f"\n{grid}\n")

    # ---- 2. Run A* ----
    t0 = time.perf_counter()
    result = a_star(grid, heuristic=euclidean, eight_connected=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"[A*]   {result}")
    print(f"[A*]   Time: {elapsed_ms:.2f} ms")

    if not result.success:
        print("[A*]   ⚠  No path found.")
        return

    print(f"[A*]   Path length: {len(result.path)} cells")
    print(f"[A*]   Path cost:   {result.cost:.2f}")

    # ---- 3. Static path image ----
    draw_grid(
        grid,
        paths=[result.path],
        title=f"A* Path  (cost={result.cost:.2f}, expanded={result.nodes_expanded})",
        save_path=os.path.join(save_dir, "week1_demo.png"),
        show=False,
    )

    # ---- 4. Step-by-step search process ----
    print("\n[viz]  Generating A* search step-by-step strip...")
    draw_search_steps(
        grid,
        expansion_order=result.expansion_order,
        final_path=result.path,
        num_snapshots=6,
        title="A* Search Process — Step by Step",
        save_path=os.path.join(save_dir, "week1_search_steps.png"),
    )

    # ---- 5. Animated GIF of robot walking the path ----
    print("[viz]  Generating robot walk GIF...")
    draw_robot_walk_gif(
        grid,
        path=result.path,
        save_path=os.path.join(save_dir, "week1_robot_walk.gif"),
        fps=4,
        title="Robot Walking A* Path",
    )

    # ---- 6. Algorithm comparison: A* vs Greedy BFS ----
    print("\n[compare]  Running Greedy BFS on same grid...")
    gbfs = greedy_bfs(grid)
    print(f"[Greedy]   Nodes expanded: {gbfs['nodes_expanded']}"
          f"   Path cost: {gbfs['cost']:.2f}" if gbfs['path'] else "   FAILED")

    draw_algorithm_comparison(
        grid,
        results={
            "A* (Optimal)": {
                "path": result.path,
                "cost": result.cost,
                "nodes_expanded": result.nodes_expanded,
                "expansion_order": result.expansion_order,
            },
            "Greedy BFS (Non-optimal)": gbfs,
        },
        title="Traditional Planners — A* vs Greedy BFS  (same grid)",
        save_path=os.path.join(save_dir, "week1_comparison.png"),
    )

    print(f"\n✓  Main demo complete!")

    # ---- 7. Trap & Narrow-Passage Scenarios ----
    print("\n" + "-" * 60)
    print("  Bonus: Trap & Narrow-Passage Scenarios")
    print("-" * 60)

    for name, env_fn in [("Trap (U-shape)", Grid.create_trap_environment),
                          ("Narrow Passage", Grid.create_narrow_passage)]:
        env = env_fn(size=20)
        res = a_star(env)
        status = f"cost={res.cost:.2f}" if res.success else "NO PATH"
        print(f"  {name:20s}  →  {status}  (expanded {res.nodes_expanded} nodes)")
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        draw_grid(
            env,
            paths=[res.path] if res.success else None,
            title=f"{name}  –  A* ({status})",
            save_path=os.path.join(save_dir, f"week1_{fname}.png"),
            show=False,
        )

    print("\n" + "=" * 60)
    print("  ALL OUTPUTS SAVED TO experiments/results/")
    print("=" * 60)
    print("  Files generated:")
    print("    • week1_demo.png            — A* path on random grid")
    print("    • week1_search_steps.png    — A* step-by-step process")
    print("    • week1_robot_walk.gif      — Animated robot walking")
    print("    • week1_comparison.png      — A* vs Greedy BFS comparison")
    print("    • week1_trap_u-shape.png    — Trap scenario")
    print("    • week1_narrow_passage.png  — Narrow passage scenario")


if __name__ == "__main__":
    main()
