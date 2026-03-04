"""
main.py  –  One-Click Demo
============================

WHAT THIS FILE DOES:
    Run with:   python main.py
    It will:
        WEEK 1 (single robot):
            1. Create a random 20×20 grid
            2. Run A* to find shortest path
            3. Show step-by-step search + animated GIF
            4. A* vs Greedy BFS comparison
            5. Trap + narrow passage scenarios

        PHASE 1  (multi-robot + dynamic environment):
            6. Create a 20×20 grid with 3 robots + 4 dynamic obstacles
            7. Generate 4 candidate paths per robot
            8. Visualize candidate paths
            9. Compare classical selection methods (greedy/random/brute-force)
            10. Run time-stepped simulation with selected paths
            11. Generate multi-robot animated GIF

    All outputs saved to experiments/results/
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
                       greedy_bfs, draw_algorithm_comparison,
                       draw_multi_robot_grid, draw_candidate_paths,
                       draw_selection_comparison, draw_simulation_gif)
from multi_robot import (generate_candidate_paths, prepare_qaoa_input,
                         greedy_select, random_select, brute_force_select,
                         evaluate_selection)
from dynamic_env import Simulation


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
    print("  ALL WEEK 1 OUTPUTS SAVED")
    print("=" * 60)

    # ==============================================================
    #  PHASE 1 — Multi-Robot + Dynamic Environment
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 1 — Multi-Robot Dynamic Environment")
    print("=" * 60)

    # ---- 8. Create multi-robot environment ----
    print("\n[env]  Creating 20×20 grid with 3 robots + 4 dynamic obstacles...")
    mr_grid = Grid.create_multi_robot_env(
        size=20, obstacle_ratio=0.12,
        num_robots=3, num_dynamic=4, seed=42,
    )
    print(f"[env]  Robots:")
    for cfg in mr_grid.robot_configs:
        print(f"         Robot {cfg['id']}: {cfg['start']} → {cfg['goal']}")
    print(f"[env]  Dynamic obstacles: {len(mr_grid.dynamic_obstacles)}")

    # ---- 9. Visualize the environment (before paths) ----
    draw_multi_robot_grid(
        mr_grid,
        title="Multi-Robot Environment  (3 robots, 4 dynamic obstacles)",
        save_path=os.path.join(save_dir, "phase1_environment.png"),
    )

    # ---- 10. Generate candidate paths ----
    print("\n[paths]  Generating candidate paths per robot...")
    qaoa_input = prepare_qaoa_input(mr_grid, num_candidates=4, seed=42)
    all_candidates = qaoa_input["all_candidates"]

    # ---- 11. Visualize candidate paths ----
    draw_candidate_paths(
        mr_grid,
        all_candidates,
        title="Candidate Paths per Robot  (4 options each)",
        save_path=os.path.join(save_dir, "phase1_candidate_paths.png"),
    )

    # ---- 12. Classical selection methods ----
    print("\n[select]  Running classical baseline selectors...")
    sel_greedy = greedy_select(all_candidates)
    sel_random = random_select(all_candidates, seed=42)
    sel_brute  = brute_force_select(all_candidates, conflict_penalty=10.0)

    selections = {
        "Greedy (shortest)": sel_greedy,
        "Random": sel_random,
        "Brute-Force (optimal)": sel_brute,
    }
    eval_results = {}
    for name, sel in selections.items():
        ev = evaluate_selection(all_candidates, sel, conflict_penalty=10.0)
        eval_results[name] = ev
        print(f"  {name:25s}  →  cost={ev['total_cost']:.1f}  "
              f"conflicts={ev['total_conflicts']}  score={ev['score']:.1f}")

    # ---- 13. Visualize selection comparison ----
    draw_selection_comparison(
        mr_grid,
        all_candidates,
        selections,
        eval_results,
        title="Classical Selection Methods — Comparison",
        save_path=os.path.join(save_dir, "phase1_selection_comparison.png"),
    )

    # ---- 14. Run simulation with brute-force (best classical) paths ----
    print("\n[sim]  Running time-stepped simulation with brute-force selection...")
    bf_paths = {}
    for cfg in mr_grid.robot_configs:
        rid = cfg["id"]
        chosen_idx = sel_brute[rid]
        bf_paths[rid] = all_candidates[rid][chosen_idx]["path"]

    sim = Simulation(mr_grid, bf_paths, max_ticks=200)
    sim_result = sim.run()
    print(f"[sim]  Completed in {sim_result.total_ticks} ticks")
    print(f"[sim]  All reached goal: {sim_result.all_reached_goal}")
    print(f"[sim]  Total collisions: {sim_result.total_collisions}")
    for rid, stats in sim_result.per_robot_stats.items():
        print(f"         Robot {rid}: path_len={stats['path_length']}, "
              f"time={stats['time_to_goal']}, collisions={stats['collisions']}")

    # ---- 15. Visualize the environment with selected paths ----
    draw_multi_robot_grid(
        mr_grid,
        robot_paths=bf_paths,
        title=f"Brute-Force Selection  (cost={eval_results['Brute-Force (optimal)']['total_cost']:.1f}, "
              f"conflicts={eval_results['Brute-Force (optimal)']['total_conflicts']})",
        save_path=os.path.join(save_dir, "phase1_bruteforce_paths.png"),
    )

    # ---- 16. Animated simulation GIF ----
    print("\n[viz]  Generating multi-robot simulation GIF...")
    draw_simulation_gif(
        mr_grid,
        sim_result,
        save_path=os.path.join(save_dir, "phase1_simulation.gif"),
        fps=4,
        title="Multi-Robot Simulation (Brute-Force)",
    )

    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE — ALL OUTPUTS SAVED")
    print("=" * 60)
    print("  Files generated:")
    print("    • week1_demo.png                  — A* on random grid")
    print("    • week1_search_steps.png          — A* step-by-step")
    print("    • week1_robot_walk.gif            — Single robot animation")
    print("    • week1_comparison.png            — A* vs Greedy BFS")
    print("    • week1_trap_u-shape.png          — Trap scenario")
    print("    • week1_narrow_passage.png        — Narrow passage")
    print("    • phase1_environment.png          — Multi-robot grid")
    print("    • phase1_candidate_paths.png      — K candidate paths/robot")
    print("    • phase1_selection_comparison.png  — Greedy vs Random vs BF")
    print("    • phase1_bruteforce_paths.png     — Best classical selection")
    print("    • phase1_simulation.gif           — Animated simulation")
    print("\n  Next: Phase 2 — QAOA quantum optimizer (qaoa_optimizer.py)")


if __name__ == "__main__":
    main()
