"""
run_classical.py  –  Generate all classical algorithm results
==============================================================

Covers:
    Week 1: A* search, Greedy BFS, trap/narrow-passage scenarios
    Phase 1: Multi-robot environment, candidate paths, classical selection
    Phase 2: APF (local), Q-Learning (RL baseline), 3-way comparison

All outputs saved to experiments/results/01_classical_baseline/
                    experiments/results/02_multi_robot_challenge/
                    experiments/results/03_classical_limitations/
"""

import os
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from apf import apf_plan
from q_learning import q_learning_full, train_q_learning


def run_classical(base_dir):
    """Run all classical algorithm demos and save results."""

    dir1 = os.path.join(base_dir, "01_classical_baseline")
    dir2 = os.path.join(base_dir, "02_multi_robot_challenge")
    dir3 = os.path.join(base_dir, "03_classical_limitations")
    for d in [dir1, dir2, dir3]:
        os.makedirs(d, exist_ok=True)

    # ==============================================================
    #  WEEK 1 — A* Baseline
    # ==============================================================

    print("\n" + "=" * 60)
    print("  WEEK 1 — A* Search Baseline")
    print("=" * 60)

    seed = random.randint(0, 999_999)
    grid = Grid(size=20, obstacle_ratio=0.15, seed=seed)
    print(f"\n[grid]  Random seed = {seed}")
    print(f"[grid]  Created {grid.size}x{grid.size} grid")

    # A* search
    t0 = time.perf_counter()
    result = a_star(grid, heuristic=euclidean, eight_connected=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[A*]   {result}")
    print(f"[A*]   Time: {elapsed_ms:.2f} ms")

    if not result.success:
        print("[A*]   No path found — skipping Week 1 visuals.")
        return None

    # Static path image
    draw_grid(grid, paths=[result.path],
              title=f"A* Path  (cost={result.cost:.2f}, expanded={result.nodes_expanded})",
              save_path=os.path.join(dir1, "astar_path.png"), show=False)

    # Step-by-step search
    draw_search_steps(grid, expansion_order=result.expansion_order,
                      final_path=result.path, num_snapshots=6,
                      title="A* Search Process — Step by Step",
                      save_path=os.path.join(dir1, "astar_search_steps.png"))

    # Robot walk GIF
    draw_robot_walk_gif(grid, path=result.path,
                        save_path=os.path.join(dir1, "astar_walk.gif"),
                        fps=4, title="Robot Walking A* Path")

    # A* vs Greedy BFS
    gbfs = greedy_bfs(grid)
    draw_algorithm_comparison(grid, results={
        "A* (Optimal)": {
            "path": result.path, "cost": result.cost,
            "nodes_expanded": result.nodes_expanded,
            "expansion_order": result.expansion_order,
        },
        "Greedy BFS (Non-optimal)": gbfs,
    }, title="A* vs Greedy BFS (same grid)",
       save_path=os.path.join(dir1, "astar_vs_greedy.png"))

    # Trap & narrow passage
    for name, env_fn in [("Trap (U-shape)", Grid.create_trap_environment),
                          ("Narrow Passage", Grid.create_narrow_passage)]:
        env = env_fn(size=20)
        res = a_star(env)
        status = f"cost={res.cost:.2f}" if res.success else "NO PATH"
        print(f"  {name:20s}  ->  {status}")
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        draw_grid(env, paths=[res.path] if res.success else None,
                  title=f"{name} — A* ({status})",
                  save_path=os.path.join(dir1, f"{fname}.png"), show=False)

    # ==============================================================
    #  PHASE 1 — Multi-Robot + Dynamic Environment
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 1 — Multi-Robot Dynamic Environment")
    print("=" * 60)

    mr_grid = Grid.create_multi_robot_env(
        size=20, obstacle_ratio=0.12, num_robots=3, num_dynamic=4, seed=42)
    for cfg in mr_grid.robot_configs:
        print(f"  Robot {cfg['id']}: {cfg['start']} -> {cfg['goal']}")

    draw_multi_robot_grid(mr_grid,
        title="Multi-Robot Environment (3 robots, 4 dynamic obstacles)",
        save_path=os.path.join(dir2, "environment.png"))

    qaoa_input = prepare_qaoa_input(mr_grid, num_candidates=4, seed=42)
    all_candidates = qaoa_input["all_candidates"]

    draw_candidate_paths(mr_grid, all_candidates,
        title="Candidate Paths per Robot (4 options each)",
        save_path=os.path.join(dir2, "candidate_paths.png"))

    # Classical selection
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
        print(f"  {name:25s}  cost={ev['total_cost']:.1f}  "
              f"conflicts={ev['total_conflicts']}  score={ev['score']:.1f}")

    draw_selection_comparison(mr_grid, all_candidates, selections, eval_results,
        title="Classical Selection Methods — Comparison",
        save_path=os.path.join(dir2, "selection_comparison.png"))

    # Simulation
    bf_paths = {}
    for cfg in mr_grid.robot_configs:
        rid = cfg["id"]
        bf_paths[rid] = all_candidates[rid][sel_brute[rid]]["path"]

    sim = Simulation(mr_grid, bf_paths, max_ticks=200)
    sim_result = sim.run()
    print(f"  Simulation: {sim_result.total_ticks} ticks, "
          f"collisions={sim_result.total_collisions}")

    draw_simulation_gif(mr_grid, sim_result,
        save_path=os.path.join(dir2, "simulation.gif"),
        fps=4, title="Multi-Robot Simulation (Brute-Force)")

    # ==============================================================
    #  PHASE 2 — APF & Q-Learning (Classical Limitations)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 2 — Classical Limitations (APF & Q-Learning)")
    print("=" * 60)

    trap_env = Grid.create_trap_environment(size=20)

    # APF on trap
    apf_result = apf_plan(trap_env)
    print(f"  APF on U-trap: {'STUCK!' if apf_result.stuck else 'OK'} "
          f"(steps={apf_result.steps_taken})")
    draw_grid(trap_env, paths=[apf_result.path] if apf_result.path else None,
              path_colors=["#FF6633"],
              title=f"APF on U-Trap — {'STUCK!' if apf_result.stuck else 'OK'}",
              save_path=os.path.join(dir3, "apf_stuck_in_trap.png"), show=False)

    # Q-Learning on trap
    ql_result = q_learning_full(trap_env, episodes=500, seed=42)
    print(f"  Q-Learning on U-trap: {'OK' if ql_result.success else 'FAIL'} "
          f"(cost={ql_result.cost:.1f})")
    draw_grid(trap_env, paths=[ql_result.path] if ql_result.path else None,
              path_colors=["#33CC66"],
              title=f"Q-Learning on U-Trap — "
                    f"{'SUCCESS' if ql_result.success else 'FAIL'}",
              save_path=os.path.join(dir3, "qlearning_trap.png"), show=False)

    # 3-way comparison on standard grid
    compare_grid = Grid(size=20, obstacle_ratio=0.15, seed=12345)
    compare_grid.add_dynamic_obstacle((10, 5), direction=(0, 1), pattern="bounce")
    compare_grid.add_dynamic_obstacle((5, 10), direction=(1, 0), pattern="bounce")

    astar_cmp = a_star(compare_grid)
    apf_cmp = apf_plan(compare_grid)
    ql_cmp = q_learning_full(compare_grid, episodes=500, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    planners = [
        ("A* (Global Optimal)", astar_cmp.path, "#3399FF",
         f"cost={astar_cmp.cost:.1f}, expanded={astar_cmp.nodes_expanded}"),
        ("APF (Local Reactive)", apf_cmp.path, "#FF6633",
         f"{'STUCK' if apf_cmp.stuck else f'cost={apf_cmp.cost:.1f}'}, "
         f"steps={apf_cmp.steps_taken}"),
        ("Q-Learning (RL Baseline)", ql_cmp.path, "#33CC66",
         f"{'OK' if ql_cmp.success else 'FAIL'}, cost={ql_cmp.cost:.1f}"),
    ]
    for ax, (name, path, color, info) in zip(axes, planners):
        draw_grid(compare_grid, paths=[path] if path else None,
                  path_colors=[color], title=f"{name}\n{info}", show=False, ax=ax)
    fig.suptitle("Classical Planners — Each Has Limitations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(dir3, "classical_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  Classical results saved to:")
    print(f"    {dir1}/")
    print(f"    {dir2}/")
    print(f"    {dir3}/")

    # Return shared data needed by quantum phase
    return {
        "mr_grid": mr_grid,
        "all_candidates": all_candidates,
        "sel_greedy": sel_greedy,
        "sel_random": sel_random,
        "sel_brute": sel_brute,
        "trap_env": trap_env,
        "ql_result": ql_result,
        "compare_grid": compare_grid,
    }
