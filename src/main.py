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

        PHASE 3  (quantum Q-learning + hybrid planner):
            12. Run quantum Q-learning on U-shaped trap
            13. Compare classical vs quantum Q-learning (path + learning curves)
            14. Run hybrid planner (A* + APF + Quantum QL) on trap
            15. Full 4-way planner comparison on standard grid

    All outputs saved to experiments/results/
"""

import sys
import os
import time
import math
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
from apf import apf_plan
from q_learning import q_learning_full, train_q_learning
from quantum_q_learning import (quantum_q_learning_full,
                                 train_quantum_q_learning,
                                 verify_qiskit_match)
from hybrid_planner import hybrid_plan
from visualize import (draw_quantum_comparison, draw_learning_curves,
                       draw_full_comparison, draw_varied_obstacle_grids)

import numpy as np
import matplotlib.pyplot as plt


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

    # ==============================================================
    #  PHASE 2 — APF & Q-Learning Local Planners
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 2 — APF & Q-Learning Local Planners")
    print("=" * 60)

    # ---- 17. APF on U-shaped trap (demonstrates local minimum) ----
    print("\n[APF]  Running APF on U-shaped trap environment...")
    trap_env = Grid.create_trap_environment(size=20)

    t0 = time.perf_counter()
    apf_result = apf_plan(trap_env)
    apf_ms = (time.perf_counter() - t0) * 1000
    print(f"[APF]  {apf_result}")
    print(f"[APF]  Time: {apf_ms:.2f} ms")

    # Save APF trap result
    draw_grid(
        trap_env,
        paths=[apf_result.path] if apf_result.path else None,
        path_colors=["#FF6633"],
        title=f"APF on U-Trap  –  {'STUCK!' if apf_result.stuck else 'OK'}  "
              f"(steps={apf_result.steps_taken})",
        save_path=os.path.join(save_dir, "phase2_apf_trap.png"),
        show=False,
    )

    # Also run A* on the same trap for comparison
    astar_trap = a_star(trap_env)
    print(f"[A*]   A* on same trap: cost={astar_trap.cost:.2f} "
          f"(expanded {astar_trap.nodes_expanded} nodes)")

    # ---- 18. Q-Learning on U-shaped trap ----
    print("\n[QL]   Training Q-learning on U-shaped trap (500 episodes)...")
    t0 = time.perf_counter()
    ql_result = q_learning_full(
        trap_env,
        episodes=500,
        seed=42,
    )
    ql_ms = (time.perf_counter() - t0) * 1000
    print(f"[QL]   {ql_result}")
    print(f"[QL]   Time: {ql_ms:.2f} ms (training + planning)")

    # Save Q-learning trap result
    draw_grid(
        trap_env,
        paths=[ql_result.path] if ql_result.path else None,
        path_colors=["#33CC66"],
        title=f"Q-Learning on U-Trap  –  "
              f"{'SUCCESS' if ql_result.success else 'FAIL'}  "
              f"(episodes={ql_result.episodes_trained})",
        save_path=os.path.join(save_dir, "phase2_qlearning_trap.png"),
        show=False,
    )

    # ---- 19. Three-way comparison on a standard dynamic grid ----
    print("\n[compare]  Running all 3 planners on a standard grid...")
    compare_grid = Grid(size=20, obstacle_ratio=0.15, seed=12345)
    compare_grid.add_dynamic_obstacle((10, 5), direction=(0, 1), pattern="bounce")
    compare_grid.add_dynamic_obstacle((5, 10), direction=(1, 0), pattern="bounce")
    compare_grid.add_dynamic_obstacle((15, 15), direction=(0, -1), pattern="bounce")

    # A*
    astar_cmp = a_star(compare_grid)
    # APF
    apf_cmp = apf_plan(compare_grid)
    # Q-Learning
    ql_cmp = q_learning_full(compare_grid, episodes=500, seed=42)

    print(f"  {'Algorithm':<20s}  {'Success':<10s}  {'Cost':<10s}  {'Steps/Expanded'}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*15}")
    print(f"  {'A*':<20s}  {'✓' if astar_cmp.success else '✗':<10s}  "
          f"{astar_cmp.cost:<10.2f}  {astar_cmp.nodes_expanded}")
    print(f"  {'APF':<20s}  {'✓' if apf_cmp.success else '✗':<10s}  "
          f"{apf_cmp.cost:<10.2f}  {apf_cmp.steps_taken}")
    print(f"  {'Q-Learning':<20s}  {'✓' if ql_cmp.success else '✗':<10s}  "
          f"{ql_cmp.cost:<10.2f}  {ql_cmp.episodes_trained} ep")

    # Comparison image
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    planners = [
        ("A* (Global Optimal)", astar_cmp.path, "#3399FF",
         f"cost={astar_cmp.cost:.1f}, expanded={astar_cmp.nodes_expanded}"),
        ("APF (Local Reactive)", apf_cmp.path, "#FF6633",
         f"{'STUCK' if apf_cmp.stuck else f'cost={apf_cmp.cost:.1f}'}, "
         f"steps={apf_cmp.steps_taken}"),
        ("Q-Learning (RL Baseline)", ql_cmp.path, "#33CC66",
         f"{'OK' if ql_cmp.success else 'FAIL'}, "
         f"cost={ql_cmp.cost:.1f}, ep={ql_cmp.episodes_trained}"),
    ]
    for ax, (name, path, color, info) in zip(axes, planners):
        draw_grid(compare_grid, paths=[path] if path else None,
                  path_colors=[color], title=f"{name}\n{info}",
                  show=False, ax=ax)

    fig.suptitle("Phase 2 — Planner Comparison (A* vs APF vs Q-Learning)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cmp_path = os.path.join(save_dir, "phase2_planner_comparison.png")
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[visualize] Saved planner comparison to {cmp_path}")

    # ---- 20. Summary ----
    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE — ALL OUTPUTS SAVED")
    print("=" * 60)
    print("  New files generated:")
    print("    • phase2_apf_trap.png             — APF stuck in U-trap")
    print("    • phase2_qlearning_trap.png        — Q-Learning on U-trap")
    print("    • phase2_planner_comparison.png    — A* vs APF vs Q-Learning")

    # ==============================================================
    #  PHASE 3 — Quantum Q-Learning + Hybrid Planner
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 3 — Quantum Q-Learning + Hybrid Planner")
    print("=" * 60)

    # ---- 21. Qiskit verification (if available) ----
    print("\n[Qiskit]  Verifying quantum backend...")
    for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
        v = verify_qiskit_match(theta, shots=1000)
        if v["qiskit_available"]:
            print(f"  θ={theta:.4f}  numpy={v['numpy_prob']:.4f}  "
                  f"qiskit={v['qiskit_prob']:.4f}  "
                  f"diff={v['difference']:.4f}")
        else:
            print(f"  θ={theta:.4f}  numpy={v['numpy_prob']:.4f}  "
                  f"(Qiskit not installed — using numpy backend)")

    # ---- 22. Quantum Q-Learning on U-shaped trap ----
    print("\n[QQL]  Training Quantum Q-Learning on U-shaped trap (500 episodes)...")
    t0 = time.perf_counter()
    qql_result = quantum_q_learning_full(
        trap_env,
        episodes=500,
        seed=42,
    )
    qql_ms = (time.perf_counter() - t0) * 1000
    print(f"[QQL]  {qql_result}")
    print(f"[QQL]  Time: {qql_ms:.2f} ms (training + planning)")

    # Save quantum QL trap result
    draw_grid(
        trap_env,
        paths=[qql_result.path] if qql_result.path else None,
        path_colors=["#CC33FF"],
        title=f"Quantum Q-Learning on U-Trap  –  "
              f"{'SUCCESS' if qql_result.success else 'FAIL'}  "
              f"(episodes={qql_result.episodes_trained})",
        save_path=os.path.join(save_dir, "phase3_quantum_ql_trap.png"),
        show=False,
    )

    # ---- 23. Classical vs Quantum Q-Learning comparison ----
    print("\n[compare]  Classical vs Quantum Q-Learning on same grid...")
    draw_quantum_comparison(
        trap_env,
        results={
            "Classical Q-Learning": {
                "path": ql_result.path,
                "color": "#33CC66",
                "info": f"{'OK' if ql_result.success else 'FAIL'}, "
                        f"cost={ql_result.cost:.1f}, ep={ql_result.episodes_trained}",
            },
            "Quantum Q-Learning": {
                "path": qql_result.path,
                "color": "#CC33FF",
                "info": f"{'OK' if qql_result.success else 'FAIL'}, "
                        f"cost={qql_result.cost:.1f}, ep={qql_result.episodes_trained}",
            },
        },
        title="Phase 3 — Classical vs Quantum Q-Learning (U-Trap)",
        save_path=os.path.join(save_dir, "phase3_ql_comparison.png"),
    )

    # ---- 24. Learning curves comparison ----
    print("[compare]  Generating learning curves...")
    # Re-train both to get reward histories for the same grid
    _, cl_rewards = train_q_learning(
        trap_env, episodes=500, seed=42,
    )
    _, qq_rewards = train_quantum_q_learning(
        trap_env, episodes=500, seed=42,
    )
    draw_learning_curves(
        curves={
            "Classical Q-Learning": cl_rewards,
            "Quantum Q-Learning": qq_rewards,
        },
        title="Phase 3 — Training Reward Curves (U-Trap, 500 episodes)",
        save_path=os.path.join(save_dir, "phase3_learning_curves.png"),
    )

    # ---- 25. Hybrid planner on U-shaped trap ----
    print("\n[Hybrid]  Running hybrid planner (A* + APF + Quantum QL) on U-trap...")
    t0 = time.perf_counter()
    hybrid_result = hybrid_plan(
        trap_env,
        q_episodes=500,
        q_seed=42,
    )
    hybrid_ms = (time.perf_counter() - t0) * 1000
    print(f"[Hybrid]  {hybrid_result}")
    print(f"[Hybrid]  Time: {hybrid_ms:.2f} ms")

    draw_grid(
        trap_env,
        paths=[hybrid_result.path] if hybrid_result.path else None,
        path_colors=["#FF9933"],
        title=f"Hybrid Planner on U-Trap  –  "
              f"{'SUCCESS' if hybrid_result.success else 'FAIL'}  "
              f"(A*={hybrid_result.astar_replans}, "
              f"APF={hybrid_result.apf_steps}, "
              f"QEscape={hybrid_result.quantum_escapes})",
        save_path=os.path.join(save_dir, "phase3_hybrid_planner.png"),
        show=False,
    )

    # ---- 26. Four-way comparison on a standard grid ----
    print("\n[compare]  Running all 4 planners on a standard grid...")

    # A*
    astar_cmp4 = a_star(compare_grid)
    # APF
    apf_cmp4 = apf_plan(compare_grid)
    # Classical Q-Learning
    ql_cmp4 = q_learning_full(compare_grid, episodes=500, seed=42)
    # Quantum Q-Learning
    qql_cmp4 = quantum_q_learning_full(compare_grid, episodes=500, seed=42)

    print(f"  {'Algorithm':<25s}  {'Success':<10s}  {'Cost':<10s}  {'Detail'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*20}")
    print(f"  {'A*':<25s}  {'Y' if astar_cmp4.success else 'N':<10s}  "
          f"{astar_cmp4.cost:<10.2f}  expanded={astar_cmp4.nodes_expanded}")
    print(f"  {'APF':<25s}  {'Y' if apf_cmp4.success else 'N':<10s}  "
          f"{apf_cmp4.cost:<10.2f}  steps={apf_cmp4.steps_taken}")
    print(f"  {'Classical Q-Learning':<25s}  {'Y' if ql_cmp4.success else 'N':<10s}  "
          f"{ql_cmp4.cost:<10.2f}  ep={ql_cmp4.episodes_trained}")
    print(f"  {'Quantum Q-Learning':<25s}  {'Y' if qql_cmp4.success else 'N':<10s}  "
          f"{qql_cmp4.cost:<10.2f}  ep={qql_cmp4.episodes_trained}")

    # 4-way comparison image
    fig4, axes4 = plt.subplots(1, 4, figsize=(28, 7))
    planners4 = [
        ("A* (Global Optimal)", astar_cmp4.path, "#3399FF",
         f"cost={astar_cmp4.cost:.1f}, expanded={astar_cmp4.nodes_expanded}"),
        ("APF (Local Reactive)", apf_cmp4.path, "#FF6633",
         f"{'STUCK' if apf_cmp4.stuck else f'cost={apf_cmp4.cost:.1f}'}, "
         f"steps={apf_cmp4.steps_taken}"),
        ("Classical Q-Learning", ql_cmp4.path, "#33CC66",
         f"{'OK' if ql_cmp4.success else 'FAIL'}, "
         f"cost={ql_cmp4.cost:.1f}, ep={ql_cmp4.episodes_trained}"),
        ("Quantum Q-Learning", qql_cmp4.path, "#CC33FF",
         f"{'OK' if qql_cmp4.success else 'FAIL'}, "
         f"cost={qql_cmp4.cost:.1f}, ep={qql_cmp4.episodes_trained}"),
    ]
    for ax, (name, path, color, info) in zip(axes4, planners4):
        draw_grid(compare_grid, paths=[path] if path else None,
                  path_colors=[color], title=f"{name}\n{info}",
                  show=False, ax=ax)

    fig4.suptitle("Phase 3 — Full Planner Comparison "
                  "(A* vs APF vs Classical QL vs Quantum QL)",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cmp4_path = os.path.join(save_dir, "phase3_planner_comparison.png")
    fig4.savefig(cmp4_path, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"\n[visualize] Saved 4-way comparison to {cmp4_path}")

    # ==============================================================
    #  COMPREHENSIVE COMPARISON — All Planners vs Obstacle Density
    # ==============================================================

    print("\n" + "=" * 60)
    print("  COMPREHENSIVE COMPARISON — Varying Obstacle Density")
    print("=" * 60)

    # ---- 27. Create grids with varying obstacle densities (realistic shapes) ----
    density_configs = [
        ("10%", 0.10),
        ("15%", 0.15),
        ("20%", 0.20),
        ("25%", 0.25),
        ("30%", 0.30),
    ]

    # Build varied-obstacle grids
    varied_grids = {}
    for label, ratio in density_configs:
        varied_grids[label] = Grid.create_varied_obstacles(
            size=20, obstacle_ratio=ratio, seed=7777,
        )
        obs_count = int(np.sum(varied_grids[label].grid == 1))
        print(f"  [{label}]  Created grid with {obs_count} obstacle cells "
              f"(varied shapes)")

    # Visualize the grids themselves
    draw_varied_obstacle_grids(
        varied_grids,
        title="Environments with Varying Obstacle Density (Realistic Shapes)",
        save_path=os.path.join(save_dir, "comparison_varied_grids.png"),
    )

    # ---- 28. Run ALL 5 planners on each density ----
    print("\n[compare]  Running all planners across densities...")
    comparison_data = {}

    for label, ratio in density_configs:
        g = varied_grids[label]
        comparison_data[label] = {}

        # A*
        t0 = time.perf_counter()
        r_astar = a_star(g, eight_connected=True)
        t_astar = (time.perf_counter() - t0) * 1000
        comparison_data[label]["A*"] = {
            "cost": r_astar.cost if r_astar.success else 0,
            "success": r_astar.success,
            "time_ms": t_astar,
            "steps": len(r_astar.path) if r_astar.path else 0,
        }

        # APF
        t0 = time.perf_counter()
        r_apf = apf_plan(g)
        t_apf = (time.perf_counter() - t0) * 1000
        comparison_data[label]["APF"] = {
            "cost": r_apf.cost if not r_apf.stuck else 0,
            "success": not r_apf.stuck,
            "time_ms": t_apf,
            "steps": len(r_apf.path) if r_apf.path else 0,
        }

        # Classical Q-Learning
        t0 = time.perf_counter()
        r_ql = q_learning_full(g, episodes=500, seed=42)
        t_ql = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Classical QL"] = {
            "cost": r_ql.cost if r_ql.success else 0,
            "success": r_ql.success,
            "time_ms": t_ql,
            "steps": len(r_ql.path) if r_ql.path else 0,
        }

        # Quantum Q-Learning
        t0 = time.perf_counter()
        r_qql = quantum_q_learning_full(g, episodes=500, seed=42)
        t_qql = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Quantum QL"] = {
            "cost": r_qql.cost if r_qql.success else 0,
            "success": r_qql.success,
            "time_ms": t_qql,
            "steps": len(r_qql.path) if r_qql.path else 0,
        }

        # Hybrid Planner
        t0 = time.perf_counter()
        r_hyb = hybrid_plan(g, q_episodes=500, q_seed=42)
        t_hyb = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Hybrid"] = {
            "cost": r_hyb.cost if r_hyb.success else 0,
            "success": r_hyb.success,
            "time_ms": t_hyb,
            "steps": len(r_hyb.path) if r_hyb.path else 0,
        }

        print(f"  [{label}]  A*={'Y' if r_astar.success else 'N'}  "
              f"APF={'Y' if not r_apf.stuck else 'N'}  "
              f"CQL={'Y' if r_ql.success else 'N'}  "
              f"QQL={'Y' if r_qql.success else 'N'}  "
              f"Hybrid={'Y' if r_hyb.success else 'N'}")

    # ---- 29. Generate the comprehensive comparison bar chart ----
    draw_full_comparison(
        comparison_data,
        title="All Planners — Performance vs Obstacle Density (Varied Shapes)",
        save_path=os.path.join(save_dir, "comparison_all_planners.png"),
    )

    # ---- 30. Print summary table ----
    print(f"\n  {'Density':<10s}  {'Planner':<15s}  {'Success':<8s}  "
          f"{'Cost':<10s}  {'Steps':<8s}  {'Time(ms)':<10s}")
    print(f"  {'-'*10}  {'-'*15}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
    for label, _ in density_configs:
        for pname in comparison_data[label]:
            info = comparison_data[label][pname]
            print(f"  {label:<10s}  {pname:<15s}  "
                  f"{'Y' if info['success'] else 'N':<8s}  "
                  f"{info['cost']:<10.2f}  "
                  f"{info['steps']:<8d}  "
                  f"{info['time_ms']:<10.1f}")

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE — ALL OUTPUTS SAVED")
    print("=" * 60)
    print("  New comparison files:")
    print("    • comparison_varied_grids.png      — Grids at different densities")
    print("    • comparison_all_planners.png       — Bar chart: all 5 planners")
    print("\n  Previous outputs (Week 1 + Phase 1-3) also regenerated.")
    print("\n  Next: Phase 4 — QAOA Multi-Robot Path Optimizer")


if __name__ == "__main__":
    main()

