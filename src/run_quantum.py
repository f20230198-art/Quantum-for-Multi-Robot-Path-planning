"""
run_quantum.py  –  Generate all quantum algorithm results
==========================================================

Covers:
    Phase 3: Quantum Q-Learning, Hybrid Planner, 4-way comparison
    Phase 4: QAOA multi-robot path optimizer, scalability benchmark

All outputs saved to experiments/results/04_quantum_advantage/
                    experiments/results/05_qaoa_coordination/
"""

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid import Grid
from a_star import a_star, euclidean
from apf import apf_plan
from q_learning import q_learning_full, train_q_learning
from quantum_q_learning import (quantum_q_learning_full,
                                 train_quantum_q_learning,
                                 verify_qiskit_match)
from hybrid_planner import hybrid_plan
from multi_robot import (prepare_qaoa_input, brute_force_select,
                         evaluate_selection)
from dynamic_env import Simulation
from visualize import (draw_grid, draw_quantum_comparison,
                       draw_learning_curves, draw_multi_robot_grid,
                       draw_simulation_gif,
                       draw_qaoa_selection_comparison, draw_scalability_chart,
                       draw_qaoa_explanation)
from qaoa_optimizer import qaoa_select, qaoa_select_full, scalability_benchmark


def run_quantum(base_dir, shared_data):
    """Run all quantum algorithm demos and save results."""

    dir4 = os.path.join(base_dir, "04_quantum_advantage")
    dir5 = os.path.join(base_dir, "05_qaoa_coordination")
    for d in [dir4, dir5]:
        os.makedirs(d, exist_ok=True)

    trap_env = shared_data["trap_env"]
    ql_result = shared_data["ql_result"]
    mr_grid = shared_data["mr_grid"]
    all_candidates = shared_data["all_candidates"]
    sel_greedy = shared_data["sel_greedy"]
    sel_random = shared_data["sel_random"]
    sel_brute = shared_data["sel_brute"]
    compare_grid = shared_data["compare_grid"]

    # ==============================================================
    #  PHASE 3 — Quantum Q-Learning + Hybrid Planner
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 3 — Quantum Q-Learning + Hybrid Planner")
    print("=" * 60)

    # Qiskit verification
    print("\n[Qiskit]  Verifying quantum backend...")
    for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
        v = verify_qiskit_match(theta, shots=1000)
        backend = "qiskit" if v["qiskit_available"] else "numpy"
        print(f"  theta={theta:.4f}  prob={v['numpy_prob']:.4f}  ({backend})")

    # Quantum Q-Learning on trap
    print("\n[QQL]  Training Quantum Q-Learning on U-trap (500 episodes)...")
    t0 = time.perf_counter()
    qql_result = quantum_q_learning_full(trap_env, episodes=500, seed=42)
    qql_ms = (time.perf_counter() - t0) * 1000
    print(f"[QQL]  {qql_result}  ({qql_ms:.0f} ms)")

    draw_grid(trap_env, paths=[qql_result.path] if qql_result.path else None,
              path_colors=["#CC33FF"],
              title=f"Quantum Q-Learning on U-Trap — "
                    f"{'SUCCESS' if qql_result.success else 'FAIL'}",
              save_path=os.path.join(dir4, "quantum_ql_trap.png"), show=False)

    # Classical vs Quantum comparison
    draw_quantum_comparison(trap_env, results={
        "Classical Q-Learning": {
            "path": ql_result.path, "color": "#33CC66",
            "info": f"cost={ql_result.cost:.1f}",
        },
        "Quantum Q-Learning": {
            "path": qql_result.path, "color": "#CC33FF",
            "info": f"cost={qql_result.cost:.1f}",
        },
    }, title="Classical vs Quantum Q-Learning (U-Trap)",
       save_path=os.path.join(dir4, "ql_vs_quantum_ql.png"))

    # Learning curves
    _, cl_rewards = train_q_learning(trap_env, episodes=500, seed=42)
    _, qq_rewards = train_quantum_q_learning(trap_env, episodes=500, seed=42)
    draw_learning_curves(curves={
        "Classical Q-Learning": cl_rewards,
        "Quantum Q-Learning": qq_rewards,
    }, title="Training Reward Curves (U-Trap, 500 episodes)",
       save_path=os.path.join(dir4, "learning_curves.png"))

    # Hybrid planner
    print("\n[Hybrid]  Running hybrid planner on U-trap...")
    t0 = time.perf_counter()
    hybrid_result = hybrid_plan(trap_env, q_episodes=500, q_seed=42)
    hybrid_ms = (time.perf_counter() - t0) * 1000
    print(f"[Hybrid]  {hybrid_result}  ({hybrid_ms:.0f} ms)")

    draw_grid(trap_env, paths=[hybrid_result.path] if hybrid_result.path else None,
              path_colors=["#FF9933"],
              title=f"Hybrid Planner (A* + APF + Quantum QL)\n"
                    f"A*={hybrid_result.astar_replans}, APF={hybrid_result.apf_steps}, "
                    f"QEscape={hybrid_result.quantum_escapes}",
              save_path=os.path.join(dir4, "hybrid_planner.png"), show=False)

    # 4-way comparison
    print("\n[compare]  4-way planner comparison...")
    astar_cmp4 = a_star(compare_grid)
    apf_cmp4 = apf_plan(compare_grid)
    ql_cmp4 = q_learning_full(compare_grid, episodes=500, seed=42)
    qql_cmp4 = quantum_q_learning_full(compare_grid, episodes=500, seed=42)

    fig4, axes4 = plt.subplots(1, 4, figsize=(28, 7))
    planners4 = [
        ("A* (Global Optimal)", astar_cmp4.path, "#3399FF",
         f"cost={astar_cmp4.cost:.1f}"),
        ("APF (Local Reactive)", apf_cmp4.path, "#FF6633",
         f"{'STUCK' if apf_cmp4.stuck else f'cost={apf_cmp4.cost:.1f}'}"),
        ("Classical Q-Learning", ql_cmp4.path, "#33CC66",
         f"{'OK' if ql_cmp4.success else 'FAIL'}, cost={ql_cmp4.cost:.1f}"),
        ("Quantum Q-Learning", qql_cmp4.path, "#CC33FF",
         f"{'OK' if qql_cmp4.success else 'FAIL'}, cost={qql_cmp4.cost:.1f}"),
    ]
    for ax, (name, path, color, info) in zip(axes4, planners4):
        draw_grid(compare_grid, paths=[path] if path else None,
                  path_colors=[color], title=f"{name}\n{info}", show=False, ax=ax)
    fig4.suptitle("All Planners Compared — Quantum QL Escapes Where Others Fail",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4.savefig(os.path.join(dir4, "4way_comparison.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ==============================================================
    #  PHASE 4 — QAOA Multi-Robot Path Optimizer
    # ==============================================================

    print("\n" + "=" * 60)
    print("  PHASE 4 — QAOA Multi-Robot Path Optimizer")
    print("=" * 60)

    print("\n[QAOA]  Running QAOA on multi-robot environment...")
    qaoa_result = qaoa_select_full(
        all_candidates, conflict_penalty=10.0, one_hot_penalty=50.0,
        p=2, num_restarts=3, seed=42)
    print(f"[QAOA]  {qaoa_result}")

    # Compare against classical
    sel_qaoa = qaoa_result.selection
    selections_p4 = {
        "Greedy (shortest)": sel_greedy,
        "Random": sel_random,
        "Brute-Force (optimal)": sel_brute,
        "QAOA (quantum)": sel_qaoa,
    }
    eval_results_p4 = {}
    for name, sel in selections_p4.items():
        ev = evaluate_selection(all_candidates, sel, conflict_penalty=10.0)
        eval_results_p4[name] = ev
        print(f"  {name:25s}  score={ev['score']:.1f}")

    draw_qaoa_selection_comparison(mr_grid, all_candidates,
        selections_p4, eval_results_p4,
        title="Path Selection: Classical vs QAOA",
        save_path=os.path.join(dir5, "qaoa_vs_classical.png"))

    # QAOA explanation figure
    print("\n[QAOA]  Generating explanation figure...")
    expl_grid = Grid.create_multi_robot_env(
        size=20, obstacle_ratio=0.10, num_robots=3, num_dynamic=2, seed=1)
    expl_input = prepare_qaoa_input(expl_grid, num_candidates=4, seed=1)
    expl_cands = expl_input["all_candidates"]
    expl_bf_sel = brute_force_select(expl_cands, conflict_penalty=10.0)
    expl_qaoa = qaoa_select_full(expl_cands, p=1, num_restarts=2, seed=1)
    expl_bf_eval = evaluate_selection(expl_cands, expl_bf_sel, 10.0)
    expl_qaoa_eval = evaluate_selection(expl_cands, expl_qaoa.selection, 10.0)

    draw_qaoa_explanation(expl_grid, expl_cands,
        expl_input["conflict_matrix"], expl_input["index_map"],
        qaoa_selection=expl_qaoa.selection, bf_selection=expl_bf_sel,
        qaoa_eval=expl_qaoa_eval, bf_eval=expl_bf_eval,
        title="How QAOA Works (Step by Step)",
        save_path=os.path.join(dir5, "qaoa_explanation.png"))

    # Scalability benchmark
    print("\n[scale]  Running scalability benchmark...")
    bench_data = scalability_benchmark(
        robot_counts=[2, 3, 4], candidates_per_robot=3,
        conflict_penalty=10.0, seed=42)

    draw_scalability_chart(bench_data,
        title="QAOA vs Brute-Force Scalability",
        save_path=os.path.join(dir5, "scalability.png"))

    # QAOA simulation
    print("\n[sim]  Running simulation with QAOA-selected paths...")
    qaoa_paths = {}
    for cfg in mr_grid.robot_configs:
        rid = cfg["id"]
        qaoa_paths[rid] = all_candidates[rid][sel_qaoa[rid]]["path"]

    sim_qaoa = Simulation(mr_grid, qaoa_paths, max_ticks=200)
    sim_qaoa_result = sim_qaoa.run()
    print(f"  Simulation: {sim_qaoa_result.total_ticks} ticks, "
          f"collisions={sim_qaoa_result.total_collisions}")

    draw_simulation_gif(mr_grid, sim_qaoa_result,
        save_path=os.path.join(dir5, "qaoa_simulation.gif"),
        fps=4, title="Multi-Robot Simulation (QAOA Selection)")

    print("\n  Quantum results saved to:")
    print(f"    {dir4}/")
    print(f"    {dir5}/")
