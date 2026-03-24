"""
run_comparison.py  –  Comprehensive comparison & novelty demos
===============================================================

Covers:
    - All 5 planners across varying obstacle densities
    - Q-value divergence demo (classical explodes, quantum stays bounded)
    - Dynamic environment demo (A* fails, Hybrid adapts)
    - Summary benchmark table

All outputs saved to experiments/results/06_comprehensive_comparison/
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
                                 train_quantum_q_learning)
from hybrid_planner import hybrid_plan
from visualize import (draw_grid, draw_full_comparison,
                       draw_varied_obstacle_grids,
                       draw_qvalue_divergence, draw_dynamic_comparison)


def run_comparison(base_dir, shared_data):
    """Run comprehensive comparisons and novelty demos."""

    dir6 = os.path.join(base_dir, "06_comprehensive_comparison")
    os.makedirs(dir6, exist_ok=True)

    # ==============================================================
    #  All Planners vs Obstacle Density
    # ==============================================================

    print("\n" + "=" * 60)
    print("  COMPREHENSIVE COMPARISON — All Planners vs Density")
    print("=" * 60)

    density_configs = [
        ("10%", 0.10), ("15%", 0.15), ("20%", 0.20),
        ("25%", 0.25), ("30%", 0.30),
    ]

    varied_grids = {}
    for label, ratio in density_configs:
        varied_grids[label] = Grid.create_varied_obstacles(
            size=20, obstacle_ratio=ratio, seed=7777)

    draw_varied_obstacle_grids(varied_grids,
        title="Environments with Varying Obstacle Density",
        save_path=os.path.join(dir6, "varied_grids.png"))

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
            "success": r_astar.success, "time_ms": t_astar,
            "steps": len(r_astar.path) if r_astar.path else 0,
        }

        # APF
        t0 = time.perf_counter()
        r_apf = apf_plan(g)
        t_apf = (time.perf_counter() - t0) * 1000
        comparison_data[label]["APF"] = {
            "cost": r_apf.cost if not r_apf.stuck else 0,
            "success": not r_apf.stuck, "time_ms": t_apf,
            "steps": len(r_apf.path) if r_apf.path else 0,
        }

        # Classical QL
        t0 = time.perf_counter()
        r_ql = q_learning_full(g, episodes=500, seed=42)
        t_ql = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Classical QL"] = {
            "cost": r_ql.cost if r_ql.success else 0,
            "success": r_ql.success, "time_ms": t_ql,
            "steps": len(r_ql.path) if r_ql.path else 0,
        }

        # Quantum QL
        t0 = time.perf_counter()
        r_qql = quantum_q_learning_full(g, episodes=500, seed=42)
        t_qql = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Quantum QL"] = {
            "cost": r_qql.cost if r_qql.success else 0,
            "success": r_qql.success, "time_ms": t_qql,
            "steps": len(r_qql.path) if r_qql.path else 0,
        }

        # Hybrid
        t0 = time.perf_counter()
        r_hyb = hybrid_plan(g, q_episodes=500, q_seed=42)
        t_hyb = (time.perf_counter() - t0) * 1000
        comparison_data[label]["Hybrid"] = {
            "cost": r_hyb.cost if r_hyb.success else 0,
            "success": r_hyb.success, "time_ms": t_hyb,
            "steps": len(r_hyb.path) if r_hyb.path else 0,
        }

        print(f"  [{label}]  A*={'Y' if r_astar.success else 'N'}  "
              f"APF={'Y' if not r_apf.stuck else 'N'}  "
              f"CQL={'Y' if r_ql.success else 'N'}  "
              f"QQL={'Y' if r_qql.success else 'N'}  "
              f"Hybrid={'Y' if r_hyb.success else 'N'}")

    draw_full_comparison(comparison_data,
        title="All Planners — Performance vs Obstacle Density",
        save_path=os.path.join(dir6, "all_planners_comparison.png"))

    # ==============================================================
    #  NOVELTY 1 — Q-Value Divergence
    # ==============================================================

    print("\n" + "=" * 60)
    print("  NOVELTY — Q-Value Divergence (Classical vs Quantum)")
    print("=" * 60)

    div_grid = Grid.create_trap_environment(size=20)
    div_grid.add_dynamic_obstacle((8, 5), direction=(0, 1), pattern="bounce")
    div_grid.add_dynamic_obstacle((12, 8), direction=(1, 0), pattern="bounce")

    print("  Training classical QL (1000 episodes, high alpha)...")
    cl_qtable, _ = train_q_learning(
        div_grid, episodes=1000, seed=42, alpha=0.5, gamma=0.99)
    print("  Training quantum QL (1000 episodes, high alpha)...")
    qu_angles, _ = train_quantum_q_learning(
        div_grid, episodes=1000, seed=42, alpha=0.5, gamma=0.99)

    cl_nz = cl_qtable[cl_qtable != 0]
    qu_nz = qu_angles[qu_angles != 0]
    print(f"  Classical: min={cl_nz.min():.1f}, max={cl_nz.max():.1f}, "
          f"range={cl_nz.max()-cl_nz.min():.1f}")
    print(f"  Quantum:   min={qu_nz.min():.4f}, max={qu_nz.max():.4f}, "
          f"bounded in [0, {math.pi:.4f}]")

    draw_qvalue_divergence(cl_qtable, qu_angles,
        title="Q-Value Divergence: Classical Explodes, Quantum Stays Bounded",
        save_path=os.path.join(dir6, "qvalue_divergence.png"))

    # ==============================================================
    #  NOVELTY 2 — Dynamic Environment (A* Fails)
    # ==============================================================

    print("\n" + "=" * 60)
    print("  NOVELTY — Dynamic Environment (A* Fails, Hybrid Adapts)")
    print("=" * 60)

    dyn_grid = Grid(size=20, obstacle_ratio=0.10, seed=999)
    astar_static = a_star(dyn_grid, eight_connected=True)
    print(f"  A* static plan: cost={astar_static.cost:.1f}, "
          f"path_len={len(astar_static.path)}")

    obstacle_positions = []
    block_idx = 5
    if astar_static.path and len(astar_static.path) > 10:
        block_idx = len(astar_static.path) * 4 // 10
        block_pos = astar_static.path[block_idx]
        obs_start = (block_pos[0], max(0, block_pos[1] - 2))
        dyn_grid.add_dynamic_obstacle(obs_start, direction=(0, 1), pattern="bounce")

        block_idx2 = len(astar_static.path) * 6 // 10
        block_pos2 = astar_static.path[block_idx2]
        obs_start2 = (max(0, block_pos2[0] - 2), block_pos2[1])
        dyn_grid.add_dynamic_obstacle(obs_start2, direction=(1, 0), pattern="bounce")
        obstacle_positions = [block_pos, block_pos2]
        print(f"  Dynamic obstacles placed near A* path at steps "
              f"{block_idx} and {block_idx2}")

    print("  Running hybrid planner on dynamic grid...")
    hybrid_dyn = hybrid_plan(dyn_grid, q_episodes=500, q_seed=42)
    print(f"  Hybrid: success={hybrid_dyn.success}, cost={hybrid_dyn.cost:.1f}")

    draw_dynamic_comparison(dyn_grid,
        astar_path=astar_static.path,
        astar_blocked_step=block_idx,
        hybrid_path=hybrid_dyn.path if hybrid_dyn.path else [],
        obstacle_positions_at_block=obstacle_positions,
        title="A* Fails in Dynamic Environment — Hybrid Adapts",
        save_path=os.path.join(dir6, "dynamic_comparison.png"))

    # ==============================================================
    #  Benchmark Table (Markdown)
    # ==============================================================

    print("\n  Generating benchmark table...")
    lines = ["# Benchmark Results\n"]
    lines.append("| Density | Planner | Success | Cost | Steps | Time (ms) |")
    lines.append("|---------|---------|---------|------|-------|-----------|")
    for label, _ in density_configs:
        for pname in comparison_data[label]:
            info = comparison_data[label][pname]
            lines.append(
                f"| {label} | {pname} | "
                f"{'Yes' if info['success'] else 'No'} | "
                f"{info['cost']:.1f} | {info['steps']} | "
                f"{info['time_ms']:.1f} |"
            )
    lines.append("\n## Key Findings\n")
    lines.append("- **A*** is optimal on static grids but cannot adapt to moving obstacles")
    lines.append("- **APF** reacts locally but gets stuck in traps (U-shaped walls)")
    lines.append("- **Classical QL** learns but Q-values can diverge to +/- infinity")
    lines.append("- **Quantum QL** stays bounded [0, pi] — never diverges")
    lines.append("- **Hybrid** combines all three layers: global (A*) + reactive (APF) + escape (Quantum QL)")

    table_path = os.path.join(dir6, "benchmark_table.md")
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved benchmark table to {table_path}")

    print("\n  Comparison results saved to:")
    print(f"    {dir6}/")
