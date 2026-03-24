"""
main.py  –  One-Click Demo (Generates All Results)
=====================================================

WHAT THIS FILE DOES:
    Run with:   python main.py

    It generates ALL project outputs in one go, organized into folders:

        experiments/results/
            01_classical_baseline/    — A*, Greedy BFS, traps
            02_multi_robot_challenge/ — Multi-robot env, candidate paths
            03_classical_limitations/ — APF stuck, Q-Learning comparison
            04_quantum_advantage/     — Quantum QL, Hybrid planner
            05_qaoa_coordination/     — QAOA optimizer, scalability
            06_comprehensive_comparison/ — All planners head-to-head,
                                          Q-value divergence, dynamic env demo

    Total: 26+ images, GIFs, and a benchmark table.

FOR INTERACTIVE DEMOS (pygame), use:
    python simulator_classical.py   — A*, APF, Q-Learning (live)
    python simulator_quantum.py     — Quantum QL, Hybrid, QAOA (live)
"""

import sys
import os

# Use non-interactive backend so plt.show() won't block
import matplotlib
matplotlib.use("Agg")

# Make sure Python can find our src/ modules when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from run_classical import run_classical
from run_quantum import run_quantum
from run_comparison import run_comparison


def main():
    print("=" * 60)
    print("  QUANTUM MULTI-ROBOT PATH PLANNING")
    print("  One-Click Demo — Generating All Results")
    print("=" * 60)

    # Output folder
    base_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    os.makedirs(base_dir, exist_ok=True)

    # Phase 1: Classical algorithms (A*, APF, Q-Learning)
    shared_data = run_classical(base_dir)
    if shared_data is None:
        print("\nERROR: A* failed on random grid. Re-run for a different seed.")
        return

    # Phase 2: Quantum algorithms (Quantum QL, Hybrid, QAOA)
    run_quantum(base_dir, shared_data)

    # Phase 3: Comprehensive comparison & novelty demos
    run_comparison(base_dir, shared_data)

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL RESULTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print("\n  Output folders:")
    print("    01_classical_baseline/       — A* search, traps, Greedy BFS")
    print("    02_multi_robot_challenge/    — Multi-robot env & selection")
    print("    03_classical_limitations/    — APF stuck, QL comparison")
    print("    04_quantum_advantage/        — Quantum QL, Hybrid, 4-way comparison")
    print("    05_qaoa_coordination/        — QAOA optimizer & scalability")
    print("    06_comprehensive_comparison/ — Head-to-head, divergence, dynamic env")
    print(f"\n  All saved to: {os.path.abspath(base_dir)}/")
    print("\n  For interactive demos:")
    print("    python simulator_classical.py   (A*, APF, Q-Learning)")
    print("    python simulator_quantum.py     (Quantum QL, Hybrid, QAOA)")


if __name__ == "__main__":
    main()
