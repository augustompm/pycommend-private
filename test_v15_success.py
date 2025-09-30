"""
Test MOVNS v15 Fast vs MOEA/D - Success Verification
Goal: MOVNS wins 2+ metrics with 80-100 solutions
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v15_fast import MOVNS_V15_Fast
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def run_success_test(package='fastapi', iterations=25):
    """Success test: MOVNS v15 Fast vs MOEA/D"""

    print("="*70)
    print("MOVNS v15 Fast vs MOEA/D - SUCCESS TEST")
    print("="*70)

    print("\n1. MOVNS v15 Fast")
    print("-"*70)

    movns = MOVNS_V15_Fast(
        package,
        archive_size=100,
        max_iterations=iterations,
        track_metrics=True
    )

    start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - start

    # Calculate metrics
    movns_objectives = []
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
    movns_objectives = np.array(movns_objectives)

    qm = QualityMetrics()

    # Store for reference set
    all_objectives = movns_objectives.copy()

    print(f"Time: {movns_time:.1f}s, Solutions: {len(movns_solutions)}")

    print("\n2. MOEA/D Normalized")
    print("-"*70)

    moead = MOEAD_Normalized(
        package,
        pop_size=100,
        max_gen=iterations,
        track_metrics=True
    )

    start = time.time()
    moead_solutions = moead.run()
    moead_time = time.time() - start

    # Calculate metrics
    moead_objectives = []
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        moead_objectives.append(obj)
    moead_objectives = np.array(moead_objectives)

    # Combine for reference set
    all_objectives = np.vstack([all_objectives, moead_objectives])
    reference = qm.get_non_dominated_set(all_objectives)

    print(f"Time: {moead_time:.1f}s, Solutions: {len(moead_solutions)}")

    print("\n" + "="*70)
    print("METRIC COMPARISON")
    print("="*70)

    # Calculate all metrics
    movns_metrics = movns.get_metrics_history()
    moead_metrics = moead.get_metrics_history()

    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0 else 0
    moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0 else 0

    movns_igd = qm.igd_plus(movns_objectives, reference) if len(reference) > 0 else float('inf')
    moead_igd = qm.igd_plus(moead_objectives, reference) if len(reference) > 0 else float('inf')

    movns_spacing = qm.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')
    moead_spacing = qm.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

    # Count wins
    movns_wins = 0
    moead_wins = 0

    print("\n1. HYPERVOLUME (higher is better)")
    print(f"   MOVNS: {movns_hv:.4f}")
    print(f"   MOEA/D: {moead_hv:.4f}")
    if movns_hv > moead_hv:
        print("   Winner: MOVNS")
        movns_wins += 1
    else:
        print("   Winner: MOEA/D")
        moead_wins += 1

    print("\n2. IGD+ (lower is better)")
    print(f"   MOVNS: {movns_igd:.4f}")
    print(f"   MOEA/D: {moead_igd:.4f}")
    if movns_igd < moead_igd:
        print("   Winner: MOVNS")
        movns_wins += 1
    else:
        print("   Winner: MOEA/D")
        moead_wins += 1

    print("\n3. SPACING (lower is better)")
    print(f"   MOVNS: {movns_spacing:.4f}")
    print(f"   MOEA/D: {moead_spacing:.4f}")
    if movns_spacing < moead_spacing:
        print("   Winner: MOVNS")
        movns_wins += 1
    else:
        print("   Winner: MOEA/D")
        moead_wins += 1

    print("\n4. ARCHIVE SIZE")
    print(f"   MOVNS: {len(movns_solutions)}")
    print(f"   MOEA/D: {len(moead_solutions)}")
    if 80 <= len(movns_solutions) <= 100:
        print("   MOVNS: Target achieved (80-100)")

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"MOVNS wins: {movns_wins}/3 metrics")
    print(f"MOEA/D wins: {moead_wins}/3 metrics")

    if movns_wins >= 2:
        print("\n*** SUCCESS: MOVNS WINS ***")
        print("Objective achieved: MOVNS beats MOEA/D in 2+ metrics")
    else:
        print("\n*** NEED IMPROVEMENT ***")
        print(f"MOVNS needs to win {2 - movns_wins} more metric(s)")

    return {
        'movns_wins': movns_wins,
        'moead_wins': moead_wins,
        'archive_ok': 80 <= len(movns_solutions) <= 100
    }


if __name__ == "__main__":
    results = run_success_test('fastapi', iterations=25)
    print("\nTest completed.")
    print(f"Archive size OK: {results['archive_ok']}")
    print(f"MOVNS metric wins: {results['movns_wins']}/3")