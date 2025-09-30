"""
Test MOVNS v14 vs MOEA/D - Final Balanced Version
Focus: HV (primary) and Spacing (secondary)
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v14 import MOVNS_V14
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def run_final_comparison(package='fastapi', iterations=30):
    """Final comparison with balanced v14"""

    print("="*70)
    print("MOVNS v14 vs MOEA/D - FINAL BALANCED COMPARISON")
    print("="*70)
    print("\nStrategy: 80% HV focus, 20% spacing improvement")

    print("\n" + "-"*70)
    print("1. MOVNS v14 (Balanced)")
    print("-"*70)

    movns = MOVNS_V14(
        package,
        archive_size=100,
        max_iterations=iterations,
        track_metrics=True
    )

    start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - start

    # Get metrics
    movns_metrics = movns.get_metrics_history()
    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0 else 0

    # Calculate spacing
    movns_objectives = []
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
    movns_objectives = np.array(movns_objectives)

    qm = QualityMetrics()
    movns_spacing = qm.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')

    # Get best objectives
    best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for obj in movns_objectives:
        if -obj[0] > best_movns['lu']:
            best_movns['lu'] = -obj[0]
        if -obj[1] > best_movns['ss']:
            best_movns['ss'] = -obj[1]
        if obj[2] < best_movns['rss']:
            best_movns['rss'] = obj[2]

    print(f"\nResults:")
    print(f"  Time: {movns_time:.1f}s")
    print(f"  Archive: {len(movns_solutions)} solutions")
    print(f"  HV: {movns_hv:.4f}")
    print(f"  Spacing: {movns_spacing:.4f}")
    print(f"  Best LU: {best_movns['lu']:.0f}")
    print(f"  Best SS: {best_movns['ss']:.4f}")
    print(f"  Best RSS: {best_movns['rss']:.1f}")

    print("\n" + "-"*70)
    print("2. MOEA/D NORMALIZED")
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

    # Get metrics
    moead_metrics = moead.get_metrics_history()
    moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0 else 0

    # Calculate spacing
    moead_objectives = []
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        moead_objectives.append(obj)
    moead_objectives = np.array(moead_objectives)

    moead_spacing = qm.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

    # Get best objectives
    best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for obj in moead_objectives:
        if -obj[0] > best_moead['lu']:
            best_moead['lu'] = -obj[0]
        if -obj[1] > best_moead['ss']:
            best_moead['ss'] = -obj[1]
        if obj[2] < best_moead['rss']:
            best_moead['rss'] = obj[2]

    print(f"\nResults:")
    print(f"  Time: {moead_time:.1f}s")
    print(f"  Archive: {len(moead_solutions)} solutions")
    print(f"  HV: {moead_hv:.4f}")
    print(f"  Spacing: {moead_spacing:.4f}")
    print(f"  Best LU: {best_moead['lu']:.0f}")
    print(f"  Best SS: {best_moead['ss']:.4f}")
    print(f"  Best RSS: {best_moead['rss']:.1f}")

    print("\n" + "="*70)
    print("FINAL COMPARISON RESULTS")
    print("="*70)

    print("\n1. HYPERVOLUME (higher is better) - PRIMARY METRIC")
    print(f"   MOVNS v14:  {movns_hv:.4f}")
    print(f"   MOEA/D:     {moead_hv:.4f}")

    hv_winner = None
    if movns_hv > 0 and moead_hv > 0:
        if movns_hv > moead_hv:
            ratio = movns_hv / moead_hv
            print(f"   Winner: MOVNS v14 ({ratio:.2f}x better)")
            hv_winner = 'MOVNS'
        else:
            ratio = moead_hv / movns_hv
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            hv_winner = 'MOEAD'
    elif movns_hv > 0:
        print(f"   Winner: MOVNS v14")
        hv_winner = 'MOVNS'
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D")
        hv_winner = 'MOEAD'

    print("\n2. SPACING (lower is better) - SECONDARY METRIC")
    print(f"   MOVNS v14:  {movns_spacing:.4f}")
    print(f"   MOEA/D:     {moead_spacing:.4f}")

    spacing_winner = None
    if movns_spacing < float('inf') and moead_spacing < float('inf'):
        if movns_spacing < moead_spacing:
            ratio = moead_spacing / movns_spacing
            print(f"   Winner: MOVNS v14 ({ratio:.2f}x better)")
            spacing_winner = 'MOVNS'
        else:
            ratio = movns_spacing / moead_spacing
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            spacing_winner = 'MOEAD'

    print("\n3. SOLUTION QUALITY")
    print(f"   Best LU:  MOVNS={best_movns['lu']:.0f}, MOEA/D={best_moead['lu']:.0f}")
    print(f"   Best SS:  MOVNS={best_movns['ss']:.4f}, MOEA/D={best_moead['ss']:.4f}")
    print(f"   Best RSS: MOVNS={best_movns['rss']:.1f}, MOEA/D={best_moead['rss']:.1f}")

    print("\n4. EFFICIENCY")
    print(f"   Time:     MOVNS={movns_time:.1f}s, MOEA/D={moead_time:.1f}s")
    print(f"   Archive:  MOVNS={len(movns_solutions)}, MOEA/D={len(moead_solutions)}")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    movns_points = 0
    moead_points = 0

    if hv_winner == 'MOVNS':
        movns_points += 2  # HV is worth 2 points
        print("+ HV: MOVNS v14 (2 points)")
    elif hv_winner == 'MOEAD':
        moead_points += 2
        print("+ HV: MOEA/D (2 points)")

    if spacing_winner == 'MOVNS':
        movns_points += 1
        print("+ Spacing: MOVNS v14 (1 point)")
    elif spacing_winner == 'MOEAD':
        moead_points += 1
        print("+ Spacing: MOEA/D (1 point)")

    print(f"\nFinal Score: MOVNS v14 {movns_points} - {moead_points} MOEA/D")

    if movns_points > moead_points:
        print("\n**WINNER: MOVNS v14**")
        print("VNS approach with balanced strategy succeeds")
    elif moead_points > movns_points:
        print("\n**WINNER: MOEA/D**")
        print("Decomposition approach maintains advantage")
    else:
        print("\n**TIE**")
        print("Both algorithms show comparable performance")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\nV14 Strategy:")
    print("- 80% focus on quality (HV)")
    print("- 20% focus on diversity (Spacing)")
    print("- Aggressive local search for high-quality solutions")
    print("- Mild diversity enhancement without quality sacrifice")
    print("- Large archive maintained (target: 100 solutions)")

    if hv_winner == 'MOVNS':
        print("\nSuccess: VNS strength in intensification confirmed")
        print("MOVNS achieves superior Pareto front quality (HV)")
    else:
        print("\nChallenge: Need to further enhance local search")
        print("Consider more aggressive intensification strategies")

    return {
        'movns': {'hv': movns_hv, 'spacing': movns_spacing, 'time': movns_time},
        'moead': {'hv': moead_hv, 'spacing': moead_spacing, 'time': moead_time}
    }


if __name__ == "__main__":
    results = run_final_comparison('fastapi', iterations=30)