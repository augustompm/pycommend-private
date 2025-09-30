"""
Test MOVNS v15 vs MOEA/D - Fixed Archive Management
Focus: Fair comparison with 80-100 solutions each
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v15 import MOVNS_V15
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def run_final_comparison(package='fastapi', iterations=30):
    """Final comparison with fixed archive management"""

    print("="*70)
    print("MOVNS v15 vs MOEA/D - FINAL COMPARISON")
    print("="*70)
    print("\nGoal: MOVNS with 80-100 solutions to compete fairly")

    print("\n" + "-"*70)
    print("1. MOVNS v15 (Fixed Archive)")
    print("-"*70)

    movns = MOVNS_V15(
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

    # Calculate IGD+ and Spacing
    movns_objectives = []
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
    movns_objectives = np.array(movns_objectives)

    # Generate reference set
    all_objectives = movns_objectives.copy()

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

    # Combine for reference set
    all_objectives = np.vstack([all_objectives, moead_objectives])

    moead_spacing = qm.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

    # Calculate IGD+ with combined reference set
    reference = qm.get_non_dominated_set(all_objectives)
    movns_igd = qm.igd_plus(movns_objectives, reference) if len(reference) > 0 else float('inf')
    moead_igd = qm.igd_plus(moead_objectives, reference) if len(reference) > 0 else float('inf')

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

    print("\n1. ARCHIVE SIZE (80-100 expected)")
    print(f"   MOVNS v15:  {len(movns_solutions)} solutions")
    print(f"   MOEA/D:     {len(moead_solutions)} solutions")
    archive_ok = 80 <= len(movns_solutions) <= 100
    print(f"   Status: {'✓ FIXED' if archive_ok else '✗ FAILED'}")

    print("\n2. HYPERVOLUME (higher is better)")
    print(f"   MOVNS v15:  {movns_hv:.4f}")
    print(f"   MOEA/D:     {moead_hv:.4f}")

    hv_winner = None
    if movns_hv > 0 and moead_hv > 0:
        if movns_hv > moead_hv:
            ratio = movns_hv / moead_hv
            print(f"   Winner: MOVNS v15 ({ratio:.2f}x better)")
            hv_winner = 'MOVNS'
        else:
            ratio = moead_hv / movns_hv
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            hv_winner = 'MOEAD'

    print("\n3. IGD+ (lower is better)")
    print(f"   MOVNS v15:  {movns_igd:.4f}")
    print(f"   MOEA/D:     {moead_igd:.4f}")

    igd_winner = None
    if movns_igd < float('inf') and moead_igd < float('inf'):
        if movns_igd < moead_igd:
            ratio = moead_igd / movns_igd
            print(f"   Winner: MOVNS v15 ({ratio:.2f}x better)")
            igd_winner = 'MOVNS'
        else:
            ratio = movns_igd / moead_igd
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            igd_winner = 'MOEAD'

    print("\n4. SPACING (lower is better)")
    print(f"   MOVNS v15:  {movns_spacing:.4f}")
    print(f"   MOEA/D:     {moead_spacing:.4f}")

    spacing_winner = None
    if movns_spacing < float('inf') and moead_spacing < float('inf'):
        if movns_spacing < moead_spacing:
            ratio = moead_spacing / movns_spacing
            print(f"   Winner: MOVNS v15 ({ratio:.2f}x better)")
            spacing_winner = 'MOVNS'
        else:
            ratio = movns_spacing / moead_spacing
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            spacing_winner = 'MOEAD'

    print("\n5. SOLUTION QUALITY")
    print(f"   Best LU:  MOVNS={best_movns['lu']:.0f}, MOEA/D={best_moead['lu']:.0f}")
    print(f"   Best SS:  MOVNS={best_movns['ss']:.4f}, MOEA/D={best_moead['ss']:.4f}")
    print(f"   Best RSS: MOVNS={best_movns['rss']:.1f}, MOEA/D={best_moead['rss']:.1f}")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    movns_points = 0
    moead_points = 0

    if hv_winner == 'MOVNS':
        movns_points += 1
        print("+ HV: MOVNS v15")
    elif hv_winner == 'MOEAD':
        moead_points += 1
        print("+ HV: MOEA/D")

    if igd_winner == 'MOVNS':
        movns_points += 1
        print("+ IGD+: MOVNS v15")
    elif igd_winner == 'MOEAD':
        moead_points += 1
        print("+ IGD+: MOEA/D")

    if spacing_winner == 'MOVNS':
        movns_points += 1
        print("+ Spacing: MOVNS v15")
    elif spacing_winner == 'MOEAD':
        moead_points += 1
        print("+ Spacing: MOEA/D")

    print(f"\nFinal Score: MOVNS v15 {movns_points} - {moead_points} MOEA/D")

    if movns_points >= 2:
        print("\n**WINNER: MOVNS v15**")
        print("VNS approach succeeds with proper archive management")
    elif moead_points >= 2:
        print("\n**WINNER: MOEA/D**")
        print("Decomposition approach maintains advantage")
    else:
        print("\n**COMPETITIVE**")
        print("Both algorithms show comparable performance")

    if archive_ok:
        print("\nArchive size FIXED: Fair comparison achieved")
    else:
        print("\nArchive size ISSUE: Further tuning needed")

    return {
        'movns': {'hv': movns_hv, 'igd': movns_igd, 'spacing': movns_spacing,
                  'time': movns_time, 'solutions': len(movns_solutions)},
        'moead': {'hv': moead_hv, 'igd': moead_igd, 'spacing': moead_spacing,
                  'time': moead_time, 'solutions': len(moead_solutions)}
    }


if __name__ == "__main__":
    results = run_final_comparison('fastapi', iterations=30)