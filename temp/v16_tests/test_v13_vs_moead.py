"""
Test MOVNS v13 vs MOEA/D
Focus: HV and Spacing metrics
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v13 import MOVNS_V13
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def calculate_spacing(solutions, algo):
    """Calculate spacing metric for distribution uniformity"""
    if not solutions or len(solutions) < 2:
        return float('inf')

    objectives = []
    for sol in solutions:
        if isinstance(sol, dict) and 'chromosome' in sol:
            obj = algo.evaluate_objectives(sol['chromosome'])
        else:
            obj = algo.evaluate_objectives(sol)
        objectives.append(obj)

    objectives = np.array(objectives)

    qm = QualityMetrics()
    spacing_value = qm.spacing(objectives)

    return spacing_value


def run_comparison(package='fastapi', iterations=25):
    """Run comparison focused on HV and Spacing"""

    print("="*70)
    print("MOVNS v13 vs MOEA/D - HV AND SPACING COMPARISON")
    print("="*70)

    print("\n" + "-"*70)
    print("1. MOVNS v13 (Distribution Optimized)")
    print("-"*70)

    movns = MOVNS_V13(
        package,
        archive_size=100,
        max_iterations=iterations,
        track_metrics=True
    )

    start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - start

    movns_metrics = movns.get_metrics_history()
    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0 else 0

    movns_spacing = calculate_spacing(movns_solutions, movns)

    movns_objectives = []
    best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
        if -obj[0] > best_movns['lu']:
            best_movns['lu'] = -obj[0]
        if -obj[1] > best_movns['ss']:
            best_movns['ss'] = -obj[1]
        if obj[2] < best_movns['rss']:
            best_movns['rss'] = obj[2]

    print(f"Completed in {movns_time:.1f}s")
    print(f"Archive: {len(movns_solutions)} solutions")
    print(f"HV: {movns_hv:.4f}")
    print(f"Spacing: {movns_spacing:.4f}")
    print(f"Best LU: {best_movns['lu']:.0f}, SS: {best_movns['ss']:.4f}, RSS: {best_movns['rss']}")

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

    moead_metrics = moead.get_metrics_history()
    moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0 else 0

    moead_spacing = calculate_spacing(moead_solutions, moead)

    moead_objectives = []
    best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        moead_objectives.append(obj)
        if -obj[0] > best_moead['lu']:
            best_moead['lu'] = -obj[0]
        if -obj[1] > best_moead['ss']:
            best_moead['ss'] = -obj[1]
        if obj[2] < best_moead['rss']:
            best_moead['rss'] = obj[2]

    print(f"Completed in {moead_time:.1f}s")
    print(f"Archive: {len(moead_solutions)} solutions")
    print(f"HV: {moead_hv:.4f}")
    print(f"Spacing: {moead_spacing:.4f}")
    print(f"Best LU: {best_moead['lu']:.0f}, SS: {best_moead['ss']:.4f}, RSS: {best_moead['rss']}")

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n1. HYPERVOLUME (higher is better)")
    print(f"   MOVNS v13:  {movns_hv:.4f}")
    print(f"   MOEA/D:     {moead_hv:.4f}")
    hv_winner = None
    if movns_hv > 0 and moead_hv > 0:
        if movns_hv > moead_hv:
            ratio = movns_hv / moead_hv
            print(f"   Winner: MOVNS v13 ({ratio:.2f}x better)")
            hv_winner = 'MOVNS'
        else:
            ratio = moead_hv / movns_hv
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            hv_winner = 'MOEAD'
    elif movns_hv > 0:
        print(f"   Winner: MOVNS v13")
        hv_winner = 'MOVNS'
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D")
        hv_winner = 'MOEAD'

    print("\n2. SPACING (lower is better - distribution uniformity)")
    print(f"   MOVNS v13:  {movns_spacing:.4f}")
    print(f"   MOEA/D:     {moead_spacing:.4f}")
    spacing_winner = None
    if movns_spacing < float('inf') and moead_spacing < float('inf'):
        if movns_spacing < moead_spacing:
            ratio = moead_spacing / movns_spacing if movns_spacing > 0 else float('inf')
            print(f"   Winner: MOVNS v13 ({ratio:.2f}x better)")
            spacing_winner = 'MOVNS'
        else:
            ratio = movns_spacing / moead_spacing if moead_spacing > 0 else float('inf')
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            spacing_winner = 'MOEAD'

    print("\n3. ARCHIVE SIZE")
    print(f"   MOVNS v13:  {len(movns_solutions)}")
    print(f"   MOEA/D:     {len(moead_solutions)}")

    print("\n4. EXECUTION TIME")
    print(f"   MOVNS v13:  {movns_time:.1f}s")
    print(f"   MOEA/D:     {moead_time:.1f}s")
    print(f"   Speed ratio: {moead_time/movns_time:.1f}x faster")

    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    movns_score = 0
    moead_score = 0

    if hv_winner == 'MOVNS':
        movns_score += 1
        print("+ HV: MOVNS v13")
    elif hv_winner == 'MOEAD':
        moead_score += 1
        print("+ HV: MOEA/D")

    if spacing_winner == 'MOVNS':
        movns_score += 1
        print("+ Spacing: MOVNS v13")
    elif spacing_winner == 'MOEAD':
        moead_score += 1
        print("+ Spacing: MOEA/D")

    print(f"\nScore: MOVNS v13 {movns_score} - {moead_score} MOEA/D")

    if movns_score == 2:
        print("\n**WINNER: MOVNS v13 (2-0)**")
        print("Successfully optimized for both HV and Spacing!")
    elif moead_score == 2:
        print("\n**WINNER: MOEA/D (0-2)**")
    else:
        print("\n**SPLIT DECISION (1-1)**")
        if hv_winner == 'MOVNS':
            print("MOVNS v13 maintains HV advantage")
        else:
            print("MOEA/D maintains HV advantage")

    print("\n" + "="*70)
    print("V13 IMPROVEMENTS")
    print("="*70)
    print("- n1_spread_flip: Removes high co-occurrence packages")
    print("- n2_diversity_exchange: Balances cluster representation")
    print("- n3_boundary_push: Pushes solutions to objective boundaries")
    print("- n4_gap_filling: Fills gaps in objective space")
    print("- Crowding-based archive management")
    print("- Spread-aware acceptance criteria")

    return {
        'movns': {
            'hv': movns_hv,
            'spacing': movns_spacing,
            'time': movns_time,
            'solutions': len(movns_solutions)
        },
        'moead': {
            'hv': moead_hv,
            'spacing': moead_spacing,
            'time': moead_time,
            'solutions': len(moead_solutions)
        }
    }


if __name__ == "__main__":
    results = run_comparison('fastapi', iterations=25)