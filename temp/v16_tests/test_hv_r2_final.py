"""
Final test with HV and R2 Indicator
R2 is weakly Pareto compliant and highly correlated with HV
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def run_comparison(package='fastapi', iterations=25):
    """Run comparison with HV and R2 indicator"""

    print("="*70)
    print("HV AND R2 INDICATOR FINAL COMPARISON")
    print("="*70)

    np.random.seed(42)

    print("\n" + "-"*70)
    print("1. MOVNS ADVANCED")
    print("-"*70)

    movns = MOVNS_Advanced(
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

    movns_objectives = []
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
    movns_objectives = np.array(movns_objectives)

    qm = QualityMetrics()
    movns_r2 = qm.r2_indicator(movns_objectives)

    best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for obj in movns_objectives:
        if -obj[0] > best_movns['lu']:
            best_movns['lu'] = -obj[0]
        if -obj[1] > best_movns['ss']:
            best_movns['ss'] = -obj[1]
        if obj[2] < best_movns['rss']:
            best_movns['rss'] = obj[2]

    print(f"Completed in {movns_time:.1f}s")
    print(f"Archive: {len(movns_solutions)} solutions")
    print(f"HV: {movns_hv:.4f}")
    print(f"R2: {movns_r2:.4f}")

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

    moead_objectives = []
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        moead_objectives.append(obj)
    moead_objectives = np.array(moead_objectives)

    moead_r2 = qm.r2_indicator(moead_objectives)

    best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for obj in moead_objectives:
        if -obj[0] > best_moead['lu']:
            best_moead['lu'] = -obj[0]
        if -obj[1] > best_moead['ss']:
            best_moead['ss'] = -obj[1]
        if obj[2] < best_moead['rss']:
            best_moead['rss'] = obj[2]

    print(f"Completed in {moead_time:.1f}s")
    print(f"Archive: {len(moead_solutions)} solutions")
    print(f"HV: {moead_hv:.4f}")
    print(f"R2: {moead_r2:.4f}")

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n1. HYPERVOLUME (higher is better)")
    print(f"   MOVNS:  {movns_hv:.4f}")
    print(f"   MOEA/D: {moead_hv:.4f}")
    hv_winner = None
    if movns_hv > 0 and moead_hv > 0:
        if movns_hv > moead_hv:
            ratio = movns_hv / moead_hv
            print(f"   Winner: MOVNS ({ratio:.2f}x better)")
            hv_winner = 'MOVNS'
        else:
            ratio = moead_hv / movns_hv
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            hv_winner = 'MOEA/D'
    elif movns_hv > 0:
        print(f"   Winner: MOVNS")
        hv_winner = 'MOVNS'
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D")
        hv_winner = 'MOEA/D'

    print("\n2. R2 INDICATOR (lower is better)")
    print(f"   MOVNS:  {movns_r2:.4f}")
    print(f"   MOEA/D: {moead_r2:.4f}")
    r2_winner = None
    if movns_r2 < float('inf') and moead_r2 < float('inf'):
        if movns_r2 < moead_r2:
            ratio = moead_r2 / movns_r2 if movns_r2 > 0 else float('inf')
            print(f"   Winner: MOVNS ({ratio:.2f}x better)")
            r2_winner = 'MOVNS'
        else:
            ratio = movns_r2 / moead_r2 if moead_r2 > 0 else float('inf')
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
            r2_winner = 'MOEA/D'

    print("\n3. BEST OBJECTIVES")
    print(f"   Linked Usage:")
    print(f"     MOVNS:  {best_movns['lu']:.0f}")
    print(f"     MOEA/D: {best_moead['lu']:.0f}")
    print(f"   Semantic Similarity:")
    print(f"     MOVNS:  {best_movns['ss']:.4f}")
    print(f"     MOEA/D: {best_moead['ss']:.4f}")
    print(f"   Set Size:")
    print(f"     MOVNS:  {best_movns['rss']}")
    print(f"     MOEA/D: {best_moead['rss']}")

    print("\n4. EXECUTION TIME")
    print(f"   MOVNS:  {movns_time:.1f}s")
    print(f"   MOEA/D: {moead_time:.1f}s")
    print(f"   Speed ratio: {moead_time/movns_time:.1f}x")

    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    movns_score = 0
    moead_score = 0

    if hv_winner == 'MOVNS':
        movns_score += 1
        print("+ HV: MOVNS")
    elif hv_winner == 'MOEA/D':
        moead_score += 1
        print("+ HV: MOEA/D")

    if r2_winner == 'MOVNS':
        movns_score += 1
        print("+ R2: MOVNS")
    elif r2_winner == 'MOEA/D':
        moead_score += 1
        print("+ R2: MOEA/D")

    print(f"\nFinal Score: MOVNS {movns_score} - {moead_score} MOEA/D")

    if movns_score == 2:
        print("\n**WINNER: MOVNS Advanced (2-0)**")
        print("Clear superiority confirmed by both HV and R2")
    elif moead_score == 2:
        print("\n**WINNER: MOEA/D Normalized (0-2)**")
        print("Clear superiority confirmed by both HV and R2")
    else:
        print("\n**SPLIT: No clear winner**")
        print("Algorithms have different strengths")

    print("\n" + "="*70)
    print("METRIC EXPLANATION")
    print("="*70)

    print("\nR2 Indicator:")
    print("- Utility-based metric (weakly Pareto compliant)")
    print("- Highly correlated with HV")
    print("- Measures average utility across weight vectors")
    print("- Lower values = better Pareto front coverage")

    if hv_winner == r2_winner and hv_winner is not None:
        print(f"\nConsistent result: {hv_winner} wins both metrics")
        print("Strong evidence of superiority")
    else:
        print("\nInconsistent results suggest complex trade-offs")


if __name__ == "__main__":
    run_comparison('fastapi', iterations=25)