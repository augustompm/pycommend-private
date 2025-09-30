"""
Quick Fair Comparison - Based on 2024 Best Practices
Same number of function evaluations for both algorithms
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


def main():
    """Fair comparison with same evaluation budget"""

    print("="*70)
    print("FAIR COMPARISON - SAME FUNCTION EVALUATIONS")
    print("="*70)

    package = 'fastapi'
    evaluation_budget = 3000

    movns_archive = 100
    movns_local_search = 20
    movns_iterations = (evaluation_budget - movns_archive) // movns_local_search
    movns_evaluations = movns_archive + (movns_iterations * movns_local_search)

    moead_pop = 100
    moead_generations = (evaluation_budget - moead_pop) // moead_pop
    moead_evaluations = moead_pop + (moead_generations * moead_pop)

    print(f"\nEvaluation Budget: {evaluation_budget}")
    print(f"\nMOVNS Advanced:")
    print(f"  Iterations: {movns_iterations}")
    print(f"  Archive: {movns_archive}")
    print(f"  Estimated evaluations: {movns_evaluations}")

    print(f"\nMOEA/D:")
    print(f"  Generations: {moead_generations}")
    print(f"  Population: {moead_pop}")
    print(f"  Estimated evaluations: {moead_evaluations}")

    print("\n" + "-"*70)
    print("Running MOVNS Advanced...")

    movns = MOVNS_Advanced(package, archive_size=movns_archive, max_iterations=movns_iterations, track_metrics=True)

    movns_start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - movns_start

    movns_metrics = movns.get_metrics_history()
    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0 else 0

    movns_best = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        if -obj[0] > movns_best['lu']:
            movns_best['lu'] = -obj[0]
        if -obj[1] > movns_best['ss']:
            movns_best['ss'] = -obj[1]
        if obj[2] < movns_best['rss']:
            movns_best['rss'] = obj[2]

    print(f"  Completed in {movns_time:.1f}s")
    print(f"  Archive size: {len(movns_solutions)}")
    print(f"  Hypervolume: {movns_hv:.4f}")

    print("\n" + "-"*70)
    print("Running MOEA/D...")

    moead = MOEAD_Normalized(package, pop_size=moead_pop, max_gen=moead_generations)

    moead_start = time.time()
    moead_solutions = moead.run()
    moead_time = time.time() - moead_start

    moead_metrics = moead.get_metrics_history()
    moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0 else 0

    moead_best = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        if -obj[0] > moead_best['lu']:
            moead_best['lu'] = -obj[0]
        if -obj[1] > moead_best['ss']:
            moead_best['ss'] = -obj[1]
        if obj[2] < moead_best['rss']:
            moead_best['rss'] = obj[2]

    print(f"  Completed in {moead_time:.1f}s")
    print(f"  Archive size: {len(moead_solutions)}")
    print(f"  Hypervolume: {moead_hv:.4f}")

    print("\n" + "="*70)
    print("FAIR COMPARISON RESULTS")
    print("="*70)

    print(f"\n1. HYPERVOLUME (main metric)")
    print(f"   MOVNS:  {movns_hv:.4f}")
    print(f"   MOEA/D: {moead_hv:.4f}")

    if movns_hv > 0 and moead_hv > 0:
        if movns_hv > moead_hv:
            ratio = movns_hv / moead_hv
            print(f"   Winner: MOVNS ({ratio:.2f}x better)")
        else:
            ratio = moead_hv / movns_hv
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")
    elif movns_hv > 0:
        print(f"   Winner: MOVNS (MOEA/D failed)")
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D (MOVNS failed)")

    print(f"\n2. ARCHIVE SIZE")
    print(f"   MOVNS:  {len(movns_solutions)} solutions")
    print(f"   MOEA/D: {len(moead_solutions)} solutions")

    print(f"\n3. BEST OBJECTIVES")
    print(f"   Linked Usage:")
    print(f"     MOVNS:  {movns_best['lu']:.0f}")
    print(f"     MOEA/D: {moead_best['lu']:.0f}")
    print(f"   Semantic Similarity:")
    print(f"     MOVNS:  {movns_best['ss']:.4f}")
    print(f"     MOEA/D: {moead_best['ss']:.4f}")
    print(f"   Set Size:")
    print(f"     MOVNS:  {movns_best['rss']}")
    print(f"     MOEA/D: {moead_best['rss']}")

    print(f"\n4. EXECUTION TIME")
    print(f"   MOVNS:  {movns_time:.1f}s")
    print(f"   MOEA/D: {moead_time:.1f}s")
    print(f"   Ratio: {moead_time/movns_time:.1f}x")

    print("\n" + "="*70)
    print("CONCLUSION (Fair Comparison)")
    print("="*70)

    score_movns = 0
    score_moead = 0

    if movns_hv > moead_hv:
        score_movns += 2
        print("✓ Hypervolume: MOVNS (2 points)")
    else:
        score_moead += 2
        print("✓ Hypervolume: MOEA/D (2 points)")

    if movns_best['lu'] > moead_best['lu']:
        score_movns += 1
        print("✓ Best LU: MOVNS (1 point)")
    else:
        score_moead += 1
        print("✓ Best LU: MOEA/D (1 point)")

    if movns_best['ss'] > moead_best['ss']:
        score_movns += 1
        print("✓ Best SS: MOVNS (1 point)")
    else:
        score_moead += 1
        print("✓ Best SS: MOEA/D (1 point)")

    if movns_time < moead_time:
        score_movns += 1
        print("✓ Speed: MOVNS (1 point)")
    else:
        score_moead += 1
        print("✓ Speed: MOEA/D (1 point)")

    print(f"\nFinal Score: MOVNS {score_movns} - {score_moead} MOEA/D")

    if score_movns > score_moead:
        print("\nWINNER: MOVNS Advanced")
        print("Under fair comparison with same evaluations, MOVNS is superior")
    elif score_moead > score_movns:
        print("\nWINNER: MOEA/D")
        print("Under fair comparison with same evaluations, MOEA/D is superior")
    else:
        print("\nTIE: Both algorithms perform similarly")


if __name__ == "__main__":
    main()