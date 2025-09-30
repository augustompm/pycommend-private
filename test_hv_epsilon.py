"""
Test with HV and Epsilon Indicator
Epsilon is weakly Pareto compliant and closer to HV behavior
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


def generate_ideal_pareto_front(n_points=200):
    """Generate ideal Pareto front for epsilon calculation"""
    pareto = []

    for i in range(n_points):
        t = i / (n_points - 1)

        lu = -10000 * (1 - t)
        ss = -1.0 * (1 - t * 0.5)
        rss = 2 + 8 * t

        pareto.append([lu, ss, rss])

    return np.array(pareto)


def run_comparison(package='fastapi', iterations=20):
    """Run comparison with HV and Epsilon indicator"""

    print("="*70)
    print("HV AND EPSILON INDICATOR COMPARISON")
    print("="*70)

    ideal_front = generate_ideal_pareto_front(200)
    print(f"Generated ideal Pareto front with {len(ideal_front)} points")

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
    movns_epsilon = qm.epsilon_indicator(movns_objectives, ideal_front)

    movns_diversity = qm.diversity(movns_objectives) if len(movns_objectives) > 1 else 0

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
    print(f"Epsilon: {movns_epsilon:.4f}")
    print(f"Diversity: {movns_diversity:.4f}")

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

    moead_epsilon = qm.epsilon_indicator(moead_objectives, ideal_front)

    moead_diversity = qm.diversity(moead_objectives) if len(moead_objectives) > 1 else 0

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
    print(f"Epsilon: {moead_epsilon:.4f}")
    print(f"Diversity: {moead_diversity:.4f}")

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n1. HYPERVOLUME (higher is better)")
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
        print(f"   Winner: MOVNS")
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D")

    print("\n2. EPSILON INDICATOR (lower is better)")
    print(f"   MOVNS:  {movns_epsilon:.4f}")
    print(f"   MOEA/D: {moead_epsilon:.4f}")
    if movns_epsilon < float('inf') and moead_epsilon < float('inf'):
        if movns_epsilon < moead_epsilon:
            ratio = moead_epsilon / movns_epsilon if movns_epsilon > 0 else float('inf')
            print(f"   Winner: MOVNS ({ratio:.2f}x better)")
        else:
            ratio = movns_epsilon / moead_epsilon if moead_epsilon > 0 else float('inf')
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")

    print("\n3. DIVERSITY (higher is better)")
    print(f"   MOVNS:  {movns_diversity:.4f}")
    print(f"   MOEA/D: {moead_diversity:.4f}")
    if movns_diversity > moead_diversity:
        print(f"   Winner: MOVNS")
    else:
        print(f"   Winner: MOEA/D")

    print("\n4. ARCHIVE SIZE")
    print(f"   MOVNS:  {len(movns_solutions)}")
    print(f"   MOEA/D: {len(moead_solutions)}")

    print("\n5. BEST OBJECTIVES")
    print(f"   Linked Usage:")
    print(f"     MOVNS:  {best_movns['lu']:.0f}")
    print(f"     MOEA/D: {best_moead['lu']:.0f}")
    print(f"   Semantic Similarity:")
    print(f"     MOVNS:  {best_movns['ss']:.4f}")
    print(f"     MOEA/D: {best_moead['ss']:.4f}")
    print(f"   Set Size:")
    print(f"     MOVNS:  {best_movns['rss']}")
    print(f"     MOEA/D: {best_moead['rss']}")

    print("\n6. EXECUTION TIME")
    print(f"   MOVNS:  {movns_time:.1f}s")
    print(f"   MOEA/D: {moead_time:.1f}s")

    print("\n" + "="*70)
    print("OVERALL ASSESSMENT (Key Metrics)")
    print("="*70)

    movns_score = 0
    moead_score = 0

    if movns_hv > moead_hv:
        movns_score += 1
        print("+ HV: MOVNS")
    else:
        moead_score += 1
        print("+ HV: MOEA/D")

    if movns_epsilon < moead_epsilon:
        movns_score += 1
        print("+ Epsilon: MOVNS")
    else:
        moead_score += 1
        print("+ Epsilon: MOEA/D")

    print(f"\nFinal Score: MOVNS {movns_score} - {moead_score} MOEA/D")

    if movns_score > moead_score:
        print("\nWINNER: MOVNS Advanced")
        print("Both HV and Epsilon confirm MOVNS superiority")
    elif moead_score > movns_score:
        print("\nWINNER: MOEA/D Normalized")
    else:
        print("\nTIE: Algorithms split the metrics")

    print("\n" + "="*70)
    print("METRIC CORRELATION ANALYSIS")
    print("="*70)

    print("\nEpsilon Indicator:")
    print("- Weakly Pareto compliant (like HV)")
    print("- Measures minimum translation to dominate reference")
    print("- Lower values = closer to ideal Pareto front")
    print("- More correlated with HV than IGD+")

    if (movns_hv > moead_hv and movns_epsilon < moead_epsilon):
        print("\nConsistent results: MOVNS wins both HV and Epsilon")
        print("Confirms MOVNS produces superior Pareto front")
    elif (moead_hv > movns_hv and moead_epsilon < movns_epsilon):
        print("\nConsistent results: MOEA/D wins both HV and Epsilon")
        print("Confirms MOEA/D produces superior Pareto front")
    else:
        print("\nInconsistent results between HV and Epsilon")
        print("May indicate different strengths in objective space")


if __name__ == "__main__":
    run_comparison('fastapi', iterations=20)