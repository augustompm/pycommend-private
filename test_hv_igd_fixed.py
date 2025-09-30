"""
Test with HV (from internal metrics) and IGD+ (calculated externally)
Fixing HV calculation to use algorithm's internal tracking
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


def generate_reference_set(n_points=500):
    """Generate reference set for IGD+ calculation"""
    reference = []

    for i in range(n_points):
        w1 = np.random.random()
        w2 = np.random.random() * (1 - w1)
        w3 = 1 - w1 - w2

        lu = -10000 * w1
        ss = -1.0 * w2
        rss = 2 + 13 * w3

        reference.append([lu, ss, rss])

    return np.array(reference)


def calculate_igd_plus(solutions, reference, algo):
    """Calculate IGD+ metric"""
    if not solutions or len(solutions) == 0:
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
    igd_plus_value = qm.igd_plus(objectives, reference)

    return igd_plus_value


def calculate_spacing(solutions, algo):
    """Calculate spacing metric"""
    if not solutions or len(solutions) == 0:
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
    spacing_value = qm.spacing(objectives) if len(objectives) > 1 else float('inf')

    return spacing_value


def run_comparison(package='fastapi', iterations=20):
    """Run comparison with HV from internal metrics and IGD+ calculated"""

    print("="*70)
    print("HV (INTERNAL) AND IGD+ COMPARISON TEST - FIXED")
    print("="*70)

    reference_set = generate_reference_set(500)
    print(f"Generated reference set with {len(reference_set)} points")

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

    # Get HV from internal metrics (this works!)
    movns_metrics = movns.get_metrics_history()
    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0 else 0

    # Calculate IGD+ externally
    movns_igd = calculate_igd_plus(movns_solutions, reference_set, movns)

    # Calculate spacing
    movns_spacing = calculate_spacing(movns_solutions, movns)

    # Get best objectives
    best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_movns['lu']:
            best_movns['lu'] = -obj[0]
        if -obj[1] > best_movns['ss']:
            best_movns['ss'] = -obj[1]
        if obj[2] < best_movns['rss']:
            best_movns['rss'] = obj[2]

    print(f"Completed in {movns_time:.1f}s")
    print(f"Archive: {len(movns_solutions)} solutions")
    print(f"HV (internal): {movns_hv:.4f}")
    print(f"IGD+: {movns_igd:.4f}")
    print(f"Spacing: {movns_spacing:.4f}")

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

    # Get HV from internal metrics
    moead_metrics = moead.get_metrics_history()
    moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0 else 0

    # Calculate IGD+ externally
    moead_igd = calculate_igd_plus(moead_solutions, reference_set, moead)

    # Calculate spacing
    moead_spacing = calculate_spacing(moead_solutions, moead)

    # Get best objectives
    best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_moead['lu']:
            best_moead['lu'] = -obj[0]
        if -obj[1] > best_moead['ss']:
            best_moead['ss'] = -obj[1]
        if obj[2] < best_moead['rss']:
            best_moead['rss'] = obj[2]

    print(f"Completed in {moead_time:.1f}s")
    print(f"Archive: {len(moead_solutions)} solutions")
    print(f"HV (internal): {moead_hv:.4f}")
    print(f"IGD+: {moead_igd:.4f}")
    print(f"Spacing: {moead_spacing:.4f}")

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n1. HYPERVOLUME (higher is better) - FROM INTERNAL TRACKING")
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
        print(f"   Winner: MOVNS (MOEA/D failed to produce HV)")
    elif moead_hv > 0:
        print(f"   Winner: MOEA/D (MOVNS failed to produce HV)")
    else:
        print(f"   Both algorithms failed to produce valid HV")

    print("\n2. IGD+ (lower is better)")
    print(f"   MOVNS:  {movns_igd:.4f}")
    print(f"   MOEA/D: {moead_igd:.4f}")
    if movns_igd < float('inf') and moead_igd < float('inf'):
        if movns_igd < moead_igd:
            ratio = moead_igd / movns_igd if movns_igd > 0 else float('inf')
            print(f"   Winner: MOVNS ({ratio:.2f}x better)")
        else:
            ratio = movns_igd / moead_igd if moead_igd > 0 else float('inf')
            print(f"   Winner: MOEA/D ({ratio:.2f}x better)")

    print("\n3. SPACING (lower is better)")
    print(f"   MOVNS:  {movns_spacing:.4f}")
    print(f"   MOEA/D: {moead_spacing:.4f}")
    if movns_spacing < moead_spacing:
        print(f"   Winner: MOVNS (better distribution)")
    else:
        print(f"   Winner: MOEA/D (better distribution)")

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
    print("OVERALL ASSESSMENT")
    print("="*70)

    movns_score = 0
    moead_score = 0

    # HV is most important (2 points)
    if movns_hv > moead_hv:
        movns_score += 2
        print("+ HV: MOVNS (2 points)")
    else:
        moead_score += 2
        print("+ HV: MOEA/D (2 points)")

    # IGD+ is second most important (2 points)
    if movns_igd < moead_igd:
        movns_score += 2
        print("+ IGD+: MOVNS (2 points)")
    else:
        moead_score += 2
        print("+ IGD+: MOEA/D (2 points)")

    # Spacing is third (1 point)
    if movns_spacing < moead_spacing:
        movns_score += 1
        print("+ Spacing: MOVNS (1 point)")
    else:
        moead_score += 1
        print("+ Spacing: MOEA/D (1 point)")

    print(f"\nFinal Score: MOVNS {movns_score} - {moead_score} MOEA/D")

    if movns_score > moead_score:
        print("\nWINNER: MOVNS Advanced")
    elif moead_score > movns_score:
        print("\nWINNER: MOEA/D Normalized")
    else:
        print("\nTIE: Both algorithms comparable")

    # Print HV history to verify tracking
    print("\n" + "="*70)
    print("HV TRACKING VERIFICATION")
    print("="*70)

    if len(movns_metrics.get('hypervolume', [])) > 0:
        print(f"\nMOVNS HV history (last 5):")
        last_5 = movns_metrics['hypervolume'][-5:]
        print(f"   {[f'{h:.4f}' for h in last_5]}")

    if moead_metrics and len(moead_metrics.get('hypervolume', [])) > 0:
        print(f"\nMOEA/D HV history (last 5):")
        last_5 = moead_metrics['hypervolume'][-5:]
        print(f"   {[f'{h:.4f}' for h in last_5]}")

    return {
        'movns': {
            'hv': movns_hv,
            'igd': movns_igd,
            'spacing': movns_spacing,
            'time': movns_time,
            'solutions': len(movns_solutions)
        },
        'moead': {
            'hv': moead_hv,
            'igd': moead_igd,
            'spacing': moead_spacing,
            'time': moead_time,
            'solutions': len(moead_solutions)
        }
    }


if __name__ == "__main__":
    results = run_comparison('fastapi', iterations=15)