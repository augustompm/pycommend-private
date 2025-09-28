"""
MOVNS vs MOEA/D Baseline Comparison
Following rules.json - Real execution, no shortcuts
"""

import sys
import time
import numpy as np

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS
from moead_vns import MOEAD_VNS


def compare_algorithms(package='numpy', iterations=10):
    """Compare MOVNS with MOEA/D on same problem"""

    print("="*80)
    print(f"MOVNS vs MOEA/D COMPARISON - {package.upper()}")
    print("="*80)

    # Test MOVNS
    print(f"\n1. Testing MOVNS on {package}:")
    print("-"*60)
    start = time.time()
    movns = MOVNS_VNS(package, archive_size=20, max_iterations=iterations, track_metrics=True)
    movns_sols = movns.run()
    movns_time = time.time() - start
    movns_metrics = movns.get_metrics_history()

    # Test MOEA/D with equivalent parameters
    print(f"\n2. Testing MOEA/D on {package}:")
    print("-"*60)
    start = time.time()
    moead = MOEAD_VNS(package, pop_size=20, max_gen=iterations, track_metrics=True)
    moead_sols = moead.run()
    moead_time = time.time() - start
    moead_metrics = moead.get_metrics_history()

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\nSolutions Found:")
    print(f"  MOVNS:  {len(movns_sols)} solutions in {movns_time:.2f}s")
    print(f"  MOEA/D: {len(moead_sols)} solutions in {moead_time:.2f}s")

    print(f"\nExecution Time:")
    if movns_time < moead_time:
        print(f"  MOVNS is {(moead_time/movns_time - 1)*100:.1f}% faster")
    else:
        print(f"  MOEA/D is {(movns_time/moead_time - 1)*100:.1f}% faster")

    # Compare hypervolume
    if movns_metrics and movns_metrics.get('hypervolume') and moead_metrics and moead_metrics.get('hypervolume'):
        movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics['hypervolume'] else 0
        moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics['hypervolume'] else 0

        print(f"\nHypervolume (Quality):")
        print(f"  MOVNS:  {movns_hv:.4f}")
        print(f"  MOEA/D: {moead_hv:.4f}")

        if movns_hv > moead_hv:
            improvement = (movns_hv / moead_hv - 1) * 100 if moead_hv > 0 else 100
            print(f"  MOVNS is {improvement:.1f}% better")
        elif moead_hv > movns_hv:
            improvement = (moead_hv / movns_hv - 1) * 100 if movns_hv > 0 else 100
            print(f"  MOEA/D is {improvement:.1f}% better")
        else:
            print(f"  Equal performance")

    # Compare best solutions
    print(f"\nBest Solution Comparison:")

    if movns_sols:
        best_movns = movns_sols[0]
        print(f"\nMOVNS Best:")
        print(f"  Packages: {', '.join(best_movns['packages'][:5])}...")
        print(f"  LU: {best_movns['objectives']['linked_usage']:.2f}")
        print(f"  SS: {best_movns['objectives']['semantic_similarity']:.4f}")
        print(f"  Size: {best_movns['objectives']['set_size']:.0f}")

    if moead_sols:
        best_moead = moead_sols[0]
        print(f"\nMOEA/D Best:")
        print(f"  Packages: {', '.join(best_moead['packages'][:5])}...")
        print(f"  LU: {best_moead['objectives']['linked_usage']:.2f}")
        print(f"  SS: {best_moead['objectives']['semantic_similarity']:.4f}")
        print(f"  Size: {best_moead['objectives']['set_size']:.0f}")

    # Determine winner
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    movns_score = 0
    moead_score = 0

    if len(movns_sols) > len(moead_sols):
        movns_score += 1
        print("✓ MOVNS found more solutions")
    elif len(moead_sols) > len(movns_sols):
        moead_score += 1
        print("✓ MOEA/D found more solutions")

    if movns_time < moead_time:
        movns_score += 1
        print("✓ MOVNS is faster")
    else:
        moead_score += 1
        print("✓ MOEA/D is faster")

    if movns_metrics and moead_metrics:
        if movns_hv > moead_hv:
            movns_score += 2  # Quality is double weighted
            print("✓ MOVNS has better quality (2x)")
        elif moead_hv > movns_hv:
            moead_score += 2
            print("✓ MOEA/D has better quality (2x)")

    print(f"\nFinal Score: MOVNS {movns_score} - {moead_score} MOEA/D")

    if movns_score > moead_score:
        print("\n[WINNER] MOVNS demonstrates superiority for VNS paper")
    elif moead_score > movns_score:
        print("\n[WINNER] MOEA/D performs better - MOVNS needs improvement")
    else:
        print("\n[TIE] Both algorithms perform equally well")

    return {
        'movns': {
            'solutions': len(movns_sols),
            'time': movns_time,
            'hypervolume': movns_hv if movns_metrics else None
        },
        'moead': {
            'solutions': len(moead_sols),
            'time': moead_time,
            'hypervolume': moead_hv if moead_metrics else None
        },
        'winner': 'movns' if movns_score > moead_score else ('moead' if moead_score > movns_score else 'tie')
    }


def main():
    """Run comparison tests"""
    print("="*80)
    print("MOVNS vs MOEA/D - BASELINE COMPARISON")
    print("For VNS Conference Paper")
    print("Following rules.json - No shortcuts")
    print("="*80)

    # Quick test with numpy
    results = compare_algorithms('numpy', iterations=5)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Result: {results['winner'].upper()}")

    if results['winner'] == 'movns':
        print("\n✓ MOVNS ready for VNS paper submission")
    elif results['winner'] == 'tie':
        print("\n✓ MOVNS competitive with state-of-the-art")
    else:
        print("\n⚠ MOVNS needs further optimization")


if __name__ == '__main__':
    main()