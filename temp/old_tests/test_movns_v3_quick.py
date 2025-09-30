"""
Quick test for MOVNS v3 vs MOEA/D
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v3 import MOVNS_V3
from optimizer.moead_normalized import MOEAD_Normalized

def quick_test():
    """Quick performance test"""
    print("="*60)
    print("MOVNS v3 vs MOEA/D - QUICK TEST")
    print("="*60)

    iterations = 15
    results = {}

    # Test MOEA/D
    print("\nTesting MOEA/D Normalized...")
    start = time.time()
    moead = MOEAD_Normalized('fastapi', pop_size=30, max_gen=iterations, track_metrics=True)
    moead_sol = moead.run()
    moead_time = time.time() - start
    moead_metrics = moead.get_metrics_history()

    if moead_metrics and 'hypervolume' in moead_metrics:
        hv = moead_metrics['hypervolume']
        if len(hv) > 0:
            results['MOEA/D'] = {
                'initial': hv[0],
                'final': hv[-1],
                'improvement': ((hv[-1] - hv[0])/hv[0]*100) if hv[0] > 0 else 0,
                'solutions': len(moead_sol),
                'time': moead_time
            }
            print(f"  HV: {hv[0]:.4f} -> {hv[-1]:.4f} ({results['MOEA/D']['improvement']:.1f}%)")
            print(f"  Solutions: {len(moead_sol)}, Time: {moead_time:.1f}s")

    # Test MOVNS v3
    print("\nTesting MOVNS v3...")
    start = time.time()
    movns = MOVNS_V3('fastapi', archive_size=50, max_iterations=iterations,
                    track_metrics=True, min_no_improvement=10)
    movns_sol = movns.run()
    movns_time = time.time() - start
    movns_metrics = movns.get_metrics_history()

    if movns_metrics and 'hypervolume' in movns_metrics:
        hv = movns_metrics['hypervolume']
        if len(hv) > 0:
            results['MOVNS v3'] = {
                'initial': hv[0],
                'final': hv[-1],
                'improvement': ((hv[-1] - hv[0])/hv[0]*100) if hv[0] > 0 else 0,
                'solutions': len(movns_sol),
                'time': movns_time
            }
            print(f"  HV: {hv[0]:.4f} -> {hv[-1]:.4f} ({results['MOVNS v3']['improvement']:.1f}%)")
            print(f"  Solutions: {len(movns_sol)}, Time: {movns_time:.1f}s")

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    if 'MOEA/D' in results and 'MOVNS v3' in results:
        moead_hv = results['MOEA/D']['final']
        movns_hv = results['MOVNS v3']['final']

        print(f"\nFinal Hypervolume:")
        print(f"  MOEA/D: {moead_hv:.4f}")
        print(f"  MOVNS v3: {movns_hv:.4f}")

        if movns_hv > moead_hv:
            improvement = (movns_hv/moead_hv - 1) * 100
            print(f"\n✓ SUCCESS: MOVNS v3 beats MOEA/D by {improvement:.1f}%!")
        else:
            deficit = (moead_hv/movns_hv - 1) * 100
            print(f"\n✗ MOVNS v3 is {deficit:.1f}% behind MOEA/D")

        print(f"\nConvergence Speed:")
        print(f"  MOEA/D: {results['MOEA/D']['improvement']:.1f}% in {results['MOEA/D']['time']:.1f}s")
        print(f"  MOVNS v3: {results['MOVNS v3']['improvement']:.1f}% in {results['MOVNS v3']['time']:.1f}s")

    print("="*60)

if __name__ == "__main__":
    quick_test()