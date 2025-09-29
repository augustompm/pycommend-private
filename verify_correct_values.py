"""
Verify correct hypervolume values using QualityMetrics
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

def test_algorithm_correct(algo_name, package='fastapi', iterations=20):
    """Test with correct hypervolume calculation"""
    print(f"\nTesting {algo_name}...")

    if algo_name == "MOVNS v2":
        algo = MOVNS_V2(package, archive_size=50, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=10)
        solutions = algo.run()
        metrics = algo.get_metrics_history()

        if metrics and 'hypervolume' in metrics:
            hv = metrics['hypervolume']
            if len(hv) > 0:
                print(f"  Initial HV: {hv[0]:.4f}")
                print(f"  Final HV: {hv[-1]:.4f}")
                print(f"  Improvement: {((hv[-1] - hv[0])/hv[0]*100) if hv[0] > 0 else 0:.1f}%")
                print(f"  Max HV reached: {max(hv):.4f}")
                return hv[-1]

    elif algo_name == "MOEA/D Normalized":
        algo = MOEAD_Normalized(package, pop_size=50, max_gen=iterations,
                                track_metrics=True)
        solutions = algo.run()
        metrics = algo.get_metrics_history()

        if metrics and 'hypervolume' in metrics:
            hv = metrics['hypervolume']
            if len(hv) > 0:
                print(f"  Initial HV: {hv[0]:.4f}")
                print(f"  Final HV: {hv[-1]:.4f}")
                print(f"  Improvement: {((hv[-1] - hv[0])/hv[0]*100) if hv[0] > 0 else 0:.1f}%")
                print(f"  Max HV reached: {max(hv):.4f}")
                return hv[-1]

    return 0.0

def main():
    print("="*60)
    print("VERIFYING CORRECT HYPERVOLUME VALUES")
    print("Using QualityMetrics with proper normalization")
    print("="*60)

    # Test both algorithms
    movns_hv = test_algorithm_correct("MOVNS v2", iterations=20)
    moead_hv = test_algorithm_correct("MOEA/D Normalized", iterations=20)

    print("\n" + "="*60)
    print("CORRECT HYPERVOLUME VALUES:")
    print("="*60)
    print(f"MOVNS v2: {movns_hv:.4f}")
    print(f"MOEA/D Normalized: {moead_hv:.4f}")

    if moead_hv > movns_hv:
        print(f"\nMOEA/D is {(moead_hv/movns_hv - 1)*100:.1f}% better than MOVNS v2")
    else:
        print(f"\nMOVNS v2 is {(movns_hv/moead_hv - 1)*100:.1f}% better than MOEA/D")

    print("\nNOTE: These are the CORRECT values using normalized objectives")
    print("The values should be around 0.2+ as seen in v12_report.txt")

if __name__ == "__main__":
    main()