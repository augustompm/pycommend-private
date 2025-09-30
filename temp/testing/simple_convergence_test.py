"""
Simple convergence test to verify algorithms are improving
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('pycommend-code/src')
os.chdir('pycommend-code')

def test_single_algorithm(algo_name, algo_class, package='fastapi', iterations=30):
    """
    Test a single algorithm and return metrics
    """
    print(f"\nTesting {algo_name}...")

    if algo_name == 'MOVNS':
        algo = algo_class(package, archive_size=100, max_iterations=iterations, track_metrics=True)
    else:
        algo = algo_class(package, pop_size=100, max_gen=iterations, track_metrics=True)

    solutions = algo.run()
    metrics = algo.get_metrics_history()

    if metrics and 'hypervolume' in metrics and metrics['hypervolume']:
        hv = metrics['hypervolume']
        print(f"  Iterations tracked: {len(hv)}")
        print(f"  Initial HV: {hv[0]:.6f}")
        print(f"  Final HV: {hv[-1]:.6f}")

        if hv[-1] > hv[0]:
            print(f"  ✓ Improvement: {((hv[-1] - hv[0]) / (hv[0] + 1e-10)) * 100:.1f}%")
        else:
            print(f"  ✗ No improvement detected")

        improvements = sum(1 for i in range(1, len(hv)) if hv[i] > hv[i-1])
        print(f"  Steps improved: {improvements}/{len(hv)-1}")

        return hv
    else:
        print(f"  ✗ No metrics recorded")
        return None

def main():
    print("CONVERGENCE VERIFICATION")
    print("="*60)

    from optimizer.movns_vns import MOVNS_VNS
    from optimizer.moead_vns import MOEAD_VNS

    results = {}

    hv_movns = test_single_algorithm('MOVNS', MOVNS_VNS, iterations=20)
    if hv_movns:
        results['MOVNS'] = hv_movns

    hv_moead = test_single_algorithm('MOEA/D', MOEAD_VNS, iterations=20)
    if hv_moead:
        results['MOEA/D'] = hv_moead

    if results:
        plt.figure(figsize=(10, 6))

        for name, hv in results.items():
            plt.plot(range(len(hv)), hv, label=f'{name} (improvement: {((hv[-1]-hv[0])/(hv[0]+1e-10))*100:.0f}%)',
                    linewidth=2, marker='o', markersize=3)

        plt.xlabel('Iteration/Generation', fontsize=12)
        plt.ylabel('Hypervolume', fontsize=12)
        plt.title('Convergence Verification - Hypervolume Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../convergence_simple.png', dpi=150)
        print(f"\nPlot saved to convergence_simple.png")
        plt.show()

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY:")

    for name, hv in results.items():
        slope = np.polyfit(range(len(hv)), hv, 1)[0]
        if slope > 0:
            print(f"  {name}: Positive trend (slope={slope:.6f})")
        else:
            print(f"  {name}: Negative/flat trend (slope={slope:.6f})")

if __name__ == "__main__":
    main()