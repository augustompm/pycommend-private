"""
Test MOEA/D with Hypervolume tracking
"""

import sys
import os
import numpy as np
import time

os.chdir('pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from moead_vns import MOEAD_VNS
from nsga2_vns import NSGA2_VNS


def test_moead_with_hypervolume():
    """Test MOEA/D with hypervolume tracking"""

    print("="*70)
    print("TESTING MOEA/D WITH HYPERVOLUME TRACKING")
    print("="*70)

    package = 'fastapi'

    # Run MOEA/D with metrics
    print(f"\n1. Running MOEA/D for {package} with metrics...")
    print("-"*70)

    moead = MOEAD_VNS(package, pop_size=20, n_neighbors=10, max_gen=20,
                      decomposition='tchebycheff', track_metrics=True)

    moead_solutions = moead.run()
    moead_metrics = moead.get_metrics_history()

    if moead_metrics and moead_metrics['hypervolume']:
        print("\nMOEA/D Metrics Evolution:")
        hv_values = moead_metrics['hypervolume']
        print(f"  Initial HV: {hv_values[0]:.4f}")
        print(f"  Final HV: {hv_values[-1]:.4f}")
        if len(hv_values) > 1:
            improvement = (hv_values[-1] - hv_values[0]) / abs(hv_values[0]) * 100
            print(f"  HV Improvement: {improvement:+.1f}%")


def compare_algorithms_with_hypervolume():
    """Compare NSGA-II and MOEA/D using hypervolume"""

    print("\n" + "="*70)
    print("ALGORITHM COMPARISON WITH HYPERVOLUME")
    print("="*70)

    package = 'scikit-learn'
    pop_size = 30
    max_gen = 20

    # Run NSGA-II
    print(f"\nRunning NSGA-II for {package}...")
    start = time.time()
    nsga2 = NSGA2_VNS(package, pop_size=pop_size, max_gen=max_gen, track_metrics=True)
    nsga2_solutions = nsga2.run()
    nsga2_time = time.time() - start
    nsga2_metrics = nsga2.get_metrics_history()

    # Run MOEA/D
    print(f"\nRunning MOEA/D for {package}...")
    start = time.time()
    moead = MOEAD_VNS(package, pop_size=pop_size, n_neighbors=15, max_gen=max_gen,
                      decomposition='tchebycheff', track_metrics=True)
    moead_solutions = moead.run()
    moead_time = time.time() - start
    moead_metrics = moead.get_metrics_history()

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\nNSGA-II:")
    print(f"  Solutions: {len(nsga2_solutions)}")
    print(f"  Time: {nsga2_time:.2f}s")
    if nsga2_metrics and nsga2_metrics['hypervolume']:
        hv_values = nsga2_metrics['hypervolume']
        print(f"  Initial HV: {hv_values[0]:.4f}")
        print(f"  Final HV: {hv_values[-1]:.4f}")
        print(f"  Final Spacing: {nsga2_metrics['spacing'][-1]:.4f}")
        print(f"  Final Diversity: {nsga2_metrics['diversity'][-1]:.4f}")

    print(f"\nMOEA/D:")
    print(f"  Solutions: {len(moead_solutions)}")
    print(f"  Time: {moead_time:.2f}s")
    if moead_metrics and moead_metrics['hypervolume']:
        hv_values = moead_metrics['hypervolume']
        print(f"  Initial HV: {hv_values[0]:.4f}")
        print(f"  Final HV: {hv_values[-1]:.4f}")
        print(f"  Final Spacing: {moead_metrics['spacing'][-1]:.4f}")
        print(f"  Final Diversity: {moead_metrics['diversity'][-1]:.4f}")

    # Determine winner by hypervolume
    print("\n" + "-"*70)
    print("WINNER BY HYPERVOLUME:")
    if nsga2_metrics and moead_metrics:
        nsga2_hv = nsga2_metrics['hypervolume'][-1] if nsga2_metrics['hypervolume'] else 0
        moead_hv = moead_metrics['hypervolume'][-1] if moead_metrics['hypervolume'] else 0

        if nsga2_hv > moead_hv:
            print(f"NSGA-II wins! (HV: {nsga2_hv:.4f} vs {moead_hv:.4f})")
            print(f"Advantage: {(nsga2_hv - moead_hv) / moead_hv * 100:.1f}%")
        elif moead_hv > nsga2_hv:
            print(f"MOEA/D wins! (HV: {moead_hv:.4f} vs {nsga2_hv:.4f})")
            print(f"Advantage: {(moead_hv - nsga2_hv) / nsga2_hv * 100:.1f}%")
        else:
            print("TIE! Both algorithms achieved same hypervolume")


def plot_hypervolume_evolution():
    """Show hypervolume evolution for both algorithms"""

    print("\n" + "="*70)
    print("HYPERVOLUME EVOLUTION")
    print("="*70)

    package = 'numpy'

    # Quick runs for evolution tracking
    nsga2 = NSGA2_VNS(package, pop_size=20, max_gen=30, track_metrics=True)
    nsga2_solutions = nsga2.run()
    nsga2_metrics = nsga2.get_metrics_history()

    moead = MOEAD_VNS(package, pop_size=20, n_neighbors=10, max_gen=30,
                      decomposition='tchebycheff', track_metrics=True)
    moead_solutions = moead.run()
    moead_metrics = moead.get_metrics_history()

    print("\nGeneration-wise Hypervolume:")
    print("-"*40)
    print("Gen | NSGA-II HV | MOEA/D HV | Leader")
    print("-"*40)

    nsga2_hv = nsga2_metrics['hypervolume'] if nsga2_metrics else []
    moead_hv = moead_metrics['hypervolume'] if moead_metrics else []

    max_len = max(len(nsga2_hv), len(moead_hv))

    for i in range(0, min(max_len, 30), 5):
        nsga2_val = nsga2_hv[i] if i < len(nsga2_hv) else 0
        moead_val = moead_hv[i] if i < len(moead_hv) else 0

        if nsga2_val > moead_val:
            leader = "NSGA-II"
        elif moead_val > nsga2_val:
            leader = "MOEA/D"
        else:
            leader = "TIE"

        print(f"{i:3d} | {nsga2_val:10.4f} | {moead_val:9.4f} | {leader}")


if __name__ == '__main__':
    test_moead_with_hypervolume()
    compare_algorithms_with_hypervolume()
    plot_hypervolume_evolution()

    print("\n" + "="*70)
    print("HYPERVOLUME TESTING COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("1. Hypervolume is a reliable single metric for comparison")
    print("2. Higher hypervolume = better convergence AND diversity")
    print("3. Both algorithms can now be objectively compared")
    print("4. No reference set needed (unlike IGD+)")
    print("\nUsage: python -m src.optimizer.moead_vns fastapi --metrics")