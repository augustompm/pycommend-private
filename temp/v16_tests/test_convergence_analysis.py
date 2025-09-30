"""
Test convergence behavior of MOVNS and MOEA/D
Analyze if algorithms are actually converging
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('pycommend-code/src')
from optimizer.movns_vns import MOVNS_VNS
from optimizer.moead_vns import MOEAD_VNS

def analyze_convergence(package='fastapi', iterations=30):
    """
    Run algorithms and analyze convergence patterns
    """
    print(f"Testing convergence for {package}")
    print("="*50)

    print("\n1. Testing MOVNS convergence...")
    movns = MOVNS_VNS(package, archive_size=100, max_iterations=iterations, track_metrics=True)
    movns_solutions = movns.run()
    movns_metrics = movns.get_metrics_history()

    if movns_metrics and 'hypervolume' in movns_metrics:
        hv_history = movns_metrics['hypervolume']
        print(f"MOVNS HV history length: {len(hv_history)}")
        if hv_history:
            print(f"Initial HV: {hv_history[0]:.6f}")
            print(f"Final HV: {hv_history[-1]:.6f}")
            print(f"Improvement: {((hv_history[-1] - hv_history[0]) / (hv_history[0] + 1e-10)) * 100:.2f}%")

            improvements = []
            for i in range(1, len(hv_history)):
                if hv_history[i] > hv_history[i-1]:
                    improvements.append(i)
            print(f"Iterations with improvement: {len(improvements)}/{len(hv_history)-1}")
            print(f"Improvement rate: {len(improvements)/(len(hv_history)-1)*100:.1f}%")

    print("\n2. Testing MOEA/D convergence...")
    moead = MOEAD_VNS(package, pop_size=100, max_gen=iterations, track_metrics=True)
    moead_solutions = moead.run()
    moead_metrics = moead.get_metrics_history()

    if moead_metrics and 'hypervolume' in moead_metrics:
        hv_history = moead_metrics['hypervolume']
        print(f"MOEA/D HV history length: {len(hv_history)}")
        if hv_history:
            print(f"Initial HV: {hv_history[0]:.6f}")
            print(f"Final HV: {hv_history[-1]:.6f}")
            print(f"Improvement: {((hv_history[-1] - hv_history[0]) / (hv_history[0] + 1e-10)) * 100:.2f}%")

            improvements = []
            for i in range(1, len(hv_history)):
                if hv_history[i] > hv_history[i-1]:
                    improvements.append(i)
            print(f"Iterations with improvement: {len(improvements)}/{len(hv_history)-1}")
            print(f"Improvement rate: {len(improvements)/(len(hv_history)-1)*100:.1f}%")

    if movns_metrics and moead_metrics:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if 'hypervolume' in movns_metrics and movns_metrics['hypervolume']:
            plt.plot(movns_metrics['hypervolume'], label='MOVNS', linewidth=2)
        if 'hypervolume' in moead_metrics and moead_metrics['hypervolume']:
            plt.plot(moead_metrics['hypervolume'], label='MOEA/D', linewidth=2)
        plt.xlabel('Iteration/Generation')
        plt.ylabel('Hypervolume')
        plt.title('Convergence Analysis - Raw Data')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        if 'hypervolume' in movns_metrics and movns_metrics['hypervolume']:
            improvements_movns = [0]
            for i in range(1, len(movns_metrics['hypervolume'])):
                if movns_metrics['hypervolume'][i] > movns_metrics['hypervolume'][i-1]:
                    improvements_movns.append(improvements_movns[-1] + 1)
                else:
                    improvements_movns.append(improvements_movns[-1])
            plt.plot(improvements_movns, label='MOVNS cumulative improvements', linewidth=2)

        if 'hypervolume' in moead_metrics and moead_metrics['hypervolume']:
            improvements_moead = [0]
            for i in range(1, len(moead_metrics['hypervolume'])):
                if moead_metrics['hypervolume'][i] > moead_metrics['hypervolume'][i-1]:
                    improvements_moead.append(improvements_moead[-1] + 1)
                else:
                    improvements_moead.append(improvements_moead[-1])
            plt.plot(improvements_moead, label='MOEA/D cumulative improvements', linewidth=2)

        plt.xlabel('Iteration/Generation')
        plt.ylabel('Cumulative Improvements')
        plt.title('Cumulative Improvement Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=150)
        plt.show()

    return movns_metrics, moead_metrics

def diagnose_convergence_issues():
    """
    Diagnose why algorithms might not be converging
    """
    print("\nDIAGNOSING CONVERGENCE ISSUES")
    print("="*50)

    print("\nChecking MOVNS implementation...")

    from optimizer.movns_vns import MOVNS_VNS
    movns = MOVNS_VNS('fastapi', archive_size=100, max_iterations=5, track_metrics=True)

    print(f"Archive limit: {movns.archive_limit}")
    print(f"Max iterations: {movns.max_iterations}")
    print(f"Number of neighborhoods: {movns.k_max}")

    print("\nRunning 5 iterations to check tracking...")
    movns.run()
    metrics = movns.get_metrics_history()

    if metrics:
        for key, values in metrics.items():
            if values:
                print(f"{key}: {len(values)} values recorded")

    print("\nChecking if calculate_metrics is being called...")
    print("Archive size each iteration should trigger metric calculation")

    return metrics

if __name__ == "__main__":
    print("CONVERGENCE ANALYSIS")
    print("="*50)

    movns_metrics, moead_metrics = analyze_convergence('fastapi', iterations=30)

    print("\n" + "="*50)
    print("DIAGNOSTIC CHECK")
    print("="*50)

    diagnose_convergence_issues()

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("Check convergence_analysis.png for visual results")