"""
Test MOEA/D convergence with normalization
Verify that normalization fixes the negative convergence issue
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('pycommend-code/src')
os.chdir('pycommend-code')


def test_moead_convergence(version='normalized', iterations=30, verbose=True):
    """
    Test MOEA/D convergence with different versions
    """
    if version == 'normalized':
        from optimizer.moead_normalized import MOEAD_Normalized
        moead = MOEAD_Normalized('fastapi', pop_size=50, max_gen=iterations, track_metrics=True)
        version_name = "MOEA/D Normalized"
    elif version == 'final':
        from optimizer.moead_final import MOEAD_Final
        moead = MOEAD_Final('fastapi', pop_size=50, max_gen=iterations, track_metrics=True)
        version_name = "MOEA/D Final (no norm)"
    else:
        from optimizer.moead import MOEAD
        moead = MOEAD('fastapi', pop_size=50, max_gen=iterations, track_metrics=True)
        version_name = "MOEA/D Original"

    print(f"\nTesting {version_name}...")
    print("="*60)

    solutions = moead.run()
    metrics = moead.get_metrics_history()

    results = {
        'version': version_name,
        'solutions': len(solutions),
        'converges': False,
        'improvement': 0,
        'hv_history': []
    }

    if metrics and 'hypervolume' in metrics and metrics['hypervolume']:
        hv = metrics['hypervolume']
        results['hv_history'] = hv

        if len(hv) > 1:
            initial = hv[0] if hv[0] > 0 else 0.001
            final = hv[-1]
            improvement = (final - initial) / initial * 100
            results['improvement'] = improvement
            results['converges'] = improvement > 0

            if verbose:
                print(f"\nResults for {version_name}:")
                print(f"  Initial HV: {hv[0]:.6f}")
                print(f"  Final HV: {hv[-1]:.6f}")
                print(f"  Improvement: {improvement:+.1f}%")

                monotonic = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
                monotonic_rate = monotonic / (len(hv) - 1) * 100
                print(f"  Monotonic steps: {monotonic}/{len(hv)-1} ({monotonic_rate:.1f}%)")

                if improvement > 0:
                    print(f"  STATUS: CONVERGING")
                else:
                    print(f"  STATUS: DIVERGING")

    return results


def compare_versions():
    """
    Compare normalized vs non-normalized versions
    """
    print("CONVERGENCE COMPARISON TEST")
    print("="*60)

    results = []

    normalized_result = test_moead_convergence('normalized', iterations=20)
    results.append(normalized_result)

    original_result = test_moead_convergence('final', iterations=20)
    results.append(original_result)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for result in results:
        if result['hv_history']:
            hv = result['hv_history']
            label = f"{result['version']} ({result['improvement']:+.0f}%)"
            color = 'green' if result['converges'] else 'red'
            plt.plot(range(len(hv)), hv, label=label, linewidth=2, color=color)

    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.title('Convergence Comparison: Normalized vs Original')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    versions = [r['version'].replace(' ', '\n') for r in results]
    improvements = [r['improvement'] for r in results]
    colors = ['green' if r['converges'] else 'red' for r in results]

    bars = plt.bar(range(len(versions)), improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Version')
    plt.ylabel('HV Improvement (%)')
    plt.title('Final Improvement Comparison')
    plt.xticks(range(len(versions)), versions)
    plt.grid(True, alpha=0.3, axis='y')

    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -10),
                f'{imp:+.0f}%', ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig('../normalized_convergence_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def test_movns_comparison():
    """
    Also test MOVNS for reference
    """
    print("\n" + "="*60)
    print("REFERENCE: Testing MOVNS convergence...")

    from optimizer.movns_vns import MOVNS_VNS
    movns = MOVNS_VNS('fastapi', archive_size=50, max_iterations=20, track_metrics=True)

    solutions = movns.run()
    metrics = movns.get_metrics_history()

    if metrics and 'hypervolume' in metrics:
        hv = metrics['hypervolume']
        if len(hv) > 1:
            improvement = (hv[-1] - hv[0]) / (hv[0] + 1e-10) * 100
            print(f"\nMOVNS Results:")
            print(f"  Initial HV: {hv[0]:.6f}")
            print(f"  Final HV: {hv[-1]:.6f}")
            print(f"  Improvement: {improvement:+.1f}%")
            print(f"  STATUS: {'CONVERGING' if improvement > 0 else 'DIVERGING'}")


def main():
    """
    Main test function
    """
    print("NORMALIZATION FIX VALIDATION")
    print("="*60)

    results = compare_versions()

    test_movns_comparison()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for result in results:
        status = "PASS" if result['converges'] else "FAIL"
        print(f"{result['version']:30s}: {status:4s} ({result['improvement']:+.1f}%)")

    if any(r['converges'] and 'Normalized' in r['version'] for r in results):
        print("\nSUCCESS: Normalization fixes convergence issue!")
    else:
        print("\nWARNING: Normalization did not fix convergence")

    print(f"\nPlot saved to: normalized_convergence_test.png")


if __name__ == "__main__":
    main()