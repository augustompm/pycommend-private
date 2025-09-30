"""
Test MOEA/D normalization with multiple packages
Verify convergence across different test cases
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir('pycommend-code')


def test_package(package_name, algorithm='normalized', generations=20):
    """
    Test a specific package
    """
    try:
        if algorithm == 'normalized':
            from optimizer.moead_vns_normalized import MOEAD_VNS_Normalized
            moead = MOEAD_VNS_Normalized(
                package_name,
                pop_size=50,
                max_gen=generations,
                n_neighbors=15,
                theta=5.0,
                track_metrics=True
            )
        else:
            from optimizer.moead_vns_final import MOEAD_VNS_Final
            moead = MOEAD_VNS_Final(
                package_name,
                pop_size=50,
                max_gen=generations,
                n_neighbors=15,
                track_metrics=True
            )

        start_time = time.time()
        solutions = moead.run()
        exec_time = time.time() - start_time

        metrics = moead.get_metrics_history()

        if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 1:
            hv = metrics['hypervolume']
            initial = hv[0] if hv[0] > 0 else 0.001
            final = hv[-1]
            improvement = (final - initial) / initial * 100

            return {
                'package': package_name,
                'algorithm': algorithm,
                'initial_hv': initial,
                'final_hv': final,
                'improvement': improvement,
                'n_solutions': len(solutions),
                'exec_time': exec_time,
                'converges': improvement > 0
            }

    except Exception as e:
        return {
            'package': package_name,
            'algorithm': algorithm,
            'error': str(e),
            'converges': False
        }

    return None


def main():
    """
    Test multiple packages
    """
    test_packages = [
        'fastapi',
        'scikit-learn',
        'pandas',
        'numpy',
        'requests',
        'django',
        'flask',
        'pytest',
        'matplotlib',
        'tensorflow'
    ]

    print("MULTI-PACKAGE CONVERGENCE TEST")
    print("="*60)
    print("Testing MOEA/D with normalization on 10 packages")
    print("="*60)

    results_normalized = []
    results_original = []

    for i, package in enumerate(test_packages, 1):
        print(f"\n[{i}/10] Testing {package}...")
        print("-"*40)

        print("  With normalization...")
        result_norm = test_package(package, 'normalized', generations=15)
        if result_norm and 'improvement' in result_norm:
            print(f"    HV: {result_norm['initial_hv']:.4f} -> {result_norm['final_hv']:.4f}")
            print(f"    Improvement: {result_norm['improvement']:+.1f}%")
            print(f"    Status: {'CONVERGES' if result_norm['converges'] else 'DIVERGES'}")
            results_normalized.append(result_norm)

        print("  Without normalization...")
        result_orig = test_package(package, 'original', generations=15)
        if result_orig and 'improvement' in result_orig:
            print(f"    HV: {result_orig['initial_hv']:.4f} -> {result_orig['final_hv']:.4f}")
            print(f"    Improvement: {result_orig['improvement']:+.1f}%")
            print(f"    Status: {'CONVERGES' if result_orig['converges'] else 'DIVERGES'}")
            results_original.append(result_orig)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nWith Normalization:")
    converged = sum(1 for r in results_normalized if r['converges'])
    print(f"  Converged: {converged}/{len(results_normalized)} ({converged/len(results_normalized)*100:.0f}%)")
    avg_improvement = np.mean([r['improvement'] for r in results_normalized])
    print(f"  Avg Improvement: {avg_improvement:+.1f}%")

    print("\nWithout Normalization:")
    converged = sum(1 for r in results_original if r['converges'])
    print(f"  Converged: {converged}/{len(results_original)} ({converged/len(results_original)*100:.0f}%)")
    avg_improvement = np.mean([r['improvement'] for r in results_original])
    print(f"  Avg Improvement: {avg_improvement:+.1f}%")

    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)

    print("\n{:<15} {:^20} {:^20}".format("Package", "Normalized", "Original"))
    print("-"*55)

    for package in test_packages:
        norm_result = next((r for r in results_normalized if r['package'] == package), None)
        orig_result = next((r for r in results_original if r['package'] == package), None)

        norm_str = "N/A"
        if norm_result:
            if 'improvement' in norm_result:
                norm_str = f"{norm_result['improvement']:+.0f}%"
            elif 'error' in norm_result:
                norm_str = "ERROR"

        orig_str = "N/A"
        if orig_result:
            if 'improvement' in orig_result:
                orig_str = f"{orig_result['improvement']:+.0f}%"
            elif 'error' in orig_result:
                orig_str = "ERROR"

        print(f"{package:<15} {norm_str:^20} {orig_str:^20}")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    norm_success = sum(1 for r in results_normalized if r['converges'])
    orig_success = sum(1 for r in results_original if r['converges'])

    if norm_success > orig_success:
        print(f"SUCCESS: Normalization improves convergence!")
        print(f"  Normalized: {norm_success}/{len(results_normalized)} packages converge")
        print(f"  Original: {orig_success}/{len(results_original)} packages converge")
        print(f"  Improvement: +{norm_success - orig_success} packages")
    else:
        print("Results inconclusive")


if __name__ == "__main__":
    main()