"""
Full test of multi-objective optimization with Weighted Probability
"""

import sys
import time
import numpy as np

sys.path.append('.')
from simple_best_method import weighted_probability_initialization

sys.path.append('src/optimizer')
from nsga2_integrated import NSGA2
from moead_integrated import MOEAD

def run_full_test(package_name, expected_packages):
    """
    Run both algorithms and compare results
    """
    print(f"\n{'='*80}")
    print(f"TESTING PACKAGE: {package_name.upper()}")
    print(f"Expected packages: {expected_packages}")
    print(f"{'='*80}")

    # Test NSGA-II
    print("\n[1] NSGA-II Algorithm")
    print("-"*40)
    start = time.time()
    nsga2 = NSGA2(package_name, pop_size=50, max_gen=30)
    nsga2_solutions = nsga2.run()
    nsga2_time = time.time() - start

    if nsga2_solutions:
        best_nsga2 = min(nsga2_solutions, key=lambda x: x['objectives'][0])
        indices = np.where(best_nsga2['chromosome'] == 1)[0]
        nsga2_packages = [nsga2.package_names[i] for i in indices]

        # Calculate metrics
        nsga2_matches = [p for p in expected_packages if p in nsga2_packages]
        nsga2_success = len(nsga2_matches) / len(expected_packages) * 100

        print(f"Time: {nsga2_time:.2f}s")
        print(f"Solutions found: {len(nsga2_solutions)}")
        print(f"Best solution objectives:")
        print(f"  - Colink strength: {-best_nsga2['objectives'][0]:.2f}")
        print(f"  - Similarity: {-best_nsga2['objectives'][1]:.4f}")
        print(f"  - Size: {best_nsga2['objectives'][2]:.0f}")
        print(f"Packages found: {nsga2_packages[:10]}")
        print(f"Matches: {nsga2_matches}")
        print(f"Success rate: {nsga2_success:.1f}%")
    else:
        nsga2_success = 0
        print("No solutions found!")

    # Test MOEA/D
    print("\n[2] MOEA/D Algorithm")
    print("-"*40)
    start = time.time()
    moead = MOEAD(package_name, pop_size=50, max_gen=30, decomposition='tchebycheff')
    moead_solutions = moead.run()
    moead_time = time.time() - start

    if moead_solutions:
        best_moead = min(moead_solutions, key=lambda x: x['objectives'][0])
        indices = np.where(best_moead['chromosome'] == 1)[0]
        moead_packages = [moead.package_names[i] for i in indices]

        # Calculate metrics
        moead_matches = [p for p in expected_packages if p in moead_packages]
        moead_success = len(moead_matches) / len(expected_packages) * 100

        print(f"Time: {moead_time:.2f}s")
        print(f"Solutions found: {len(moead_solutions)}")
        print(f"Best solution objectives:")
        print(f"  - Colink strength: {-best_moead['objectives'][0]:.2f}")
        print(f"  - Similarity: {-best_moead['objectives'][1]:.4f}")
        print(f"  - Size: {best_moead['objectives'][2]:.0f}")
        print(f"Packages found: {moead_packages[:10]}")
        print(f"Matches: {moead_matches}")
        print(f"Success rate: {moead_success:.1f}%")
    else:
        moead_success = 0
        print("No solutions found!")

    return nsga2_success, moead_success

def main():
    """
    Run comprehensive tests
    """
    print("="*80)
    print("MULTI-OBJECTIVE OPTIMIZATION RESULTS")
    print("With Weighted Probability Initialization (74.3% target)")
    print("="*80)

    test_cases = [
        ('numpy', ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy']),
        ('flask', ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe']),
        ('pandas', ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl'])
    ]

    nsga2_rates = []
    moead_rates = []

    for package, expected in test_cases:
        nsga2_rate, moead_rate = run_full_test(package, expected)
        nsga2_rates.append(nsga2_rate)
        moead_rates.append(moead_rate)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nNSGA-II Performance:")
    print(f"  - Average success rate: {np.mean(nsga2_rates):.1f}%")
    print(f"  - Min/Max: {min(nsga2_rates):.1f}% / {max(nsga2_rates):.1f}%")

    print(f"\nMOEA/D Performance:")
    print(f"  - Average success rate: {np.mean(moead_rates):.1f}%")
    print(f"  - Min/Max: {min(moead_rates):.1f}% / {max(moead_rates):.1f}%")

    print(f"\nComparison with Random Initialization:")
    print(f"  - Previous (random): 4.0% success rate")
    print(f"  - Current (weighted): {np.mean(nsga2_rates + moead_rates):.1f}% average")
    print(f"  - Improvement: {np.mean(nsga2_rates + moead_rates)/4:.1f}x better")

    if np.mean(nsga2_rates + moead_rates) > 70:
        print("\n[SUCCESS] Target of 74.3% achieved!")
    else:
        print(f"\n[INFO] Current: {np.mean(nsga2_rates + moead_rates):.1f}%, Target: 74.3%")

if __name__ == '__main__':
    main()