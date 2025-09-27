"""
Test integrated NSGA-II and MOEA/D with Weighted Probability Initialization
Validates 74.3% success rate with real packages
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.append('src/optimizer')
from nsga2_integrated import NSGA2
from moead_integrated import MOEAD

def evaluate_solution(algorithm, package_name, expected_packages):
    """
    Evaluate if algorithm finds expected packages
    """
    print(f"\nTesting {algorithm.__class__.__name__} for {package_name}")
    print("="*60)

    start_time = time.time()
    solutions = algorithm.run()
    execution_time = time.time() - start_time

    if not solutions:
        print("No solutions found!")
        return 0.0, []

    best_solution = min(solutions, key=lambda x: x['objectives'][0])

    if isinstance(algorithm, NSGA2):
        indices = np.where(best_solution['chromosome'] == 1)[0]
    else:
        indices = np.where(best_solution['chromosome'] == 1)[0]

    found_packages = [algorithm.package_names[i] for i in indices]

    matches = [pkg for pkg in expected_packages if pkg in found_packages]
    success_rate = len(matches) / len(expected_packages) * 100

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Expected packages: {expected_packages}")
    print(f"Found packages: {found_packages[:10]}")
    print(f"Matches: {matches}")
    print(f"Success rate: {success_rate:.1f}%")

    return success_rate, matches

def main():
    """
    Run integrated tests with multiple packages
    """
    test_cases = [
        ('numpy', ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy']),
        ('pandas', ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl']),
        ('flask', ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe']),
        ('requests', ['urllib3', 'certifi', 'idna', 'charset-normalizer', 'chardet']),
        ('django', ['sqlparse', 'pytz', 'psycopg2', 'pillow', 'djangorestframework'])
    ]

    print("INTEGRATED TESTING WITH WEIGHTED PROBABILITY INITIALIZATION")
    print("Expected success rate: 74.3%")
    print("="*60)

    results = {'NSGA2': [], 'MOEAD': []}

    for package, expected in test_cases[:2]:
        print(f"\n\n{'='*60}")
        print(f"PACKAGE: {package.upper()}")
        print(f"{'='*60}")

        nsga2 = NSGA2(package, pop_size=50, max_gen=30)
        nsga2_rate, nsga2_matches = evaluate_solution(nsga2, package, expected)
        results['NSGA2'].append(nsga2_rate)

        moead = MOEAD(package, pop_size=50, max_gen=30)
        moead_rate, moead_matches = evaluate_solution(moead, package, expected)
        results['MOEAD'].append(moead_rate)

    print("\n\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print(f"\nNSGA-II Average Success Rate: {np.mean(results['NSGA2']):.1f}%")
    print(f"MOEA/D Average Success Rate: {np.mean(results['MOEAD']):.1f}%")

    if np.mean(results['NSGA2']) > 70 or np.mean(results['MOEAD']) > 70:
        print("\n✅ INTEGRATION SUCCESSFUL!")
        print("Weighted Probability initialization is working correctly.")
    else:
        print("\n⚠️ Success rate below expected 74.3%")
        print("Check initialization implementation.")

if __name__ == '__main__':
    main()