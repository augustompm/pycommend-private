"""
Validation script for NSGA-II v5 with comprehensive testing
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.append('src/optimizer')
from nsga2_v5 import NSGA2_V5


def test_single_package(package_name, expected_packages):
    """Test a single package and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {package_name.upper()}")
    print(f"Expected: {expected_packages}")
    print(f"{'='*60}")

    try:
        nsga2 = NSGA2_V5(package_name, pop_size=50, max_gen=30)

        start_time = time.time()
        solutions = nsga2.run()
        execution_time = time.time() - start_time

        if not solutions:
            print("No solutions found!")
            return 0.0, [], execution_time

        best = min(solutions, key=lambda x: x['objectives'][0])
        indices = np.where(best['chromosome'] == 1)[0]
        found_packages = [nsga2.package_names[idx] for idx in indices]

        matches = [pkg for pkg in expected_packages if pkg in found_packages]
        success_rate = len(matches) / len(expected_packages) * 100

        print(f"\nResults:")
        print(f"  Time: {execution_time:.2f}s")
        print(f"  Pareto solutions: {len(solutions)}")
        print(f"  Solution size: {len(indices)}")
        print(f"  Objectives [F1, F2, F3, F4]: {best['objectives']}")
        print(f"\nPackages found ({len(found_packages)}):")
        for pkg in found_packages[:10]:
            marker = " [MATCH]" if pkg in expected_packages else ""
            print(f"    - {pkg}{marker}")
        print(f"\nMatches: {matches}")
        print(f"Success rate: {success_rate:.1f}%")

        f1_colink = -best['objectives'][0]
        f2_similarity = -best['objectives'][1]
        f3_coherence = -best['objectives'][2]
        f4_size = best['objectives'][3]

        print(f"\nObjective values:")
        print(f"  F1 (Colink strength): {f1_colink:.2f}")
        print(f"  F2 (Similarity): {f2_similarity:.4f}")
        print(f"  F3 (Coherence): {f3_coherence:.4f}")
        print(f"  F4 (Size): {f4_size:.1f}")

        return success_rate, matches, execution_time

    except Exception as e:
        print(f"Error testing {package_name}: {e}")
        return 0.0, [], 0.0


def main():
    """Run comprehensive validation"""
    print("="*80)
    print("NSGA-II v5 VALIDATION - Full SBERT Integration")
    print("Target: >70% Success Rate")
    print("="*80)

    test_cases = {
        'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
        'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
        'pandas': ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl'],
        'requests': ['urllib3', 'certifi', 'idna', 'charset-normalizer', 'chardet'],
        'django': ['sqlparse', 'pytz', 'psycopg2', 'pillow', 'djangorestframework'],
        'tensorflow': ['numpy', 'protobuf', 'tensorboard', 'keras', 'grpcio'],
        'scikit-learn': ['numpy', 'scipy', 'joblib', 'threadpoolctl', 'pandas'],
        'matplotlib': ['numpy', 'pillow', 'cycler', 'pyparsing', 'kiwisolver'],
        'pytest': ['pluggy', 'iniconfig', 'packaging', 'tomli', 'attrs'],
        'fastapi': ['uvicorn', 'pydantic', 'starlette', 'typing-extensions', 'httpx']
    }

    results = {}
    total_time = 0

    for package, expected in list(test_cases.items())[:5]:
        success_rate, matches, exec_time = test_single_package(package, expected)
        results[package] = {
            'success_rate': success_rate,
            'matches': matches,
            'expected': expected,
            'time': exec_time
        }
        total_time += exec_time

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    success_rates = []
    for package, data in results.items():
        success_rates.append(data['success_rate'])
        print(f"\n{package.upper()}:")
        print(f"  Success rate: {data['success_rate']:.1f}%")
        print(f"  Matches: {len(data['matches'])}/{len(data['expected'])}")
        print(f"  Found: {data['matches']}")
        print(f"  Time: {data['time']:.2f}s")

    avg_success = np.mean(success_rates) if success_rates else 0
    min_success = min(success_rates) if success_rates else 0
    max_success = max(success_rates) if success_rates else 0

    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    print(f"Average success rate: {avg_success:.1f}%")
    print(f"Min/Max: {min_success:.1f}% / {max_success:.1f}%")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Average time per package: {total_time/len(results):.2f}s")

    print("\n" + "="*80)
    if avg_success >= 70:
        print("SUCCESS: Target of 70% achieved!")
        print(f"NSGA-II v5 with SBERT: {avg_success:.1f}% success rate")
    elif avg_success >= 50:
        print("GOOD: Significant improvement achieved")
        print(f"NSGA-II v5: {avg_success:.1f}% (target: 70%)")
    else:
        print("NEEDS IMPROVEMENT")
        print(f"Current: {avg_success:.1f}% (target: 70%)")

    print("\nComparison:")
    print(f"  v4 (Weighted Probability): 26.7%")
    print(f"  v5 (Full SBERT): {avg_success:.1f}%")
    print(f"  Improvement: {avg_success/26.7:.1f}x")
    print("="*80)


if __name__ == '__main__':
    main()