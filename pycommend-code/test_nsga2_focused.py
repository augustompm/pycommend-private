"""
Focused test for NSGA-II v5 - Testing real performance metrics
"""

import sys
import time
import numpy as np

sys.path.append('src/optimizer')
from nsga2_v5 import NSGA2_V5


def test_single_package(package_name, expected, pop_size=100, max_gen=50):
    """Test a single package thoroughly"""

    print(f"\nTesting {package_name.upper()}")
    print(f"Expected: {expected}")
    print("-"*50)

    nsga2 = NSGA2_V5(package_name, pop_size=pop_size, max_gen=max_gen)

    start = time.time()
    solutions = nsga2.run()
    elapsed = time.time() - start

    if not solutions:
        return 0, [], elapsed

    best = min(solutions, key=lambda x: x['objectives'][0])
    indices = np.where(best['chromosome'] == 1)[0]
    found = [nsga2.package_names[idx] for idx in indices]

    matches = [pkg for pkg in expected if pkg in found]
    success_rate = len(matches) / len(expected) * 100

    print(f"Time: {elapsed:.2f}s")
    print(f"Pareto size: {len(solutions)}")
    print(f"Found packages: {len(found)}")
    print(f"Matches: {matches}")
    print(f"Success rate: {success_rate:.1f}%")

    print(f"\nTop 10 packages found:")
    for i, pkg in enumerate(found[:10]):
        idx = indices[i]
        colink = nsga2.rel_matrix[nsga2.main_package_idx, idx]
        sim = nsga2.sim_matrix[nsga2.main_package_idx, idx]
        marker = " [MATCH]" if pkg in expected else ""
        print(f"  {pkg}: colink={colink:.2f}, sim={sim:.4f}{marker}")

    print(f"\nObjectives of best solution:")
    print(f"  F1 (colink): {-best['objectives'][0]:.2f}")
    print(f"  F2 (similarity): {-best['objectives'][1]:.4f}")
    print(f"  F3 (coherence): {-best['objectives'][2]:.4f}")
    print(f"  F4 (size): {best['objectives'][3]:.1f}")

    return success_rate, matches, elapsed


def main():
    print("="*60)
    print("NSGA-II v5 FOCUSED PERFORMANCE TEST")
    print("="*60)

    test_cases = {
        'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
        'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
        'pandas': ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl']
    }

    results = {}
    total_time = 0

    for package, expected in test_cases.items():
        rate, matches, time_taken = test_single_package(package, expected, 100, 50)
        results[package] = {
            'rate': rate,
            'matches': matches,
            'time': time_taken
        }
        total_time += time_taken

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    rates = []
    for package, data in results.items():
        rates.append(data['rate'])
        print(f"\n{package.upper()}:")
        print(f"  Success rate: {data['rate']:.1f}%")
        print(f"  Matches: {data['matches']}")
        print(f"  Time: {data['time']:.2f}s")

    avg_rate = np.mean(rates)

    print(f"\n{'='*60}")
    print(f"AVERAGE SUCCESS RATE: {avg_rate:.1f}%")
    print(f"TOTAL TIME: {total_time:.2f}s")

    print(f"\nComparison:")
    print(f"  v1 (random): 4.0%")
    print(f"  v4 (weighted): 26.7%")
    print(f"  v5 (SBERT): {avg_rate:.1f}%")

    if avg_rate > 26.7:
        print(f"\nIMPROVEMENT: {avg_rate/26.7:.2f}x over v4")
    else:
        print(f"\nNO IMPROVEMENT over v4")

    print("="*60)


if __name__ == '__main__':
    main()