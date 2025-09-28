"""
Test Semantic Improvements vs Original Method
"""

import sys
import numpy as np
import pickle

sys.path.append('temp')
from semantic_improvements_implementation import SemanticEnhancedRecommender

def test_package(package_name, expected_packages):
    """
    Compare semantic vs original methods
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {package_name.upper()}")
    print(f"Expected: {expected_packages}")
    print(f"{'='*80}")

    try:
        # Initialize semantic recommender
        recommender = SemanticEnhancedRecommender(package_name, pop_size=10)

        # Test different initialization strategies
        strategies = ['random', 'cluster_based', 'similarity_based', 'co_occurrence_based', 'multi_modal']
        results = {}

        for strategy in strategies:
            # Generate solutions
            solutions = []
            for _ in range(5):
                solution = recommender.smart_initialization(strategy=strategy)

                # Get package names
                packages = [recommender.package_names[idx] for idx in solution[:10]]

                # Check matches
                matches = [p for p in expected_packages if p in packages]
                success_rate = len(matches) / len(expected_packages) * 100

                solutions.append({
                    'packages': packages,
                    'matches': matches,
                    'success_rate': success_rate
                })

            # Average success rate
            avg_success = np.mean([s['success_rate'] for s in solutions])
            best_solution = max(solutions, key=lambda x: x['success_rate'])

            results[strategy] = {
                'avg_success': avg_success,
                'best_matches': best_solution['matches'],
                'best_packages': best_solution['packages'][:5]
            }

            print(f"\n{strategy.upper()}:")
            print(f"  Average success: {avg_success:.1f}%")
            print(f"  Best matches: {best_solution['matches']}")
            print(f"  Sample packages: {best_solution['packages'][:5]}")

        # Find winner
        winner = max(results.items(), key=lambda x: x[1]['avg_success'])
        print(f"\n🏆 WINNER: {winner[0].upper()} with {winner[1]['avg_success']:.1f}% success rate")

        return results

    except Exception as e:
        print(f"Error testing {package_name}: {e}")
        return None

def main():
    """
    Run comprehensive semantic improvement tests
    """
    print("="*80)
    print("SEMANTIC IMPROVEMENTS TEST")
    print("="*80)

    test_cases = {
        'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
        'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
        'pandas': ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl']
    }

    all_results = {}
    for package, expected in test_cases.items():
        results = test_package(package, expected)
        if results:
            all_results[package] = results

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)

    for package, results in all_results.items():
        print(f"\n{package.upper()}:")
        for strategy, data in results.items():
            print(f"  {strategy}: {data['avg_success']:.1f}%")

    # Overall winner
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)

    # Calculate average improvements
    semantic_strategies = ['cluster_based', 'similarity_based', 'co_occurrence_based', 'multi_modal']

    for package in all_results:
        random_score = all_results[package]['random']['avg_success']
        best_semantic = max([all_results[package][s]['avg_success'] for s in semantic_strategies])
        improvement = (best_semantic / max(random_score, 0.1)) if random_score > 0 else float('inf')

        print(f"\n{package.upper()}:")
        print(f"  Random: {random_score:.1f}%")
        print(f"  Best Semantic: {best_semantic:.1f}%")
        print(f"  Improvement: {improvement:.1f}x")

if __name__ == '__main__':
    main()