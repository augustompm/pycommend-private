"""
Unit tests for initialization methods with real PyCommend data
Testing which method from 2023-2024 research performs best
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import random
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)


class SimpleInitializationMethods:
    """Simplified initialization methods based on 2023-2024 research"""

    def __init__(self, rel_matrix, main_package_idx: int, package_names: List[str]):
        self.rel_matrix = rel_matrix
        self.main_idx = main_package_idx
        self.package_names = package_names
        self.n_packages = len(package_names)

        # Pre-compute connections
        self.connections = rel_matrix[main_package_idx].toarray().flatten()
        self.ranked_indices = np.argsort(self.connections)[::-1]

        # Remove zero connections
        non_zero_mask = self.connections[self.ranked_indices] > 0
        self.valid_candidates = self.ranked_indices[non_zero_mask]

        print(f"\nInitializing for package: {package_names[main_package_idx]}")
        print(f"Total packages with connections: {len(self.valid_candidates)}")
        print(f"Top 10 connected packages:")
        for i in range(min(10, len(self.valid_candidates))):
            idx = self.valid_candidates[i]
            print(f"  {i+1}. {package_names[idx]}: {self.connections[idx]} co-occurrences")

    def method1_random_baseline(self, pop_size: int = 20) -> List[np.ndarray]:
        """
        BASELINE: Pure random initialization (traditional approach)
        """
        population = []
        for _ in range(pop_size):
            size = random.randint(3, 10)
            individual = np.random.choice(self.n_packages, size, replace=False)
            population.append(individual)
        return population

    def method2_top_k_pool(self, pop_size: int = 20, k: int = 100) -> List[np.ndarray]:
        """
        Method from NSGA-II/SDR-OLS (Zhang et al., 2023)
        Source: Mathematics MDPI, vol. 11(8), April 2023
        Link: https://ideas.repec.org/a/gam/jmathe/v11y2023i8p1911-d1126279.html

        Uses top-K strongest connections as candidate pool
        """
        population = []
        pool = self.valid_candidates[:min(k, len(self.valid_candidates))]

        if len(pool) == 0:
            return self.method1_random_baseline(pop_size)

        for _ in range(pop_size):
            size = random.randint(3, 10)
            size = min(size, len(pool))
            individual = np.random.choice(pool, size, replace=False)
            population.append(individual)
        return population

    def method3_weighted_probability(self, pop_size: int = 20, k: int = 100) -> List[np.ndarray]:
        """
        Weighted sampling based on connection strength
        Common in 2023-2024 hybrid approaches
        """
        population = []

        if len(self.valid_candidates) == 0:
            return self.method1_random_baseline(pop_size)

        # Use top-K candidates with weighted probability
        pool = self.valid_candidates[:min(k, len(self.valid_candidates))]
        weights = self.connections[pool].astype(float)
        weights = weights / weights.sum()

        for _ in range(pop_size):
            size = random.randint(3, 10)
            size = min(size, len(pool))
            individual = np.random.choice(pool, size, replace=False, p=weights)
            population.append(individual)
        return population

    def method4_tiered_selection(self, pop_size: int = 20) -> List[np.ndarray]:
        """
        Tiered selection based on connection strength percentiles
        Inspired by multi-tier approaches in recent MOEA research
        """
        population = []

        if len(self.valid_candidates) == 0:
            return self.method1_random_baseline(pop_size)

        # Create tiers
        n_valid = len(self.valid_candidates)
        tier1 = self.valid_candidates[:min(20, n_valid)]     # Top 20
        tier2 = self.valid_candidates[:min(50, n_valid)]     # Top 50
        tier3 = self.valid_candidates[:min(100, n_valid)]    # Top 100

        for i in range(pop_size):
            # Progressive tier selection
            if i < pop_size * 0.4:  # 40% from tier 1
                pool = tier1
            elif i < pop_size * 0.7:  # 30% from tier 2
                pool = tier2
            else:  # 30% from tier 3
                pool = tier3

            size = random.randint(3, 10)
            size = min(size, len(pool))
            individual = np.random.choice(pool, size, replace=False)
            population.append(individual)

        return population

    def method5_opposition_based(self, pop_size: int = 20) -> List[np.ndarray]:
        """
        Opposition-Based Learning (OBL) initialization
        From NSGA-II/SDR-OLS (Zhang et al., 2023)
        Creates opposition individuals from strong connections
        """
        population = []

        if len(self.valid_candidates) < 20:
            return self.method1_random_baseline(pop_size)

        # First half: top connections
        half_size = pop_size // 2
        for _ in range(half_size):
            size = random.randint(3, 10)
            pool = self.valid_candidates[:50]
            size = min(size, len(pool))
            individual = np.random.choice(pool, size, replace=False)
            population.append(individual)

        # Second half: opposition (exclude top connections)
        opposition_pool = self.valid_candidates[20:min(200, len(self.valid_candidates))]
        for _ in range(pop_size - half_size):
            size = random.randint(3, 10)
            size = min(size, len(opposition_pool))
            individual = np.random.choice(opposition_pool, size, replace=False)
            population.append(individual)

        return population


def evaluate_population(population: List[np.ndarray], connections: np.ndarray,
                        package_names: List[str], main_package_name: str) -> Dict[str, Any]:
    """Evaluate quality of initialized population"""

    # Known good packages for common test cases
    expected_packages = {
        'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy', 'pytest', 'pillow'],
        'pandas': ['numpy', 'matplotlib', 'scipy', 'seaborn', 'openpyxl', 'pytest', 'scikit-learn'],
        'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe', 'pytest'],
        'django': ['pillow', 'psycopg2', 'gunicorn', 'celery', 'pytest', 'requests'],
        'requests': ['urllib3', 'certifi', 'chardet', 'idna', 'pytest']
    }

    # Get expected packages for this main package
    expected = expected_packages.get(main_package_name, [])

    # Metrics
    total_connection_strength = []
    expected_found = set()
    unique_packages = set()

    for individual in population:
        # Connection strength
        ind_strength = sum(connections[idx] for idx in individual)
        total_connection_strength.append(ind_strength)

        # Check for expected packages
        for idx in individual:
            pkg_name = package_names[idx]
            unique_packages.add(pkg_name)
            if pkg_name in expected:
                expected_found.add(pkg_name)

    # Calculate metrics
    avg_strength = np.mean(total_connection_strength)
    max_strength = np.max(total_connection_strength)
    coverage = len(expected_found) / len(expected) * 100 if expected else 0

    return {
        'avg_connection_strength': avg_strength,
        'max_connection_strength': max_strength,
        'expected_coverage': coverage,
        'expected_found': list(expected_found),
        'diversity': len(unique_packages),
        'top_packages': list(unique_packages)[:20]
    }


def run_tests():
    """Run tests with real PyCommend data"""

    # Load real data
    print("Loading PyCommend data...")
    with open('E:/pycommend/pycommend-code/data/package_relationships_10k.pkl', 'rb') as f:
        data = pickle.load(f)

    rel_matrix = data['matrix']
    package_names = data['package_names']

    # Test packages
    test_packages = ['numpy', 'pandas', 'flask', 'requests', 'scikit-learn']

    results = {}

    for test_pkg in test_packages:
        if test_pkg not in package_names:
            print(f"\nSkipping {test_pkg} - not in dataset")
            continue

        main_idx = package_names.index(test_pkg)

        print(f"\n{'='*60}")
        print(f"Testing initialization methods for: {test_pkg}")
        print(f"{'='*60}")

        # Initialize methods
        init = SimpleInitializationMethods(rel_matrix, main_idx, package_names)

        # Test each method
        methods = [
            ('Random (Baseline)', init.method1_random_baseline),
            ('Top-K Pool (Zhang 2023)', init.method2_top_k_pool),
            ('Weighted Probability', init.method3_weighted_probability),
            ('Tiered Selection', init.method4_tiered_selection),
            ('Opposition-Based (Zhang 2023)', init.method5_opposition_based)
        ]

        method_results = {}

        for method_name, method_func in methods:
            print(f"\nTesting: {method_name}")

            # Generate population
            population = method_func(pop_size=20)

            # Evaluate
            eval_results = evaluate_population(
                population,
                init.connections,
                package_names,
                test_pkg
            )

            method_results[method_name] = eval_results

            print(f"  Avg connection strength: {eval_results['avg_connection_strength']:.1f}")
            print(f"  Max connection strength: {eval_results['max_connection_strength']:.1f}")
            print(f"  Expected packages found: {eval_results['expected_coverage']:.1f}%")
            print(f"  Found: {eval_results['expected_found']}")
            print(f"  Diversity: {eval_results['diversity']} unique packages")

        results[test_pkg] = method_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Best Methods by Metric")
    print(f"{'='*60}")

    # Aggregate scores
    method_scores = {}
    for pkg, pkg_results in results.items():
        for method, metrics in pkg_results.items():
            if method not in method_scores:
                method_scores[method] = {
                    'avg_strength': [],
                    'coverage': [],
                    'diversity': []
                }
            method_scores[method]['avg_strength'].append(metrics['avg_connection_strength'])
            method_scores[method]['coverage'].append(metrics['expected_coverage'])
            method_scores[method]['diversity'].append(metrics['diversity'])

    print("\nAverage Performance Across All Test Packages:")
    print(f"{'Method':<30} {'Avg Strength':<15} {'Coverage %':<15} {'Diversity':<10}")
    print("-" * 70)

    for method, scores in method_scores.items():
        avg_str = np.mean(scores['avg_strength'])
        avg_cov = np.mean(scores['coverage'])
        avg_div = np.mean(scores['diversity'])
        print(f"{method:<30} {avg_str:<15.1f} {avg_cov:<15.1f} {avg_div:<10.1f}")

    # Winner
    best_method = max(method_scores.items(),
                     key=lambda x: np.mean(x[1]['coverage']) + np.mean(x[1]['avg_strength'])/1000)

    print(f"\n{'='*60}")
    print(f"WINNER: {best_method[0]}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    results = run_tests()

    print("\n" + "="*60)
    print("SOURCES AND REFERENCES")
    print("="*60)
    print("\n1. NSGA-II/SDR-OLS (Zhang et al., 2023)")
    print("   Paper: 'A Novel Large-Scale Many-Objective Optimization Method'")
    print("   Source: Mathematics MDPI, vol. 11(8), April 2023")
    print("   Link: https://ideas.repec.org/a/gam/jmathe/v11y2023i8p1911-d1126279.html")
    print("\n2. Latin Hypercube Sampling methods (2024)")
    print("   Multiple papers from Nature Scientific Reports")
    print("   Link: https://www.nature.com/articles/s41598-024-63739-9")
    print("\n3. Hybrid initialization approaches (2023-2024)")
    print("   Various IEEE and SpringerLink publications")