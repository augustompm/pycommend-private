"""
Unit tests for NSGA-II v5 with full SBERT integration
"""

import sys
import unittest
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'optimizer'))
from nsga2_v5 import NSGA2_V5


class TestNSGA2V5(unittest.TestCase):
    """Test suite for NSGA-II v5"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_packages = {
            'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
            'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
            'pandas': ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl'],
            'requests': ['urllib3', 'certifi', 'idna', 'charset-normalizer', 'chardet'],
            'django': ['sqlparse', 'pytz', 'psycopg2', 'pillow', 'djangorestframework']
        }

    def test_initialization(self):
        """Test algorithm initialization"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        self.assertEqual(nsga2.n_objectives, 4)
        self.assertEqual(nsga2.package_name, 'numpy')
        self.assertEqual(nsga2.pop_size, 10)
        self.assertIsNotNone(nsga2.embeddings)
        self.assertEqual(nsga2.embeddings.shape[1], 384)
        print("✓ Initialization test passed")

    def test_data_loading(self):
        """Test all data sources are loaded"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        self.assertIsNotNone(nsga2.rel_matrix)
        self.assertIsNotNone(nsga2.sim_matrix)
        self.assertIsNotNone(nsga2.embeddings)

        self.assertEqual(len(nsga2.package_names), nsga2.embeddings.shape[0])
        print(f"✓ Data loading test passed - {len(nsga2.package_names)} packages")

    def test_semantic_components(self):
        """Test semantic clustering and candidate pools"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        self.assertEqual(nsga2.n_clusters, 200)
        self.assertIsNotNone(nsga2.cluster_labels)
        self.assertIsNotNone(nsga2.cooccur_candidates)
        self.assertIsNotNone(nsga2.semantic_candidates)
        self.assertIsNotNone(nsga2.cluster_candidates)

        self.assertGreater(len(nsga2.cooccur_candidates), 0)
        self.assertGreater(len(nsga2.semantic_candidates), 0)
        print(f"✓ Semantic components test passed - Cluster {nsga2.target_cluster}")

    def test_four_objectives(self):
        """Test 4 objectives calculation"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        chromosome = np.zeros(nsga2.n_packages, dtype=np.int8)
        selected = nsga2.semantic_candidates[:5]
        chromosome[selected] = 1

        objectives = nsga2.evaluate_objectives(chromosome)

        self.assertEqual(len(objectives), 4)
        self.assertTrue(all(np.isfinite(objectives)))

        self.assertLess(objectives[0], 0)
        self.assertLess(objectives[1], 0)
        self.assertLess(objectives[2], 0)
        self.assertGreater(objectives[3], 0)

        print(f"✓ Four objectives test passed: {objectives}")

    def test_smart_initialization(self):
        """Test different initialization strategies"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        strategies = ['cooccur', 'semantic', 'cluster', 'hybrid']
        for strategy in strategies:
            chromosome = nsga2.smart_initialization(strategy)

            self.assertEqual(len(chromosome), nsga2.n_packages)
            self.assertGreaterEqual(np.sum(chromosome), nsga2.min_size)
            self.assertLessEqual(np.sum(chromosome), nsga2.max_size)

            print(f"✓ {strategy} initialization: {np.sum(chromosome)} packages")

    def test_population_diversity(self):
        """Test population initialization diversity"""
        nsga2 = NSGA2_V5('numpy', pop_size=20, max_gen=5)
        population = nsga2.initialize_population()

        self.assertEqual(len(population), 20)

        unique_solutions = set()
        for ind in population:
            solution_str = ''.join(map(str, ind['chromosome']))
            unique_solutions.add(solution_str)

        diversity_ratio = len(unique_solutions) / len(population)
        self.assertGreater(diversity_ratio, 0.8)

        print(f"✓ Population diversity test passed: {diversity_ratio:.2%} unique")

    def test_dominance(self):
        """Test dominance calculation for 4 objectives"""
        nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=5)

        obj1 = np.array([-10, -0.8, -0.7, 5])
        obj2 = np.array([-5, -0.6, -0.5, 7])
        obj3 = np.array([-10, -0.8, -0.7, 5])

        self.assertTrue(nsga2.dominates(obj1, obj2))
        self.assertFalse(nsga2.dominates(obj2, obj1))
        self.assertFalse(nsga2.dominates(obj1, obj3))

        print("✓ Dominance test passed for 4 objectives")

    def test_convergence(self):
        """Test algorithm convergence"""
        nsga2 = NSGA2_V5('numpy', pop_size=20, max_gen=10)

        start_time = time.time()
        solutions = nsga2.run()
        execution_time = time.time() - start_time

        self.assertGreater(len(solutions), 0)
        self.assertLess(execution_time, 30)

        best = min(solutions, key=lambda x: x['objectives'][0])
        self.assertEqual(len(best['objectives']), 4)

        print(f"✓ Convergence test passed in {execution_time:.2f}s")
        print(f"  Found {len(solutions)} Pareto solutions")

    def test_package_recommendations(self):
        """Test recommendation quality for known packages"""
        results = {}

        for package, expected in list(self.test_packages.items())[:3]:
            nsga2 = NSGA2_V5(package, pop_size=50, max_gen=20)
            solutions = nsga2.run()

            if solutions:
                best = min(solutions, key=lambda x: x['objectives'][0])
                indices = np.where(best['chromosome'] == 1)[0]
                found = [nsga2.package_names[idx] for idx in indices]

                matches = [pkg for pkg in expected if pkg in found]
                success_rate = len(matches) / len(expected) * 100

                results[package] = {
                    'expected': expected,
                    'found': found[:10],
                    'matches': matches,
                    'success_rate': success_rate
                }

                print(f"✓ {package}: {success_rate:.1f}% success")
                print(f"  Matches: {matches}")

        avg_success = np.mean([r['success_rate'] for r in results.values()])
        self.assertGreater(avg_success, 30)

        print(f"\n✓ Overall success rate: {avg_success:.1f}%")
        return results

    def test_semantic_coherence(self):
        """Test semantic coherence objective (F3)"""
        nsga2 = NSGA2_V5('numpy', pop_size=30, max_gen=15)
        solutions = nsga2.run()

        coherence_values = []
        for sol in solutions[:5]:
            f3_coherence = -sol['objectives'][2]
            coherence_values.append(f3_coherence)

            indices = np.where(sol['chromosome'] == 1)[0]
            packages = [nsga2.package_names[idx] for idx in indices[:5]]

            print(f"✓ Coherence={f3_coherence:.3f} for: {packages}")

        avg_coherence = np.mean(coherence_values)
        self.assertGreater(avg_coherence, 0.3)

        print(f"✓ Average semantic coherence: {avg_coherence:.3f}")


def run_comprehensive_test():
    """Run comprehensive test suite with detailed results"""
    print("="*60)
    print("NSGA-II v5 COMPREHENSIVE TEST SUITE")
    print("="*60)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestNSGA2V5)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY")
        print("NSGA-II v5 is ready for production")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("="*60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_test()