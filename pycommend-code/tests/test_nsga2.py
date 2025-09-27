"""
Unit tests for NSGA2 optimizer
"""

import sys
import os
import unittest
import numpy as np
import pickle
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'optimizer'))


class TestNSGA2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures"""
        cls.data_dir = Path(__file__).parent.parent / 'data'
        cls.rel_matrix_file = cls.data_dir / 'package_relationships_10k.pkl'
        cls.sim_matrix_file = cls.data_dir / 'package_similarity_matrix_10k.pkl'

    def setUp(self):
        """Set up test fixtures"""
        # Change to pycommend-code directory for data access
        self.original_dir = os.getcwd()
        os.chdir(Path(__file__).parent.parent)

    def tearDown(self):
        """Restore original directory"""
        os.chdir(self.original_dir)

    def test_nsga2_import(self):
        """Test that NSGA2 can be imported"""
        try:
            from nsga2 import NSGA2
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import NSGA2: {e}")

    def test_nsga2_initialization(self):
        """Test NSGA2 initialization with valid package"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Try to initialize with a common package
        try:
            nsga = NSGA2('numpy', pop_size=10, max_gen=5)
            self.assertIsNotNone(nsga)
            self.assertEqual(nsga.package_name, 'numpy')
            self.assertEqual(nsga.pop_size, 10)
            self.assertEqual(nsga.max_gen, 5)
        except ValueError as e:
            # If numpy is not in the dataset, try another common package
            try:
                nsga = NSGA2('requests', pop_size=10, max_gen=5)
                self.assertIsNotNone(nsga)
            except ValueError:
                self.skipTest("No common packages found in dataset")

    def test_nsga2_invalid_package(self):
        """Test NSGA2 initialization with invalid package"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Should raise ValueError for non-existent package
        with self.assertRaises(ValueError):
            NSGA2('nonexistent_package_xyz123', pop_size=10, max_gen=5)

    def test_dominates_function(self):
        """Test the dominates function logic"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package for initialization
        with open(self.rel_matrix_file, 'rb') as f:
            data = pickle.load(f)
            first_package = data['package_names'][0]

        nsga = NSGA2(first_package, pop_size=10, max_gen=5)

        # Test domination logic (minimization)
        # obj1 dominates obj2 if all objectives are <= and at least one is <
        obj1 = [1.0, 2.0, 3.0]
        obj2 = [2.0, 2.0, 4.0]
        self.assertTrue(nsga.dominates(obj1, obj2))

        # obj1 does not dominate obj2 (equal)
        obj1 = [1.0, 2.0, 3.0]
        obj2 = [1.0, 2.0, 3.0]
        self.assertFalse(nsga.dominates(obj1, obj2))

        # obj1 does not dominate obj2 (worse in one objective)
        obj1 = [1.0, 3.0, 3.0]
        obj2 = [2.0, 2.0, 3.0]
        self.assertFalse(nsga.dominates(obj1, obj2))

    def test_create_individual(self):
        """Test individual creation"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package
        with open(self.rel_matrix_file, 'rb') as f:
            data = pickle.load(f)
            first_package = data['package_names'][0]

        nsga = NSGA2(first_package, pop_size=10, max_gen=5)

        # Create individual
        individual = nsga.create_individual()

        # Check structure
        self.assertIn('chromosome', individual)
        self.assertIn('objectives', individual)
        self.assertIn('rank', individual)
        self.assertIn('crowding_distance', individual)

        # Check chromosome properties
        chromosome = individual['chromosome']
        self.assertEqual(len(chromosome), nsga.n_packages)
        self.assertIn(chromosome.dtype, [np.int8, np.int16, np.int32, np.int64])

        # Check that package itself is not selected
        pkg_idx = nsga.pkg_to_idx[nsga.package_name]
        self.assertEqual(chromosome[pkg_idx], 0)

        # Check size constraints
        selected = np.sum(chromosome)
        self.assertGreaterEqual(selected, nsga.min_size)
        self.assertLessEqual(selected, nsga.max_size)

    def test_evaluate_objectives(self):
        """Test objective evaluation"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package
        with open(self.rel_matrix_file, 'rb') as f:
            data = pickle.load(f)
            first_package = data['package_names'][0]

        nsga = NSGA2(first_package, pop_size=10, max_gen=5)

        # Create a test chromosome
        chromosome = np.zeros(nsga.n_packages, dtype=np.int8)

        # Select a few packages
        selected_indices = [i for i in range(min(5, nsga.n_packages))
                          if i != nsga.pkg_to_idx[nsga.package_name]][:3]
        chromosome[selected_indices] = 1

        # Evaluate objectives
        objectives = nsga.evaluate_objectives(chromosome)

        # Check that we get 3 objectives
        self.assertEqual(len(objectives), 3)

        # F3 should equal the number of selected packages
        self.assertEqual(objectives[2], len(selected_indices))

    def test_repair_chromosome(self):
        """Test chromosome repair mechanism"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package
        with open(self.rel_matrix_file, 'rb') as f:
            data = pickle.load(f)
            first_package = data['package_names'][0]

        nsga = NSGA2(first_package, pop_size=10, max_gen=5)

        # Test repair for too small chromosome
        small_chromosome = np.zeros(nsga.n_packages, dtype=np.int8)
        small_chromosome[0] = 1  # Only one selected

        repaired = nsga.repair_chromosome(small_chromosome)
        selected = np.sum(repaired)
        self.assertGreaterEqual(selected, nsga.min_size)

        # Test repair for too large chromosome
        large_chromosome = np.ones(nsga.n_packages, dtype=np.int8)
        large_chromosome[nsga.pkg_to_idx[nsga.package_name]] = 0

        repaired = nsga.repair_chromosome(large_chromosome)
        selected = np.sum(repaired)
        self.assertLessEqual(selected, nsga.max_size)

    def test_fast_non_dominated_sort(self):
        """Test non-dominated sorting"""
        if not (self.rel_matrix_file.exists() and self.sim_matrix_file.exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package
        with open(self.rel_matrix_file, 'rb') as f:
            data = pickle.load(f)
            first_package = data['package_names'][0]

        nsga = NSGA2(first_package, pop_size=10, max_gen=5)

        # Create a small population with known domination relationships
        population = [
            {'objectives': [1, 1, 1], 'rank': None},  # Best (front 0)
            {'objectives': [2, 2, 2], 'rank': None},  # Dominated (front 1)
            {'objectives': [1, 2, 1], 'rank': None},  # Mixed (front 0 or 1)
            {'objectives': [3, 3, 3], 'rank': None},  # Worst (front 2)
        ]

        fronts = nsga.fast_non_dominated_sort(population)

        # Check that we have fronts
        self.assertGreater(len(fronts), 0)

        # Check that all individuals are assigned
        all_individuals = []
        for front in fronts:
            all_individuals.extend(front)
        self.assertEqual(len(set(all_individuals)), len(population))

        # Check rank assignment
        for i, ind in enumerate(population):
            self.assertIsNotNone(ind['rank'])


class TestNSGA2Integration(unittest.TestCase):
    """Integration tests for NSGA2"""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures"""
        cls.data_dir = Path(__file__).parent.parent / 'data'
        cls.temp_dir = None

    def setUp(self):
        """Set up test fixtures"""
        self.original_dir = os.getcwd()
        os.chdir(Path(__file__).parent.parent)
        # Create temporary results directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        os.chdir(self.original_dir)
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_nsga2_small_run(self):
        """Test a small complete run of NSGA2"""
        if not (Path('data/package_relationships_10k.pkl').exists() and
                Path('data/package_similarity_matrix_10k.pkl').exists()):
            self.skipTest("Required data files not found")

        from nsga2 import NSGA2

        # Get first available package
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            data = pickle.load(f)
            test_package = data['package_names'][0]

        # Run with very small parameters for speed
        nsga = NSGA2(test_package, pop_size=4, max_gen=2)

        # Run the algorithm
        final_front = nsga.run()

        # Check that we got results
        self.assertIsNotNone(final_front)
        self.assertIsInstance(final_front, list)
        self.assertGreater(len(final_front), 0)

        # Check structure of results
        for solution in final_front:
            self.assertIn('chromosome', solution)
            self.assertIn('objectives', solution)
            self.assertIn('rank', solution)
            self.assertEqual(solution['rank'], 0)  # All should be rank 0 (Pareto front)


def run_tests():
    """Run all NSGA2 tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNSGA2))
    suite.addTests(loader.loadTestsFromTestCase(TestNSGA2Integration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Print summary
    print("\n" + "="*70)
    print("NSGA2 TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All NSGA2 tests passed!")
    else:
        print("\n✗ Some tests failed. See details above.")

    sys.exit(0 if result.wasSuccessful() else 1)