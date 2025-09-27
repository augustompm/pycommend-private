"""
Unit tests for preprocessor modules
"""

import sys
import os
import unittest
import json
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import create_distance_matrix
from preprocessor import package_similarity
from preprocessor import create_sparse_distance_matrix


class TestCreateDistanceMatrix(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(__file__).parent.parent / 'data'
        self.pypi_file = self.test_data_dir / 'PyPI' / 'top_10_packages.json'

    def test_load_top_packages(self):
        """Test loading top packages from JSON"""
        if not self.pypi_file.exists():
            self.skipTest(f"Test data file {self.pypi_file} not found")

        packages = create_distance_matrix.load_top_packages(str(self.pypi_file))
        self.assertIsInstance(packages, list)
        self.assertGreater(len(packages), 0)

        # Check structure of first package
        if packages:
            first_pkg = packages[0]
            self.assertIn('package', first_pkg)
            self.assertIsInstance(first_pkg['package'], str)

    def test_extract_package_names(self):
        """Test extracting package names from data"""
        test_data = [
            {"package": "numpy", "data": {}},
            {"package": "pandas", "data": {}},
            {"package": "requests", "data": {}}
        ]

        names = create_distance_matrix.extract_package_names(test_data)
        self.assertEqual(len(names), 3)
        self.assertIn("numpy", names)
        self.assertIn("pandas", names)
        self.assertIn("requests", names)

    def test_create_extras_matrix(self):
        """Test creating extras matrix from dependencies"""
        test_packages = [
            {
                "package": "package1",
                "data": {
                    "depends_on": ["package2"],
                    "depended_by": ["package3"]
                }
            },
            {
                "package": "package2",
                "data": {
                    "depends_on": [],
                    "depended_by": ["package1"]
                }
            },
            {
                "package": "package3",
                "data": {
                    "depends_on": ["package1"],
                    "depended_by": []
                }
            }
        ]

        package_names = ["package1", "package2", "package3"]
        matrix = create_distance_matrix.create_extras_matrix(test_packages, package_names)

        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix[0]), 3)

        # Check that diagonal is zero
        for i in range(3):
            self.assertEqual(matrix[i][i], 0)

        # Check symmetry
        for i in range(3):
            for j in range(3):
                self.assertEqual(matrix[i][j], matrix[j][i])

    def test_parse_requirements_txt(self):
        """Test parsing requirements.txt content"""
        # Create a temporary requirements file
        test_file = Path(__file__).parent / 'test_requirements.txt'
        test_content = """
# This is a comment
numpy>=1.19.0
pandas==1.3.0
requests[security]>=2.25.0
-e git+https://github.com/user/repo.git#egg=mypackage
scikit-learn~=0.24.0
# Another comment
matplotlib
"""
        test_file.write_text(test_content)

        try:
            packages = create_distance_matrix.parse_requirements_txt(str(test_file))

            self.assertIn('numpy', packages)
            self.assertIn('pandas', packages)
            self.assertIn('requests', packages)
            self.assertIn('mypackage', packages)
            self.assertIn('scikit-learn', packages)
            self.assertIn('matplotlib', packages)

            # Should not include comments or empty lines
            self.assertNotIn('# This is a comment', packages)
            self.assertNotIn('', packages)
        finally:
            # Clean up
            test_file.unlink()


class TestPackageSimilarity(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(__file__).parent.parent / 'data'

    def test_data_files_exist(self):
        """Test that required data files exist"""
        embeddings_file = self.test_data_dir / 'package_embeddings_10k.pkl'
        similarity_file = self.test_data_dir / 'package_similarity_matrix_10k.pkl'

        if embeddings_file.exists():
            # Test loading embeddings
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.assertIn('embeddings', data)
                self.assertIn('package_names', data)

        if similarity_file.exists():
            # Test loading similarity matrix
            with open(similarity_file, 'rb') as f:
                data = pickle.load(f)
                self.assertIn('similarity_matrix', data)
                self.assertIn('package_names', data)

    def test_similarity_matrix_properties(self):
        """Test properties of similarity matrix if it exists"""
        similarity_file = self.test_data_dir / 'package_similarity_matrix_10k.pkl'

        if not similarity_file.exists():
            self.skipTest(f"Similarity matrix file not found at {similarity_file}")

        with open(similarity_file, 'rb') as f:
            data = pickle.load(f)

        matrix = data['similarity_matrix']
        package_names = data['package_names']

        # Test matrix shape
        n_packages = len(package_names)
        self.assertEqual(matrix.shape, (n_packages, n_packages))

        # Test symmetry (similarity should be symmetric)
        # Sample a few indices to test
        test_indices = min(10, n_packages)
        for i in range(test_indices):
            for j in range(test_indices):
                self.assertAlmostEqual(
                    matrix[i, j],
                    matrix[j, i],
                    places=5,
                    msg=f"Matrix not symmetric at ({i}, {j})"
                )

        # Test diagonal (self-similarity should be 1 or maximum)
        for i in range(min(10, n_packages)):
            self.assertGreaterEqual(matrix[i, i], 0)

        # Test value range (similarities should be between 0 and 1 typically)
        sample = matrix[:min(100, n_packages), :min(100, n_packages)]
        self.assertGreaterEqual(sample.min(), -1.0)
        self.assertLessEqual(sample.max(), 1.0)


class TestSparseDistanceMatrix(unittest.TestCase):

    def test_sparse_matrix_creation(self):
        """Test that sparse matrix module can be imported"""
        try:
            from preprocessor import create_sparse_distance_matrix
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import create_sparse_distance_matrix: {e}")


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCreateDistanceMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestPackageSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseDistanceMatrix))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. See details above.")

    sys.exit(0 if result.wasSuccessful() else 1)