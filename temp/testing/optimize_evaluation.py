"""
Optimized evaluation functions using parallelization and caching
For Ryzen 6 (12 cores) + RTX 16GB
"""

import numpy as np
from functools import lru_cache
from multiprocessing import Pool
import hashlib

class OptimizedEvaluator:
    def __init__(self, rel_matrix, sim_matrix, embeddings, main_package_idx):
        self.rel_matrix = rel_matrix
        self.sim_matrix = sim_matrix
        self.embeddings = embeddings
        self.main_package_idx = main_package_idx
        self.cache = {}

    def chromosome_to_key(self, chromosome):
        """Convert chromosome to hashable key for caching"""
        return tuple(np.where(chromosome == 1)[0])

    @lru_cache(maxsize=5000)
    def evaluate_objectives_cached(self, indices_tuple):
        """Cached evaluation of objectives"""
        indices = np.array(indices_tuple)

        # Fast linked usage calculation
        lu_score = self.calculate_linked_usage_fast(indices)

        # Fast semantic similarity
        ss_score = self.calculate_semantic_similarity_fast(indices)

        # Size
        rss_score = len(indices)

        return np.array([-lu_score, -ss_score, rss_score])

    def calculate_linked_usage_fast(self, indices):
        """Vectorized linked usage calculation"""
        if len(indices) == 0:
            return 0

        # Extract submatrix and sum (vectorized)
        if hasattr(self.rel_matrix, 'toarray'):
            submatrix = self.rel_matrix[indices][:, indices].toarray()
        else:
            submatrix = self.rel_matrix[np.ix_(indices, indices)]

        # Sum all connections (excluding diagonal)
        score = submatrix.sum() - np.diagonal(submatrix).sum()
        return score

    def calculate_semantic_similarity_fast(self, indices):
        """Optimized semantic similarity using vectorization"""
        if len(indices) <= 1:
            return 0

        # Vectorized centroid calculation
        embeddings_subset = self.embeddings[indices]
        centroid = embeddings_subset.mean(axis=0)

        # Vectorized cosine similarity
        norms = np.linalg.norm(embeddings_subset, axis=1)
        centroid_norm = np.linalg.norm(centroid)

        # Dot product with centroid
        dots = embeddings_subset @ centroid

        # Cosine similarities
        similarities = dots / (norms * centroid_norm + 1e-10)

        return similarities.mean()

    def evaluate_batch(self, chromosomes):
        """Evaluate multiple chromosomes in batch"""
        results = []
        for chromosome in chromosomes:
            key = self.chromosome_to_key(chromosome)
            obj = self.evaluate_objectives_cached(key)
            results.append(obj)
        return np.array(results)


def parallel_evaluate_batch(evaluator, chromosomes, n_processes=10):
    """
    Parallel evaluation using multiprocessing
    Uses 10 cores, leaving 2 for system
    """
    chunk_size = len(chromosomes) // n_processes + 1
    chunks = [chromosomes[i:i+chunk_size]
              for i in range(0, len(chromosomes), chunk_size)]

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(evaluator.evaluate_batch,
                              [(chunk,) for chunk in chunks])

    return np.vstack(results)


# Example benchmark
if __name__ == "__main__":
    import time
    import pickle
    import os

    print("Loading data...")
    data_dir = 'pycommend-code/data'

    with open(os.path.join(data_dir, 'package_relationships_10k.pkl'), 'rb') as f:
        rel_matrix = pickle.load(f)

    with open(os.path.join(data_dir, 'package_similarity_matrix_10k.pkl'), 'rb') as f:
        sim_matrix = pickle.load(f)

    with open(os.path.join(data_dir, 'package_embeddings_10k.pkl'), 'rb') as f:
        embeddings = pickle.load(f)

    n_packages = len(embeddings)

    # Create evaluator
    evaluator = OptimizedEvaluator(rel_matrix, sim_matrix, embeddings, 0)

    # Generate test chromosomes
    print("\nGenerating 1000 test chromosomes...")
    test_chromosomes = []
    for _ in range(1000):
        chromosome = np.zeros(n_packages)
        indices = np.random.choice(n_packages, size=5, replace=False)
        chromosome[indices] = 1
        test_chromosomes.append(chromosome)

    # Benchmark sequential
    print("\nBenchmarking sequential evaluation...")
    start = time.time()
    seq_results = evaluator.evaluate_batch(test_chromosomes)
    seq_time = time.time() - start
    print(f"Sequential: {seq_time:.2f}s for 1000 evaluations")
    print(f"Speed: {1000/seq_time:.1f} evals/sec")

    # Benchmark parallel
    print("\nBenchmarking parallel evaluation (10 cores)...")
    start = time.time()
    par_results = parallel_evaluate_batch(evaluator, test_chromosomes, n_processes=10)
    par_time = time.time() - start
    print(f"Parallel: {par_time:.2f}s for 1000 evaluations")
    print(f"Speed: {1000/par_time:.1f} evals/sec")
    print(f"Speedup: {seq_time/par_time:.1f}x")

    # Check cache effectiveness
    print(f"\nCache size: {len(evaluator.evaluate_objectives_cached.cache_info())}")
    print(f"Cache info: {evaluator.evaluate_objectives_cached.cache_info()}")