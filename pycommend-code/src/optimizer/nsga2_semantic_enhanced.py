#!/usr/bin/env python3
"""
NSGA-II with Semantic Enhancements for PyCommend
=================================================

This is an enhanced version of the original NSGA-II algorithm that incorporates
semantic improvements based on the comprehensive analysis of available data sources.

Key Improvements:
1. Intelligent initialization using semantic clustering and similarity
2. Multi-modal objective function combining all three data sources
3. Embedding-based candidate pre-filtering
4. Semantic coherence as an additional optimization objective

Author: PyCommend Team
Date: September 2024
"""

import numpy as np
import pickle
import random
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

class SemanticNSGA2:
    def __init__(self, package_name, pop_size=100, max_gen=100, semantic_mode='full'):
        """
        Enhanced NSGA-II with semantic improvements.

        Args:
            package_name: Target package for recommendations
            pop_size: Population size
            max_gen: Maximum generations
            semantic_mode: 'full', 'cluster_only', 'similarity_only', or 'basic'
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.semantic_mode = semantic_mode

        # Load all semantic data
        self.load_semantic_data()

        # Initialize semantic enhancements
        if semantic_mode != 'basic':
            self.setup_semantic_enhancements()

        # Algorithm parameters
        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / self.n_packages
        self.min_size = 3
        self.max_size = 15

        # Calculate thresholds
        self.colink_threshold = self.calculate_colink_threshold()

        print(f"Semantic NSGA-II initialized for '{package_name}' (mode: {semantic_mode})")
        print(f"Packages: {self.n_packages}, Threshold: {self.colink_threshold:.2f}")

    def load_semantic_data(self):
        """Load all three semantic data sources."""
        # Load relationships matrix
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)
        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']

        # Load similarity matrix
        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)
        self.sim_matrix = sim_data['similarity_matrix']

        # Load embeddings
        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            emb_data = pickle.load(f)
        self.embeddings = emb_data['embeddings']

        # Create mappings
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}
        self.n_packages = len(self.package_names)

        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{self.package_name}' not found")

        self.package_idx = self.pkg_to_idx[self.package_name]

    def setup_semantic_enhancements(self):
        """Setup semantic clustering and candidate filtering."""
        print("Setting up semantic enhancements...")

        # Create semantic clusters
        if self.semantic_mode in ['full', 'cluster_only']:
            self.cluster_model = KMeans(n_clusters=50, random_state=42, n_init=10)
            self.package_clusters = self.cluster_model.fit_predict(self.embeddings)
            self.package_cluster = self.package_clusters[self.package_idx]

            # Create cluster pools
            self.cluster_pools = {}
            for cluster_id in range(50):
                cluster_packages = np.where(self.package_clusters == cluster_id)[0]
                self.cluster_pools[cluster_id] = cluster_packages

        # Create similarity-based candidate pools
        if self.semantic_mode in ['full', 'similarity_only']:
            target_embedding = self.embeddings[self.package_idx]
            all_similarities = cosine_similarity([target_embedding], self.embeddings)[0]

            self.candidate_pools = {
                'high': np.where(all_similarities > 0.6)[0],
                'medium': np.where(all_similarities > 0.4)[0],
                'low': np.where(all_similarities > 0.2)[0],
            }

            # Remove target package from pools
            for pool_name in self.candidate_pools:
                self.candidate_pools[pool_name] = self.candidate_pools[pool_name][
                    self.candidate_pools[pool_name] != self.package_idx
                ]

    def calculate_colink_threshold(self):
        """Calculate threshold for strong connections."""
        sample_size = 10000
        sample_values = []

        rows = np.random.randint(0, self.rel_matrix.shape[0], sample_size)
        cols = np.random.randint(0, self.rel_matrix.shape[1], sample_size)

        for i in range(sample_size):
            val = self.rel_matrix[rows[i], cols[i]]
            if val > 0:
                sample_values.append(val)

        return np.percentile(sample_values, 75) if sample_values else 1.0

    def create_individual_smart(self):
        """Smart initialization using semantic information."""
        if self.semantic_mode == 'basic':
            return self.create_individual_random()
        elif self.semantic_mode == 'cluster_only':
            return self.create_individual_cluster()
        elif self.semantic_mode == 'similarity_only':
            return self.create_individual_similarity()
        else:  # full mode
            return self.create_individual_multimodal()

    def create_individual_random(self):
        """Basic random initialization."""
        size = random.randint(self.min_size, self.max_size)
        candidates = [i for i in range(self.n_packages) if i != self.package_idx]
        selected = random.sample(candidates, size)

        chromosome = np.zeros(self.n_packages)
        chromosome[selected] = 1

        return {
            'chromosome': chromosome,
            'objectives': None,
            'rank': None,
            'crowding_distance': 0
        }

    def create_individual_cluster(self):
        """Cluster-based initialization."""
        size = random.randint(self.min_size, self.max_size)

        # 70% from same cluster, 30% from other clusters
        same_cluster_size = int(0.7 * size)
        other_size = size - same_cluster_size

        selected = []

        # From same cluster
        same_cluster_candidates = self.cluster_pools[self.package_cluster]
        same_cluster_candidates = same_cluster_candidates[same_cluster_candidates != self.package_idx]

        if len(same_cluster_candidates) >= same_cluster_size:
            selected.extend(np.random.choice(same_cluster_candidates, same_cluster_size, replace=False))
        else:
            selected.extend(same_cluster_candidates)

        # From other clusters
        if other_size > 0:
            other_candidates = np.setdiff1d(range(self.n_packages),
                                          np.concatenate([same_cluster_candidates, [self.package_idx]]))
            if len(other_candidates) >= other_size:
                selected.extend(np.random.choice(other_candidates, other_size, replace=False))

        chromosome = np.zeros(self.n_packages)
        chromosome[selected] = 1

        return {
            'chromosome': chromosome,
            'objectives': None,
            'rank': None,
            'crowding_distance': 0
        }

    def create_individual_similarity(self):
        """Similarity-based initialization."""
        size = random.randint(self.min_size, self.max_size)

        # 80% high similarity, 20% medium similarity
        high_size = int(0.8 * size)
        medium_size = size - high_size

        selected = []

        # High similarity candidates
        high_candidates = self.candidate_pools['high']
        if len(high_candidates) >= high_size:
            selected.extend(np.random.choice(high_candidates, high_size, replace=False))
        else:
            selected.extend(high_candidates)

        # Medium similarity candidates
        if medium_size > 0:
            medium_candidates = np.setdiff1d(self.candidate_pools['medium'], selected)
            if len(medium_candidates) >= medium_size:
                selected.extend(np.random.choice(medium_candidates, medium_size, replace=False))

        chromosome = np.zeros(self.n_packages)
        chromosome[selected] = 1

        return {
            'chromosome': chromosome,
            'objectives': None,
            'rank': None,
            'crowding_distance': 0
        }

    def create_individual_multimodal(self):
        """Multi-modal initialization combining all approaches."""
        size = random.randint(self.min_size, self.max_size)

        # 40% co-occurrence + 40% similarity + 20% cluster diversity
        co_occ_size = int(0.4 * size)
        sim_size = int(0.4 * size)
        cluster_size = size - co_occ_size - sim_size

        selected = []

        # 1. Top co-occurrence candidates
        target_row = self.rel_matrix[self.package_idx].toarray().flatten()
        target_row[self.package_idx] = 0
        co_occ_candidates = np.argsort(target_row)[::-1]
        co_occ_candidates = co_occ_candidates[target_row[co_occ_candidates] > 0][:50]

        if len(co_occ_candidates) >= co_occ_size:
            selected.extend(np.random.choice(co_occ_candidates, co_occ_size, replace=False))
        else:
            selected.extend(co_occ_candidates)

        # 2. High similarity candidates
        sim_candidates = np.setdiff1d(self.candidate_pools['high'], selected)
        if len(sim_candidates) >= sim_size:
            selected.extend(np.random.choice(sim_candidates, sim_size, replace=False))
        elif len(sim_candidates) > 0:
            selected.extend(sim_candidates)

        # 3. Cluster diversity
        if cluster_size > 0:
            cluster_candidates = np.setdiff1d(self.cluster_pools[self.package_cluster],
                                            np.concatenate([selected, [self.package_idx]]))
            if len(cluster_candidates) >= cluster_size:
                selected.extend(np.random.choice(cluster_candidates, cluster_size, replace=False))

        chromosome = np.zeros(self.n_packages)
        chromosome[selected] = 1

        return {
            'chromosome': chromosome,
            'objectives': None,
            'rank': None,
            'crowding_distance': 0
        }

    def evaluate_objectives(self, chromosome):
        """Enhanced objective evaluation with semantic information."""
        package_idx = self.package_idx
        selected_indices = [i for i in range(self.n_packages) if chromosome[i] == 1]
        all_indices = selected_indices + [package_idx]

        size = len(selected_indices)

        if size < self.min_size:
            if self.semantic_mode == 'full':
                return [float('inf')] * 4
            else:
                return [float('inf')] * 3

        # F1: Enhanced co-occurrence score
        f1 = self.compute_enhanced_cooccurrence(all_indices)

        # F2: Multi-modal semantic similarity
        f2 = self.compute_multimodal_similarity(selected_indices, package_idx)

        # F3: Solution size
        f3 = size

        if self.semantic_mode == 'full':
            # F4: Semantic coherence (only in full mode)
            f4 = self.compute_semantic_coherence(selected_indices)
            return [f1, f2, f3, f4]
        else:
            return [f1, f2, f3]

    def compute_enhanced_cooccurrence(self, all_indices):
        """Enhanced co-occurrence with diversity bonus."""
        total_score = 0.0
        count = 0
        strong_connections = set()

        for i in range(len(all_indices)):
            for j in range(i+1, len(all_indices)):
                link_value = self.rel_matrix[all_indices[i], all_indices[j]]
                total_score += link_value
                count += 1

                if link_value > self.colink_threshold:
                    strong_connections.add(all_indices[i])
                    strong_connections.add(all_indices[j])

        avg_score = total_score / count if count > 0 else 0.0

        # Diversity bonus
        if self.package_idx in strong_connections:
            strong_connections.remove(self.package_idx)

        proportion_strong = len(strong_connections) / len(all_indices)
        diversity_bonus = 1 + np.log1p(proportion_strong) if proportion_strong > 0 else 1

        return -avg_score * diversity_bonus

    def compute_multimodal_similarity(self, selected_indices, package_idx):
        """Combine precomputed similarity with direct embedding similarity."""
        if not selected_indices:
            return 0.0

        # Precomputed similarity matrix
        sim_matrix_score = np.mean([self.sim_matrix[package_idx, idx] for idx in selected_indices])

        # Direct embedding similarity
        target_emb = self.embeddings[package_idx]
        selected_embs = self.embeddings[selected_indices]
        direct_similarities = cosine_similarity([target_emb], selected_embs)[0]
        direct_score = np.mean(direct_similarities)

        # Weighted combination
        combined_score = 0.7 * sim_matrix_score + 0.3 * direct_score

        return -combined_score

    def compute_semantic_coherence(self, selected_indices):
        """Measure semantic coherence of selected packages."""
        if len(selected_indices) < 2:
            return 0.0

        selected_embs = self.embeddings[selected_indices]
        pairwise_similarities = cosine_similarity(selected_embs)

        # Remove diagonal (self-similarities)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        similarities = pairwise_similarities[mask]

        mean_coherence = np.mean(similarities)
        std_coherence = np.std(similarities)

        # High mean, low std = good coherence
        coherence_score = mean_coherence - 0.1 * std_coherence

        return -coherence_score

    def run(self):
        """Main NSGA-II algorithm loop with semantic enhancements."""
        print(f"Starting Semantic NSGA-II (mode: {self.semantic_mode})...")
        start_time = time.time()

        # Initialize population with smart initialization
        print("Initializing population with semantic intelligence...")
        P = []
        for _ in range(self.pop_size):
            individual = self.create_individual_smart()
            individual['objectives'] = self.evaluate_objectives(individual['chromosome'])
            P.append(individual)

        # Main evolutionary loop
        for generation in range(self.max_gen):
            if generation % 20 == 0:
                print(f"Generation {generation}/{self.max_gen}")

            # Create offspring population Q
            Q = []
            while len(Q) < self.pop_size:
                # Selection
                parent1_idx = self.tournament_selection(P)
                parent2_idx = self.tournament_selection(P)

                # Crossover
                child1_chrom, child2_chrom = self.sbx_crossover(
                    P[parent1_idx]['chromosome'],
                    P[parent2_idx]['chromosome']
                )

                # Mutation
                child1_chrom = self.polynomial_mutation(child1_chrom)
                child2_chrom = self.polynomial_mutation(child2_chrom)

                # Create offspring individuals
                for child_chrom in [child1_chrom, child2_chrom]:
                    if len(Q) < self.pop_size:
                        child = {
                            'chromosome': child_chrom,
                            'objectives': self.evaluate_objectives(child_chrom),
                            'rank': None,
                            'crowding_distance': 0
                        }
                        Q.append(child)

            # Combine populations and select next generation
            R = P + Q
            P = self.environmental_selection(R)

        end_time = time.time()

        # Extract best solutions from final Pareto front
        fronts = self.non_dominated_sort(P)
        best_solutions = []

        for individual in [P[i] for i in fronts[0]]:
            selected_indices = [i for i in range(self.n_packages) if individual['chromosome'][i] == 1]
            packages = [self.package_names[i] for i in selected_indices]
            best_solutions.append({
                'packages': packages,
                'objectives': individual['objectives'],
                'size': len(packages)
            })

        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(best_solutions)} solutions in Pareto front")

        return best_solutions

    # Include all the other NSGA-II methods (non_dominated_sort, crowding_distance_assignment, etc.)
    # [The rest of the NSGA-II implementation remains the same as the original]
    # For brevity, I'm not repeating all methods, but they would be included in the full implementation

    def non_dominated_sort(self, P):
        """Non-dominated sorting for NSGA-II."""
        # Implementation would be the same as the original NSGA-II
        pass

    def environmental_selection(self, R):
        """Environmental selection for next generation."""
        # Implementation would be the same as the original NSGA-II
        pass

    def tournament_selection(self, P):
        """Tournament selection."""
        # Implementation would be the same as the original NSGA-II
        pass

    def sbx_crossover(self, parent1, parent2):
        """SBX crossover adapted for binary chromosomes."""
        # Implementation would be the same as the original NSGA-II
        pass

    def polynomial_mutation(self, chromosome):
        """Polynomial mutation adapted for binary chromosomes."""
        # Implementation would be the same as the original NSGA-II
        pass

def main():
    """Test the semantic enhanced NSGA-II."""
    parser = argparse.ArgumentParser(description='Semantic Enhanced NSGA-II for Package Recommendation')
    parser.add_argument('--package', type=str, required=True, help='Target package name')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'cluster_only', 'similarity_only', 'basic'],
                       help='Semantic enhancement mode')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--max_gen', type=int, default=100, help='Maximum generations')

    args = parser.parse_args()

    # Run semantic enhanced NSGA-II
    nsga2 = SemanticNSGA2(
        package_name=args.package,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        semantic_mode=args.mode
    )

    solutions = nsga2.run()

    # Display results
    print(f"\nBest recommendations for '{args.package}':")
    print("=" * 80)

    for i, solution in enumerate(solutions[:5]):  # Show top 5 solutions
        print(f"\nSolution {i+1}:")
        print(f"  Packages: {solution['packages']}")
        print(f"  Objectives: {[f'{obj:.4f}' for obj in solution['objectives']]}")
        print(f"  Size: {solution['size']}")

if __name__ == "__main__":
    main()