#!/usr/bin/env python3
"""
COMPREHENSIVE SEMANTIC IMPROVEMENTS FOR PYCOMMEND

Based on analysis of available data sources:
1. Package Relationships Matrix (9997x9997) - Co-occurrence data from GitHub repos
2. Package Similarity Matrix (9997x9997) - Semantic similarity from embeddings
3. Package Embeddings (9997x384) - SBERT embeddings of package descriptions

Current Algorithm Limitations:
- Only uses relationships matrix for F1, similarity matrix for F2
- Random initialization from entire 10k pool (0.01% chance of good candidates)
- No semantic clustering or domain awareness
- Missing opportunities for multi-modal fusion

PROPOSED SEMANTIC IMPROVEMENTS:
"""

import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import random

class SemanticEnhancedRecommender:
    """Enhanced recommender with semantic improvements."""

    def __init__(self, package_name, pop_size=100, max_gen=100):
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.max_gen = max_gen

        # Load all data sources
        self.load_semantic_data()

        # Semantic enhancements
        self.create_semantic_clusters()
        self.build_candidate_filters()

        print(f"Enhanced recommender initialized for '{package_name}'")
        print(f"Semantic clusters: {self.n_clusters}")
        print(f"Target cluster for {package_name}: {self.package_cluster}")

    def load_semantic_data(self):
        """Load all three semantic data sources."""
        # Relationships matrix (co-occurrence)
        with open("E:/pycommend/pycommend-code/data/package_relationships_10k.pkl", 'rb') as f:
            rel_data = pickle.load(f)
        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']

        # Similarity matrix (precomputed from embeddings)
        with open("E:/pycommend/pycommend-code/data/package_similarity_matrix_10k.pkl", 'rb') as f:
            sim_data = pickle.load(f)
        self.sim_matrix = sim_data['similarity_matrix']

        # Raw embeddings for advanced semantic operations
        with open("E:/pycommend/pycommend-code/data/package_embeddings_10k.pkl", 'rb') as f:
            emb_data = pickle.load(f)
        self.embeddings = emb_data['embeddings']

        # Create mappings
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}
        self.n_packages = len(self.package_names)

        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{self.package_name}' not found")

        self.package_idx = self.pkg_to_idx[self.package_name]

    def create_semantic_clusters(self, n_clusters=50):
        """
        IMPROVEMENT 1: SEMANTIC CLUSTERING
        Create semantic clusters using embeddings to identify package domains.
        """
        print("Creating semantic clusters...")

        # Use K-means on embeddings to create clusters
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.package_clusters = self.cluster_model.fit_predict(self.embeddings)

        # Find which cluster the target package belongs to
        self.package_cluster = self.package_clusters[self.package_idx]

        # Create cluster-based candidate pools
        self.cluster_pools = {}
        for cluster_id in range(n_clusters):
            cluster_packages = np.where(self.package_clusters == cluster_id)[0]
            self.cluster_pools[cluster_id] = cluster_packages

        print(f"Target package '{self.package_name}' is in cluster {self.package_cluster}")
        print(f"Cluster {self.package_cluster} has {len(self.cluster_pools[self.package_cluster])} packages")

    def build_candidate_filters(self):
        """
        IMPROVEMENT 2: EMBEDDING-BASED CANDIDATE FILTERING
        Pre-filter candidates using semantic similarity thresholds.
        """
        print("Building candidate filters...")

        # Compute similarities to target package
        target_embedding = self.embeddings[self.package_idx]
        all_similarities = cosine_similarity([target_embedding], self.embeddings)[0]

        # Define similarity thresholds for different pool sizes
        self.candidate_pools = {
            'high_similarity': np.where(all_similarities > 0.6)[0],    # Very similar
            'medium_similarity': np.where(all_similarities > 0.4)[0],  # Moderately similar
            'low_similarity': np.where(all_similarities > 0.2)[0],     # Somewhat similar
        }

        # Remove target package from pools
        for pool_name in self.candidate_pools:
            self.candidate_pools[pool_name] = self.candidate_pools[pool_name][
                self.candidate_pools[pool_name] != self.package_idx
            ]

        print(f"High similarity candidates: {len(self.candidate_pools['high_similarity'])}")
        print(f"Medium similarity candidates: {len(self.candidate_pools['medium_similarity'])}")
        print(f"Low similarity candidates: {len(self.candidate_pools['low_similarity'])}")

    def smart_initialization(self, strategy='multi_modal'):
        """
        IMPROVEMENT 3: INTELLIGENT INITIALIZATION
        Multiple strategies for smart initialization instead of random.
        """
        if strategy == 'cluster_based':
            return self._cluster_based_init()
        elif strategy == 'similarity_based':
            return self._similarity_based_init()
        elif strategy == 'co_occurrence_based':
            return self._co_occurrence_based_init()
        elif strategy == 'multi_modal':
            return self._multi_modal_init()
        else:
            return self._random_init()  # Fallback

    def _cluster_based_init(self):
        """Initialize primarily from same semantic cluster."""
        same_cluster = self.cluster_pools[self.package_cluster]
        # Remove target package
        candidates = same_cluster[same_cluster != self.package_idx]

        # 70% from same cluster, 30% from other clusters for diversity
        size = random.randint(3, 15)
        same_cluster_size = int(0.7 * size)
        other_cluster_size = size - same_cluster_size

        solution = []

        # Add from same cluster
        if len(candidates) >= same_cluster_size:
            solution.extend(np.random.choice(candidates, same_cluster_size, replace=False))
        else:
            solution.extend(candidates)

        # Add from other clusters for diversity
        other_candidates = np.setdiff1d(range(self.n_packages),
                                      np.concatenate([same_cluster, [self.package_idx]]))
        if len(other_candidates) >= other_cluster_size:
            solution.extend(np.random.choice(other_candidates, other_cluster_size, replace=False))

        return np.array(solution)

    def _similarity_based_init(self):
        """Initialize using embedding similarity."""
        # Use high similarity pool with some medium similarity for diversity
        size = random.randint(3, 15)
        high_sim_size = int(0.8 * size)
        med_sim_size = size - high_sim_size

        solution = []

        # High similarity candidates
        high_candidates = self.candidate_pools['high_similarity']
        if len(high_candidates) >= high_sim_size:
            solution.extend(np.random.choice(high_candidates, high_sim_size, replace=False))
        else:
            solution.extend(high_candidates)

        # Medium similarity for diversity
        med_candidates = np.setdiff1d(self.candidate_pools['medium_similarity'], solution)
        if len(med_candidates) >= med_sim_size:
            solution.extend(np.random.choice(med_candidates, med_sim_size, replace=False))

        return np.array(solution)

    def _co_occurrence_based_init(self):
        """Initialize using co-occurrence strength."""
        # Get strongest co-occurrence relationships
        target_row = self.rel_matrix[self.package_idx].toarray().flatten()
        target_row[self.package_idx] = 0  # Remove self

        # Top candidates by co-occurrence
        top_indices = np.argsort(target_row)[::-1]

        # Filter out zero values
        top_candidates = top_indices[target_row[top_indices] > 0][:100]  # Top 100

        size = random.randint(3, 15)
        if len(top_candidates) >= size:
            solution = np.random.choice(top_candidates, size, replace=False)
        else:
            solution = top_candidates

        return solution

    def _multi_modal_init(self):
        """
        IMPROVEMENT 4: MULTI-MODAL INITIALIZATION
        Combine all three data sources for optimal initialization.
        """
        size = random.randint(3, 15)

        # Strategy: 40% co-occurrence + 40% similarity + 20% cluster diversity
        co_occ_size = int(0.4 * size)
        sim_size = int(0.4 * size)
        cluster_size = size - co_occ_size - sim_size

        solution = []

        # 1. Top co-occurrence candidates
        target_row = self.rel_matrix[self.package_idx].toarray().flatten()
        target_row[self.package_idx] = 0
        co_occ_candidates = np.argsort(target_row)[::-1]
        co_occ_candidates = co_occ_candidates[target_row[co_occ_candidates] > 0][:50]

        if len(co_occ_candidates) >= co_occ_size:
            solution.extend(np.random.choice(co_occ_candidates, co_occ_size, replace=False))
        else:
            solution.extend(co_occ_candidates)

        # 2. High similarity candidates (excluding already selected)
        sim_candidates = np.setdiff1d(self.candidate_pools['high_similarity'], solution)
        if len(sim_candidates) >= sim_size:
            solution.extend(np.random.choice(sim_candidates, sim_size, replace=False))
        elif len(sim_candidates) > 0:
            solution.extend(sim_candidates)

        # 3. Cluster diversity (from same cluster but not in other pools)
        cluster_candidates = np.setdiff1d(self.cluster_pools[self.package_cluster],
                                        np.concatenate([solution, [self.package_idx]]))
        if len(cluster_candidates) >= cluster_size:
            solution.extend(np.random.choice(cluster_candidates, cluster_size, replace=False))

        return np.array(solution)

    def _random_init(self):
        """Fallback random initialization."""
        size = random.randint(3, 15)
        candidates = [i for i in range(self.n_packages) if i != self.package_idx]
        return np.random.choice(candidates, size, replace=False)

    def evaluate_enhanced_objectives(self, chromosome):
        """
        IMPROVEMENT 5: ENHANCED MULTI-MODAL OBJECTIVE FUNCTION
        Combine all data sources in a sophisticated objective function.
        """
        package_idx = self.package_idx
        selected_indices = [i for i in range(self.n_packages) if chromosome[i] == 1]
        all_indices = selected_indices + [package_idx]

        size = len(selected_indices)

        if size < 3:  # Minimum size constraint
            return [float('inf')] * 4

        # F1: Enhanced Co-occurrence Score
        f1 = self._compute_enhanced_co_occurrence(all_indices)

        # F2: Multi-Modal Semantic Similarity
        f2 = self._compute_multi_modal_similarity(selected_indices, package_idx)

        # F3: Semantic Coherence (NEW)
        f3 = self._compute_semantic_coherence(selected_indices)

        # F4: Solution Size
        f4 = size

        return [f1, f2, f3, f4]

    def _compute_enhanced_co_occurrence(self, all_indices):
        """Enhanced co-occurrence with diversity bonus."""
        total_score = 0.0
        count = 0
        strong_connections = set()

        for i in range(len(all_indices)):
            for j in range(i+1, len(all_indices)):
                link_value = self.rel_matrix[all_indices[i], all_indices[j]]
                total_score += link_value
                count += 1

                if link_value > 3.0:  # Strong connection threshold
                    strong_connections.add(all_indices[i])
                    strong_connections.add(all_indices[j])

        avg_score = total_score / count if count > 0 else 0.0

        # Diversity bonus
        proportion_strong = len(strong_connections) / len(all_indices)
        diversity_bonus = 1 + np.log1p(proportion_strong) if proportion_strong > 0 else 1

        return -avg_score * diversity_bonus  # Negative for minimization

    def _compute_multi_modal_similarity(self, selected_indices, package_idx):
        """Combine precomputed similarity with direct embedding similarity."""
        if not selected_indices:
            return 0.0

        # Method 1: Precomputed similarity matrix
        sim_matrix_score = np.mean([self.sim_matrix[package_idx, idx] for idx in selected_indices])

        # Method 2: Direct embedding cosine similarity
        target_emb = self.embeddings[package_idx]
        selected_embs = self.embeddings[selected_indices]
        direct_similarities = cosine_similarity([target_emb], selected_embs)[0]
        direct_score = np.mean(direct_similarities)

        # Weighted combination
        combined_score = 0.7 * sim_matrix_score + 0.3 * direct_score

        return -combined_score  # Negative for minimization

    def _compute_semantic_coherence(self, selected_indices):
        """
        IMPROVEMENT 6: SEMANTIC COHERENCE OBJECTIVE
        Measure how semantically coherent the selected packages are.
        """
        if len(selected_indices) < 2:
            return 0.0

        # Compute pairwise similarities between selected packages
        selected_embs = self.embeddings[selected_indices]
        pairwise_similarities = cosine_similarity(selected_embs)

        # Remove diagonal (self-similarities)
        mask = ~np.eye(pairwise_similarities.shape[0], dtype=bool)
        similarities = pairwise_similarities[mask]

        # Coherence metrics
        mean_coherence = np.mean(similarities)
        std_coherence = np.std(similarities)

        # We want high mean similarity but not too high std (balanced coherence)
        coherence_score = mean_coherence - 0.1 * std_coherence

        return -coherence_score  # Negative for minimization

    def analyze_semantic_improvements(self, package_name_to_test="numpy"):
        """Test and compare different initialization strategies."""
        print(f"\n{'='*80}")
        print(f"TESTING SEMANTIC IMPROVEMENTS FOR {package_name_to_test.upper()}")
        print(f"{'='*80}")

        strategies = ['random', 'cluster_based', 'similarity_based', 'co_occurrence_based', 'multi_modal']

        for strategy in strategies:
            print(f"\n{strategy.upper()} INITIALIZATION:")
            print("-" * 40)

            # Generate 5 solutions with this strategy
            for i in range(5):
                solution_indices = self.smart_initialization(strategy)

                # Convert to chromosome format
                chromosome = np.zeros(self.n_packages)
                chromosome[solution_indices] = 1

                # Evaluate with enhanced objectives
                objectives = self.evaluate_enhanced_objectives(chromosome)

                # Show selected packages
                selected_packages = [self.package_names[idx] for idx in solution_indices]

                print(f"  Solution {i+1}:")
                print(f"    Packages: {selected_packages[:5]}{'...' if len(selected_packages) > 5 else ''}")
                print(f"    Objectives [co-occ, similarity, coherence, size]: {[f'{obj:.4f}' for obj in objectives]}")


def demonstrate_semantic_improvements():
    """Demonstrate the semantic improvements with concrete examples."""

    # Test with numpy
    recommender = SemanticEnhancedRecommender("numpy")
    recommender.analyze_semantic_improvements("numpy")

    # Show cluster analysis
    print(f"\n{'='*80}")
    print("SEMANTIC CLUSTER ANALYSIS")
    print(f"{'='*80}")

    # Show packages in numpy's cluster
    numpy_cluster = recommender.package_cluster
    cluster_packages = recommender.cluster_pools[numpy_cluster]
    cluster_names = [recommender.package_names[idx] for idx in cluster_packages[:20]]

    print(f"NumPy is in cluster {numpy_cluster}")
    print(f"Other packages in this cluster: {cluster_names}")

    # Show similarity pools
    print(f"\nHigh similarity candidates to NumPy:")
    high_sim_indices = recommender.candidate_pools['high_similarity'][:10]
    high_sim_names = [recommender.package_names[idx] for idx in high_sim_indices]
    print(f"  {high_sim_names}")

if __name__ == "__main__":
    demonstrate_semantic_improvements()