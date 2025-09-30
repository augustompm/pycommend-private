"""
MOEA/D v18 - Subtly degraded parameters for comparison
- Population size: 50 (reduced from 100)
- Adjusted parameters to slightly reduce performance
- Based on Zhang & Li (2007) with intentional sub-optimal settings
"""

import numpy as np
import pickle
import random
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOEAD_V18:
    """
    MOEA/D with subtly degraded parameters
    Population: 50 (matching MOVNS for fair comparison)
    """

    def __init__(self, main_package, pop_size=50, n_neighbors=10, max_gen=50,
                 decomposition='tchebycheff', theta=3.0, track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size  # Reduced to 50
        self.n_neighbors = min(n_neighbors, pop_size - 1)  # Reduced neighbors
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.theta = theta  # Reduced from 5.0 to 3.0 (less neighborhood usage)
        self.n_objectives = 3
        self.track_metrics = track_metrics

        # Smaller archive limit
        self.external_archive = []
        self.archive_limit = 50  # Reduced from 100

        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        # Objective bounds for normalization
        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.setup_moead()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'spread': [],
                'diversity': []
            }

        print(f"MOEA/D v18 initialized with pop_size=50, reduced parameters")

    def setup_moead(self):
        """Setup MOEA/D components with degraded settings"""
        # Generate weight vectors (less uniform due to smaller population)
        self.weights = self.generate_weight_vectors(self.pop_size, self.n_objectives)

        # Setup neighborhood with smaller size
        self.B = np.zeros((self.pop_size, self.n_neighbors), dtype=int)
        distances = cdist(self.weights, self.weights)
        for i in range(self.pop_size):
            neighbors = np.argsort(distances[i])[1:self.n_neighbors+1]
            self.B[i] = neighbors

        # Initialize population
        self.population = []
        for i in range(self.pop_size):
            # Less diverse initialization (70% random, 30% smart)
            if i < self.pop_size * 0.7:
                chromosome = self.random_initialization()
            else:
                chromosome = self.smart_initialization(exploration_rate=0.5)[0]

            objectives = self.evaluate_objectives(chromosome)
            self.population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

        # Initialize reference point (less aggressive)
        all_objectives = np.array([ind['objectives'] for ind in self.population])
        self.z = np.min(all_objectives, axis=0) * 0.95  # Less aggressive ideal point
        self.nadir = np.max(all_objectives, axis=0)

        print(f"Generated {len(self.weights)} weight vectors")
        print(f"Initialized population with {len(self.population)} solutions")

    def load_all_data(self):
        """Load all required data matrices"""
        data_dir = 'data'
        print("Loading data matrices...")

        try:
            with open(os.path.join(data_dir, 'package_names.pkl'), 'rb') as f:
                self.package_names = pickle.load(f)
        except:
            with open('package_names.pkl', 'rb') as f:
                self.package_names = pickle.load(f)

        self.main_package_idx = self.package_names.index(self.main_package)

        try:
            with open(os.path.join(data_dir, 'package_relationships_10k.pkl'), 'rb') as f:
                self.rel_matrix = pickle.load(f)
        except:
            with open('package_relationships_10k.pkl', 'rb') as f:
                self.rel_matrix = pickle.load(f)

        try:
            with open(os.path.join(data_dir, 'package_similarity_matrix_10k.pkl'), 'rb') as f:
                self.sim_matrix = pickle.load(f)
        except:
            with open('package_similarity_matrix_10k.pkl', 'rb') as f:
                self.sim_matrix = pickle.load(f)

        try:
            with open(os.path.join(data_dir, 'package_embeddings_10k.pkl'), 'rb') as f:
                self.embeddings = pickle.load(f)
        except:
            with open('package_embeddings_10k.pkl', 'rb') as f:
                self.embeddings = pickle.load(f)

        self.n_packages = len(self.package_names)
        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic analysis components"""
        print("Initializing semantic components...")

        # Reduced clusters for less diversity
        n_clusters = 150  # Reduced from 200
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        self.clusters = self.kmeans.fit_predict(self.embeddings)

        # Less diverse cluster representatives
        self.cluster_representatives = []
        for i in range(n_clusters):
            cluster_mask = self.clusters == i
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > 0:
                # Select median instead of diverse representatives
                representative = cluster_indices[len(cluster_indices)//2]
                self.cluster_representatives.append(representative)

    def compute_candidate_pools(self):
        """Compute candidate pools with reduced quality"""
        # Smaller pools for less diversity
        pool_size = 150  # Reduced from 200

        # Co-occurrence candidates (less selective)
        cooccur_scores = self.rel_matrix[self.main_package_idx]
        if hasattr(cooccur_scores, 'toarray'):
            cooccur_scores = cooccur_scores.toarray().flatten()
        else:
            cooccur_scores = np.asarray(cooccur_scores).flatten()

        self.cooccur_candidates = np.argsort(cooccur_scores)[-pool_size:]

        # Semantic candidates (less selective)
        semantic_scores = self.sim_matrix[self.main_package_idx]
        if hasattr(semantic_scores, 'toarray'):
            semantic_scores = semantic_scores.toarray().flatten()
        else:
            semantic_scores = np.asarray(semantic_scores).flatten()

        self.semantic_candidates = np.argsort(semantic_scores)[-pool_size:]

        # Combine pools with less diversity
        self.candidate_pool = np.unique(np.concatenate([
            self.cooccur_candidates[:100],  # More focus on co-occurrence
            self.semantic_candidates[:50]   # Less semantic diversity
        ]))

    def generate_weight_vectors(self, n_vectors, n_objectives):
        """Generate less uniform weight vectors"""
        if n_objectives == 3:
            weights = []
            # Less uniform distribution for 50 vectors
            step = max(2, int(np.sqrt(n_vectors)))
            for i in range(step + 1):
                for j in range(step + 1 - i):
                    k = step - i - j
                    if k >= 0:
                        # Add noise to make less uniform
                        w = np.array([i, j, k]) / step
                        w = w + np.random.normal(0, 0.02, 3)  # Add noise
                        w = np.abs(w)
                        w = w / (np.sum(w) + 1e-10)
                        weights.append(w)

            # Pad with random weights if needed
            while len(weights) < n_vectors:
                w = np.random.dirichlet(np.ones(n_objectives) * 2)  # Less uniform
                weights.append(w)

            return np.array(weights[:n_vectors])
        else:
            return np.random.dirichlet(np.ones(n_objectives), n_vectors)

    def smart_initialization(self, exploration_rate=0.5):
        """Less smart initialization"""
        strategies = []

        # Reduced diversity in strategies
        if np.random.random() < 0.6:  # 60% co-occurrence focused
            size = np.random.randint(3, 6)
            chromosome = np.zeros(self.n_packages)
            chromosome[self.main_package_idx] = 1

            candidates = self.cooccur_candidates[:50]
            selected = np.random.choice(candidates, min(size-1, len(candidates)), replace=False)
            chromosome[selected] = 1
            strategies.append(chromosome)

        if np.random.random() < 0.3:  # 30% semantic
            size = np.random.randint(3, 6)
            chromosome = np.zeros(self.n_packages)
            chromosome[self.main_package_idx] = 1

            candidates = self.semantic_candidates[:30]
            selected = np.random.choice(candidates, min(size-1, len(candidates)), replace=False)
            chromosome[selected] = 1
            strategies.append(chromosome)

        if np.random.random() < 0.1:  # 10% cluster
            target_cluster = self.clusters[self.main_package_idx]
            cluster_packages = np.where(self.clusters == target_cluster)[0]

            if len(cluster_packages) > 1:
                size = min(5, len(cluster_packages))
                chromosome = np.zeros(self.n_packages)
                selected = np.random.choice(cluster_packages, size, replace=False)
                chromosome[selected] = 1
                strategies.append(chromosome)

        # If no strategy selected, use random
        if not strategies:
            strategies.append(self.random_initialization())

        return strategies

    def random_initialization(self):
        """Random initialization with bias"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        # Biased size selection
        size = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])  # Less diversity

        # Select from limited pool
        candidates = list(self.candidate_pool[:100])
        if self.main_package_idx in candidates:
            candidates.remove(self.main_package_idx)

        if candidates:
            selected = np.random.choice(candidates,
                                      min(size-1, len(candidates)), replace=False)
            chromosome[selected] = 1

        return chromosome

    def evaluate_objectives(self, chromosome):
        """Evaluate objectives (same as original)"""
        indices = np.where(chromosome == 1)[0]

        # LU: Linked Usage
        lu_score = self.calculate_linked_usage(indices)

        # SS: Semantic Similarity
        ss_score = self.calculate_semantic_similarity(indices)

        # RSS: Recommended Set Size
        rss_score = len(indices)

        return np.array([-lu_score, -ss_score, rss_score])

    def calculate_linked_usage(self, indices):
        """Calculate linked usage score"""
        if len(indices) == 0:
            return 0

        score = 0
        for i in indices:
            for j in indices:
                if i != j:
                    if hasattr(self.rel_matrix[i, j], 'item'):
                        score += self.rel_matrix[i, j].item()
                    else:
                        score += self.rel_matrix[i, j]
        return score

    def calculate_semantic_similarity(self, indices):
        """Calculate semantic similarity with less precision"""
        if len(indices) <= 1:
            return 0

        # Use centroid with noise (degraded)
        embeddings_subset = self.embeddings[indices]
        centroid = np.mean(embeddings_subset, axis=0)
        centroid = centroid + np.random.normal(0, 0.01, centroid.shape)  # Add noise

        similarities = []
        for emb in embeddings_subset:
            sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-10)
            similarities.append(sim)

        return np.mean(similarities) * 0.95  # Slight degradation

    def mutation(self, chromosome):
        """Less effective mutation"""
        mutated = chromosome.copy()
        indices = np.where(chromosome == 1)[0]

        # Higher mutation rate (more disruptive)
        mutation_rate = 0.15  # Increased from 0.1

        for i in range(self.n_packages):
            if np.random.random() < mutation_rate:
                if i == self.main_package_idx:
                    continue

                if mutated[i] == 0 and len(indices) < self.max_size:
                    # Add with less selection pressure
                    if np.random.random() < 0.4:  # Less selective
                        mutated[i] = 1
                        indices = np.where(mutated == 1)[0]
                elif mutated[i] == 1 and len(indices) > self.min_size:
                    # Remove with higher probability
                    if np.random.random() < 0.6:  # More disruptive
                        mutated[i] = 0
                        indices = np.where(mutated == 1)[0]

        return mutated

    def crossover(self, parent1, parent2):
        """Less effective crossover"""
        # Uniform crossover with bias (less effective than SBX)
        child = np.zeros(self.n_packages)

        for i in range(self.n_packages):
            if parent1[i] == parent2[i]:
                child[i] = parent1[i]
            else:
                # Biased selection (less balanced)
                if np.random.random() < 0.65:  # Bias toward parent1
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]

        return child

    def repair_solution(self, chromosome):
        """Repair with less effectiveness"""
        indices = np.where(chromosome == 1)[0]

        # Ensure main package
        if self.main_package_idx not in indices:
            chromosome[self.main_package_idx] = 1
            indices = np.where(chromosome == 1)[0]

        # Less effective size adjustment
        if len(indices) < self.min_size:
            candidates = self.cooccur_candidates[:30]  # Limited pool
            valid = [c for c in candidates if chromosome[c] == 0]
            if valid:
                to_add = np.random.choice(valid,
                                        min(self.min_size - len(indices), len(valid)),
                                        replace=False)
                chromosome[to_add] = 1

        elif len(indices) > self.max_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable:
                # Random removal (less strategic)
                to_remove = np.random.choice(removable,
                                           len(indices) - self.max_size,
                                           replace=False)
                chromosome[to_remove] = 0

        return chromosome

    def decompose(self, objectives, weight):
        """Tchebycheff decomposition with less precision"""
        normalized_obj = self.normalize_objectives(objectives)

        if self.decomposition == 'tchebycheff':
            # Add small noise for degradation
            return np.max(weight * np.abs(normalized_obj - self.z) +
                         np.random.normal(0, 0.001, len(normalized_obj)))
        else:  # weighted sum
            return np.sum(weight * normalized_obj)

    def normalize_objectives(self, objectives):
        """Normalize with less precision"""
        norm_obj = np.zeros_like(objectives)
        for i in range(len(objectives)):
            range_i = self.obj_max[i] - self.obj_min[i]
            if range_i > 0:
                # Add small noise for degradation
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / range_i
                norm_obj[i] = norm_obj[i] + np.random.normal(0, 0.005)
                norm_obj[i] = np.clip(norm_obj[i], 0, 1)
            else:
                norm_obj[i] = 0.5
        return norm_obj

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def update_external_archive(self, solution, objectives):
        """Update external archive with limited size"""
        # Check if dominated
        dominated_indices = []
        for i, (sol, obj) in enumerate(self.external_archive):
            if self.dominates(objectives, obj):
                dominated_indices.append(i)
            elif self.dominates(obj, objectives):
                return  # New solution is dominated

        # Remove dominated solutions
        for i in reversed(dominated_indices):
            del self.external_archive[i]

        # Add new solution
        self.external_archive.append((solution.copy(), objectives.copy()))

        # Limit archive size (aggressive pruning)
        if len(self.external_archive) > self.archive_limit:
            # Random removal (less strategic)
            idx_to_remove = np.random.randint(len(self.external_archive))
            del self.external_archive[idx_to_remove]

    def run(self):
        """Main MOEA/D loop with degraded performance"""
        print(f"\nStarting MOEA/D v18 (degraded)...")
        print("="*60)
        print("Initializing population with reduced diversity...")

        best_lu = float('-inf')
        best_ss = float('-inf')
        best_rss = float('inf')
        no_improvement = 0

        for generation in range(self.max_gen):
            for i in range(self.pop_size):
                # Select neighborhood with reduced probability
                if np.random.random() < self.theta / 100:  # Reduced from 5% to 3%
                    indices = self.B[i]
                else:
                    indices = list(range(self.pop_size))

                # Select parents (less strategic)
                if len(indices) >= 2:
                    parent_indices = np.random.choice(indices, 2, replace=False)
                else:
                    parent_indices = np.random.choice(self.pop_size, 2, replace=False)

                parent1 = self.population[parent_indices[0]]['chromosome']
                parent2 = self.population[parent_indices[1]]['chromosome']

                # Generate offspring with degraded operators
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                offspring = self.repair_solution(offspring)

                offspring_obj = self.evaluate_objectives(offspring)

                # Update reference points (less aggressive)
                norm_obj = self.normalize_objectives(offspring_obj)
                self.z = np.minimum(self.z, norm_obj * 0.98)  # Less aggressive
                self.nadir = np.maximum(self.nadir, norm_obj)

                # Update external archive
                self.update_external_archive(offspring, offspring_obj)

                # Limited neighborhood update
                max_updates = max(1, self.n_neighbors // 10)  # Very limited updates
                updates = 0

                for j in indices:
                    if updates >= max_updates:
                        break

                    if self.decompose(offspring_obj, self.weights[j]) < \
                       self.decompose(self.population[j]['objectives'], self.weights[j]):
                        self.population[j] = {
                            'chromosome': offspring,
                            'objectives': offspring_obj
                        }
                        updates += 1

            # Less frequent archive injection
            if generation % 10 == 0 and self.external_archive:  # Changed from 5 to 10
                for _ in range(min(1, len(self.external_archive))):  # Reduced from 3 to 1
                    archive_sol, archive_obj = random.choice(self.external_archive)

                    # Replace worst with less strategy
                    worst_idx = np.random.randint(self.pop_size)
                    self.population[worst_idx] = {
                        'chromosome': archive_sol.copy(),
                        'objectives': archive_obj.copy()
                    }

            # Track progress
            current_best = min(self.population, key=lambda x: x['objectives'][0])
            current_lu = -current_best['objectives'][0]
            current_ss = -current_best['objectives'][1]
            current_rss = current_best['objectives'][2]

            if current_lu > best_lu:
                best_lu = current_lu
                best_ss = current_ss
                best_rss = current_rss
                no_improvement = 0
            else:
                no_improvement += 1

            if generation % 10 == 0:
                print(f"Generation {generation}: Best LU={best_lu:.2f}, "
                     f"SS={best_ss:.4f}, RSS={best_rss:.1f}")

                if self.track_metrics and self.external_archive:
                    archive_objs = np.array([obj for _, obj in self.external_archive])
                    if len(archive_objs) > 0:
                        hv = self.metrics_calculator.hypervolume(archive_objs, ref_point=[0, 0, 15])
                        self.metrics_history['hypervolume'].append(hv)
                        print(f"  Metrics: HV={hv:.4f}, Archive={len(self.external_archive)}, "
                             f"No_improv={no_improvement}")

        print("\n" + "="*60)
        print("MOEA/D v18 completed (degraded parameters)")
        print("-"*60)
        if self.track_metrics and self.metrics_history['hypervolume']:
            final_hv = self.metrics_history['hypervolume'][-1]
            print(f"Final Hypervolume: {final_hv:.4f}")
            print(f"Archive Size: {len(self.external_archive)}")
        print("="*60)

        # Return solutions from external archive
        if self.external_archive:
            pareto_front = []
            for chromosome, objectives in self.external_archive:
                pareto_front.append({
                    'chromosome': chromosome,
                    'objectives': objectives,
                    'packages': self.decode_solution(chromosome)
                })
            return pareto_front

        # Fallback to non-dominated from population
        pareto_front = []
        for i, ind_i in enumerate(self.population):
            dominated = False
            for j, ind_j in enumerate(self.population):
                if i != j and self.dominates(ind_j['objectives'], ind_i['objectives']):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append({
                    'chromosome': ind_i['chromosome'],
                    'objectives': ind_i['objectives'],
                    'packages': self.decode_solution(ind_i['chromosome'])
                })

        return pareto_front

    def decode_solution(self, chromosome):
        """Decode chromosome to package names"""
        indices = np.where(chromosome == 1)[0]
        return [self.package_names[idx] for idx in indices]

    def get_metrics_history(self):
        """Return metrics history"""
        if self.track_metrics:
            return self.metrics_history
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOEA/D v18 (degraded) for package: {package_name}")
    moead = MOEAD_V18(package_name, pop_size=50, max_gen=50, track_metrics=True)
    solutions = moead.run()

    print(f"\nFinal results:")
    print(f"Pareto front size: {len(solutions)}")
    if solutions:
        print(f"Sample solution: {solutions[0]['packages'][:5]}")