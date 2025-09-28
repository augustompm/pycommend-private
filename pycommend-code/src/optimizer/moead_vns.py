"""
MOEA/D for PyCommend VNS - Multi-Objective Library Recommendation
Based on Zhang & Li (2007) IEEE Transactions on Evolutionary Computation
Aligned with ICVNS 2025 presentation
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

# Add path for quality metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOEAD_VNS:
    """
    MOEA/D for library recommendation with 3 objectives:
    1. LU (Linked Usage): Maximize co-occurrence in real projects
    2. SS (Semantic Similarity): Maximize weighted topical coherence
    3. RSS (Recommended Set Size): Minimize set size

    Reference: Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary
    algorithm based on decomposition. IEEE Transactions on evolutionary computation, 11(6), 712-731.
    """

    def __init__(self, main_package, pop_size=100, n_neighbors=20, max_gen=50,
                 decomposition='tchebycheff', theta=5.0, track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.theta = theta
        self.n_objectives = 3
        self.track_metrics = track_metrics

        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.setup_moead()

        # Initialize quality metrics if tracking is enabled
        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'spread': [],
                'diversity': []
            }

        print(f"MOEA/D-VNS initialized for '{main_package}'")
        print(f"Using decomposition: {decomposition}")
        print(f"Objectives: LU (Linked Usage), SS (Semantic Similarity), RSS (Set Size)")
        if self.track_metrics:
            print("Quality metrics tracking: ENABLED (Hypervolume, Spacing, Spread, Diversity)")

    def load_all_data(self):
        """Load all required data matrices"""
        print("Loading data matrices...")

        with open('data/package_relationships_10k.pkl', 'rb') as f:
            relationships_data = pickle.load(f)
        self.rel_matrix = relationships_data['matrix']
        self.package_names = relationships_data['package_names']
        self.n_packages = len(self.package_names)

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            similarity_data = pickle.load(f)
        self.sim_matrix = similarity_data['similarity_matrix']

        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        self.embeddings = embeddings_data['embeddings']

        if self.main_package not in self.package_names:
            raise ValueError(f"Package '{self.main_package}' not found in dataset")
        self.main_package_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic clustering for better initialization"""
        print("Initializing semantic components...")

        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.sum(self.cluster_labels == self.target_cluster)
        print(f"Target package in cluster {self.target_cluster} with {cluster_members} members")

    def compute_candidate_pools(self):
        """Precompute candidate pools for smart initialization"""
        main_idx = self.main_package_idx

        cooccur_scores = self.rel_matrix[main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0][:200]

        target_embedding = self.embeddings[main_idx]
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        self.cluster_candidates = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = self.cluster_candidates[self.cluster_candidates != main_idx]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

    def setup_moead(self):
        """Setup MOEA/D specific components"""
        self.generate_weight_vectors()
        self.compute_neighborhoods()
        self.z = np.full(self.n_objectives, np.inf)
        self.nadir = np.full(self.n_objectives, -np.inf)

    def generate_weight_vectors(self):
        """Generate uniformly distributed weight vectors using Das-Dennis method"""
        if self.n_objectives == 3:
            self.weights = self.generate_uniform_weights_3d()
        else:
            self.weights = np.random.rand(self.pop_size, self.n_objectives)
            self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)

        print(f"Generated {len(self.weights)} weight vectors")

    def generate_uniform_weights_3d(self):
        """Generate uniform weight vectors for 3 objectives"""
        weights = []
        H = int(np.ceil(np.power(self.pop_size, 1.0/(self.n_objectives-1))))

        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                if k >= 0:
                    w = np.array([i/H, j/H, k/H])
                    if np.sum(w) > 0:
                        weights.append(w)

        weights = np.array(weights)

        if len(weights) > self.pop_size:
            indices = np.random.choice(len(weights), self.pop_size, replace=False)
            weights = weights[indices]
        elif len(weights) < self.pop_size:
            extra = self.pop_size - len(weights)
            random_weights = np.random.rand(extra, self.n_objectives)
            random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)
            weights = np.vstack([weights, random_weights])

        return weights

    def compute_neighborhoods(self):
        """Compute T-nearest neighbors for each weight vector"""
        distances = cdist(self.weights, self.weights)
        self.neighbors = np.zeros((self.pop_size, self.n_neighbors), dtype=int)

        for i in range(self.pop_size):
            sorted_indices = np.argsort(distances[i])
            self.neighbors[i] = sorted_indices[1:self.n_neighbors + 1]

    def evaluate_objectives(self, chromosome):
        """
        Evaluate 3 objectives as per presentation:
        LU: Linked Usage (maximize co-occurrence)
        SS: Semantic Similarity (maximize topical coherence)
        RSS: Recommended Set Size (minimize)
        """
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        threshold = np.percentile(self.rel_matrix[main_idx].data, 75) if self.rel_matrix[main_idx].data.size > 0 else 1.0
        strong_links = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            if len(indices) > 1:
                selected_embeddings = self.embeddings[indices]
                centroid = np.mean(selected_embeddings, axis=0)
                coherence_scores = cosine_similarity(selected_embeddings, [centroid]).flatten()
                internal_coherence = np.mean(coherence_scores)
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        rss_score = len(indices)
        if len(indices) < self.ideal_size:
            rss_score += (self.ideal_size - len(indices)) * 0.5
        elif len(indices) > self.ideal_size * 1.5:
            rss_score += (len(indices) - self.ideal_size * 1.5) * 0.3

        return np.array([-lu_score, -ss_score, rss_score])

    def decompose(self, objectives, weight, z=None):
        """Decomposition function for scalarization"""
        if z is None:
            z = self.z

        if self.decomposition == 'weighted_sum':
            return np.sum(weight * objectives)

        elif self.decomposition == 'tchebycheff':
            return np.max(weight * np.abs(objectives - z))

        elif self.decomposition == 'pbi':
            d1 = np.abs(np.dot(objectives - z, weight)) / np.linalg.norm(weight)
            d2 = np.linalg.norm((objectives - z) - d1 * weight / np.linalg.norm(weight))
            return d1 + self.theta * d2

        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition}")

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solutions using domain knowledge - more aggressive"""
        chromosome = np.zeros(self.n_packages, dtype=np.int8)

        if strategy == 'small':
            size = random.randint(3, 5)
        elif strategy == 'medium':
            size = random.randint(6, 9)
        elif strategy == 'large':
            size = random.randint(10, 15)
        else:
            size = random.randint(5, 12)

        if strategy == 'cooccur' and len(self.cooccur_candidates) > 0:
            # Use top candidates with higher probability
            weights = self.rel_matrix[self.main_package_idx, self.cooccur_candidates].toarray().flatten()
            if weights.sum() > 0:
                # Boost weights for better candidates
                weights = np.power(weights, 0.5)  # Square root to boost selection
                weights = weights / weights.sum()
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False, p=weights)
            else:
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False)

        elif strategy == 'semantic' and len(self.semantic_candidates) > 0:
            selected = self.semantic_candidates[:min(size, len(self.semantic_candidates))]

        elif strategy == 'cluster' and len(self.cluster_candidates) > 0:
            selected = np.random.choice(self.cluster_candidates,
                                      min(size, len(self.cluster_candidates)),
                                      replace=False)

        else:
            candidates = []

            # More aggressive: use more top candidates
            if len(self.cooccur_candidates) > 0:
                candidates.extend(self.cooccur_candidates[:min(size*2//3, len(self.cooccur_candidates))])
            if len(self.semantic_candidates) > 0:
                candidates.extend(self.semantic_candidates[:min(size//2, len(self.semantic_candidates))])
            if len(self.cluster_candidates) > 0:
                sample_size = min(size//3, len(self.cluster_candidates))
                candidates.extend(np.random.choice(self.cluster_candidates, sample_size, replace=False))

            if len(candidates) > 0:
                candidates = list(set(candidates))
                selected = np.random.choice(candidates, min(size, len(candidates)), replace=False)
            else:
                valid_indices = list(range(self.n_packages))
                valid_indices.remove(self.main_package_idx)
                selected = np.random.choice(valid_indices, size, replace=False)

        chromosome[selected] = 1
        return chromosome

    def differential_evolution(self, target, indices):
        """Enhanced Differential Evolution operator for binary representation"""
        # Select best individual from neighbors for guidance
        best_idx = min(indices, key=lambda x: self.decompose(self.population[x]['objectives'], self.weights[x]))
        r1, r2 = np.random.choice([idx for idx in indices if idx != best_idx], 2, replace=False)

        x_best = self.population[best_idx]['chromosome']
        x_r1 = self.population[r1]['chromosome']
        x_r2 = self.population[r2]['chromosome']

        mutant = target.copy()

        # Higher crossover rate for better exploration
        cr = 0.95
        # Scaling factor for difference vector
        f = 0.8

        for i in range(self.n_packages):
            if random.random() < cr:
                # Use best individual with higher probability
                if random.random() < 0.7 and x_best[i] == 1:
                    mutant[i] = 1
                elif x_r1[i] != x_r2[i]:
                    # Apply difference vector with scaling
                    if random.random() < f:
                        mutant[i] = x_r1[i]
                    else:
                        mutant[i] = target[i]

        return mutant

    def repair_solution(self, chromosome):
        """Repair solution to satisfy constraints"""
        active = np.where(chromosome == 1)[0]

        if len(active) < self.min_size:
            candidates = np.where(chromosome == 0)[0]
            candidates = [c for c in candidates if c != self.main_package_idx]
            if candidates:
                n_add = min(self.min_size - len(active), len(candidates))
                add_indices = np.random.choice(candidates, n_add, replace=False)
                chromosome[add_indices] = 1

        elif len(active) > self.max_size:
            n_remove = len(active) - self.max_size
            remove_indices = np.random.choice(active, n_remove, replace=False)
            chromosome[remove_indices] = 0

        return chromosome

    def mutation(self, chromosome):
        """Polynomial mutation for binary representation"""
        mutated = chromosome.copy()
        mutation_rate = 1.0 / self.n_packages

        for i in range(self.n_packages):
            if i == self.main_package_idx:
                continue

            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]

        return self.repair_solution(mutated)

    def update_reference_point(self, objectives):
        """Update ideal reference point z*"""
        self.z = np.minimum(self.z, objectives)

    def update_nadir_point(self, objectives):
        """Update nadir point for normalization"""
        self.nadir = np.maximum(self.nadir, objectives)

    def calculate_metrics(self, population):
        """
        Calculate quality metrics for current population
        """
        # Get all objective vectors
        all_objectives = np.array([ind['objectives'] for ind in population])

        # Find non-dominated solutions
        non_dominated_mask = []
        for i in range(len(all_objectives)):
            is_dominated = False
            for j in range(len(all_objectives)):
                if i != j and self.dominates(all_objectives[j], all_objectives[i]):
                    is_dominated = True
                    break
            non_dominated_mask.append(not is_dominated)

        pareto_objectives = all_objectives[non_dominated_mask]

        if len(pareto_objectives) == 0:
            return None

        metrics = {}

        # Calculate Hypervolume
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(pareto_objectives)

        # Calculate Spacing
        metrics['spacing'] = self.metrics_calculator.spacing(pareto_objectives)

        # Calculate Spread
        metrics['spread'] = self.metrics_calculator.spread(pareto_objectives)

        # Calculate Diversity
        metrics['diversity'] = self.metrics_calculator.diversity(pareto_objectives)

        return metrics

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (for minimization)"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def get_metrics_history(self):
        """Return the metrics history if tracking was enabled"""
        if self.track_metrics:
            return self.metrics_history
        else:
            return None

    def run(self):
        """Main MOEA/D loop"""
        print(f"\nStarting MOEA/D for PyCommend VNS...")
        print("="*60)

        self.population = []
        # More cooccur-focused strategies for better performance
        strategies = ['cooccur', 'cooccur', 'hybrid', 'cooccur', 'medium', 'large']

        print("Initializing population with enhanced strategies...")
        for i in range(self.pop_size):
            strategy = strategies[i % len(strategies)]
            chromosome = self.smart_initialization(strategy)
            objectives = self.evaluate_objectives(chromosome)

            self.population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

            self.update_reference_point(objectives)
            self.update_nadir_point(objectives)

        print(f"Population initialized with {len(self.population)} solutions")

        for generation in range(self.max_gen):
            for i in range(self.pop_size):
                if random.random() < 0.9:
                    indices = self.neighbors[i]
                else:
                    indices = list(range(self.pop_size))

                offspring = self.differential_evolution(
                    self.population[i]['chromosome'], indices)
                offspring = self.mutation(offspring)
                offspring = self.repair_solution(offspring)

                offspring_obj = self.evaluate_objectives(offspring)

                self.update_reference_point(offspring_obj)
                self.update_nadir_point(offspring_obj)

                c = 0
                for j in indices:
                    if c >= self.n_neighbors * 0.1:
                        break

                    if self.decompose(offspring_obj, self.weights[j]) < \
                       self.decompose(self.population[j]['objectives'], self.weights[j]):
                        self.population[j] = {
                            'chromosome': offspring,
                            'objectives': offspring_obj
                        }
                        c += 1

            # Calculate metrics if tracking is enabled
            if self.track_metrics:
                metrics = self.calculate_metrics(self.population)
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics:
                            self.metrics_history[key].append(metrics[key])

            if generation % 10 == 0:
                best_lu = min([ind['objectives'][0] for ind in self.population])
                best_ss = min([ind['objectives'][1] for ind in self.population])
                best_rss = min([ind['objectives'][2] for ind in self.population])
                print(f"Generation {generation}: Best LU={-best_lu:.2f}, "
                      f"SS={-best_ss:.4f}, RSS={best_rss:.1f}")

                # Print metrics if tracking
                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, "
                          f"Spacing={metrics.get('spacing', 0):.4f}, "
                          f"Spread={metrics.get('spread', 0):.4f}, "
                          f"Diversity={metrics.get('diversity', 0):.4f}")

        # Print final metrics summary if tracking
        pareto_front = self.get_pareto_front()

        if self.track_metrics and self.metrics_history['hypervolume']:
            print("\n" + "="*60)
            print("FINAL METRICS SUMMARY")
            print("-"*60)
            print(f"Final Hypervolume: {self.metrics_history['hypervolume'][-1]:.4f}")
            print(f"Final Spacing: {self.metrics_history['spacing'][-1]:.4f}")
            print(f"Final Spread: {self.metrics_history['spread'][-1]:.4f}")
            print(f"Final Diversity: {self.metrics_history['diversity'][-1]:.4f}")

            # Calculate improvement
            if len(self.metrics_history['hypervolume']) > 1:
                initial_hv = self.metrics_history['hypervolume'][0]
                final_hv = self.metrics_history['hypervolume'][-1]
                if initial_hv > 0:
                    improvement = (final_hv - initial_hv) / initial_hv * 100
                    print(f"Hypervolume Improvement: {improvement:.1f}%")
            print("="*60)

        # Return in same format as MOVNS
        return self.get_recommendations(pareto_front)

    def get_pareto_front(self):
        """Extract non-dominated solutions"""
        pareto_front = []

        for i, sol_i in enumerate(self.population):
            dominated = False
            for j, sol_j in enumerate(self.population):
                if i != j and self.dominates(sol_j['objectives'], sol_i['objectives']):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(sol_i)

        return pareto_front

    def get_recommendations(self, solutions):
        """Extract package recommendations from solutions"""
        recommendations = []

        for sol in solutions:
            indices = np.where(sol['chromosome'] == 1)[0]
            packages = [self.package_names[idx] for idx in indices]

            lu_score = -sol['objectives'][0]
            ss_score = -sol['objectives'][1]
            rss_score = sol['objectives'][2]

            recommendations.append({
                'packages': packages,
                'size': len(packages),
                'linked_usage': lu_score,
                'semantic_similarity': ss_score,
                'objectives': {
                    'linked_usage': lu_score,
                    'semantic_similarity': ss_score,
                    'set_size': rss_score
                }
            })

        recommendations = sorted(recommendations, key=lambda x: (x['size'], -x['linked_usage']))

        return recommendations


def main():
    """Test the implementation"""
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    # Check if metrics tracking is requested
    track_metrics = '--metrics' in sys.argv or '--track-metrics' in sys.argv

    print(f"MOEA/D-VNS - Library Recommendation for '{package_name}'")
    print("="*60)

    moead = MOEAD_VNS(package_name, pop_size=100, n_neighbors=20, max_gen=50,
                      decomposition='tchebycheff', track_metrics=track_metrics)

    solutions = moead.run()

    print(f"\nFound {len(solutions)} Pareto-optimal solutions")
    print("="*60)

    recommendations = moead.get_recommendations(solutions)

    shown_sizes = set()
    for rec in recommendations[:10]:
        size = rec['size']
        if size not in shown_sizes:
            print(f"\nSize {size} recommendation:")
            print(f"  Packages: {', '.join(rec['packages'])}")
            print(f"  LU (Linked Usage): {rec['linked_usage']:.2f}")
            print(f"  SS (Semantic Similarity): {rec['semantic_similarity']:.4f}")
            shown_sizes.add(size)

            if len(shown_sizes) >= 5:
                break

    return recommendations


if __name__ == '__main__':
    main()