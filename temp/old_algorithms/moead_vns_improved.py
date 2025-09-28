"""
MOEA/D-AWA (Adaptive Weight Adjustment) for PyCommend VNS
Improved version based on 2024 literature review
Incorporates adaptive weights, external archive, and problem-specific operators
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
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOEAD_VNS_Improved:
    """
    Enhanced MOEA/D with Adaptive Weight Adjustment for PyCommend

    Key improvements:
    1. Adaptive weight vectors based on population distribution
    2. External archive for elite preservation
    3. Normalized decomposition to prevent numerical issues
    4. Dynamic parameter adaptation
    5. Problem-specific operators for discrete binary problem
    """

    def __init__(self, main_package, pop_size=100, n_neighbors=20, max_gen=50,
                 decomposition='tchebycheff', theta=5.0, track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.initial_theta = theta
        self.theta = theta
        self.n_objectives = 3
        self.track_metrics = track_metrics

        # Problem constraints
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        # External archive for elite preservation
        self.archive = []
        self.max_archive_size = pop_size * 2

        # Adaptive parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.update_rate = 0.3  # Probability of updating neighbor

        # Load data and initialize
        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.setup_moead()

        # Initialize metrics tracking
        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'spread': [],
                'diversity': []
            }

        print(f"Enhanced MOEA/D-AWA initialized for '{main_package}'")
        print(f"Using adaptive weight adjustment and external archive")
        if self.track_metrics:
            print("Quality metrics tracking: ENABLED")

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

        if self.main_package in self.package_names:
            self.main_package_idx = self.package_names.index(self.main_package)
        else:
            raise ValueError(f"Package '{self.main_package}' not found in data")

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic clustering and similarity structures"""
        print("Initializing semantic components...")

        # K-means clustering for semantic groups
        self.n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        # Find main package cluster
        self.main_cluster = self.cluster_labels[self.main_package_idx]
        self.cluster_members = np.where(self.cluster_labels == self.main_cluster)[0]
        print(f"Target package in cluster {self.main_cluster} with {len(self.cluster_members)} members")

    def compute_candidate_pools(self):
        """Pre-compute candidate pools for efficient initialization"""
        # Co-occurrence based candidates
        cooccur_scores = self.rel_matrix[self.main_package_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1][:200]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0]

        # Semantic similarity based candidates
        main_embedding = self.embeddings[self.main_package_idx]
        similarities = cosine_similarity([main_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        # Cluster-based candidates
        self.cluster_candidates = self.cluster_members[self.cluster_members != self.main_package_idx]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

    def setup_moead(self):
        """Initialize MOEA/D components with adaptive features"""
        # Generate initial weight vectors
        self.weights = self.generate_weight_vectors(self.pop_size)
        self.active_weights = np.ones(self.pop_size, dtype=bool)  # Track active subproblems

        # Calculate neighborhood
        self.neighbors = self.calculate_neighborhoods(self.weights)

        # Initialize reference points
        self.z = np.array([np.inf, np.inf, np.inf])  # Ideal point
        self.nadir = np.array([-np.inf, -np.inf, -np.inf])  # Nadir point

        # For normalization
        self.obj_min = np.array([np.inf, np.inf, np.inf])
        self.obj_max = np.array([-np.inf, -np.inf, -np.inf])

        # Utility tracking for adaptive weights
        self.utility = np.ones(self.pop_size)
        self.delta = np.zeros((self.pop_size, self.n_objectives))

        print(f"Generated {self.pop_size} weight vectors with adaptive adjustment")

    def generate_weight_vectors(self, n_vectors):
        """Generate uniformly distributed weight vectors using Das-Dennis method"""
        if self.n_objectives == 3:
            weights = []
            h = int(np.sqrt(n_vectors) * 1.5)

            for i in range(h + 1):
                for j in range(h + 1):
                    if i + j <= h:
                        w1 = i / h
                        w2 = j / h
                        w3 = 1 - w1 - w2
                        weights.append([w1, w2, w3])

            weights = np.array(weights)

            # Add random weights if needed
            while len(weights) < n_vectors:
                w = np.random.dirichlet(np.ones(self.n_objectives))
                weights = np.vstack([weights, w])

            # Limit to requested size
            if len(weights) > n_vectors:
                indices = np.random.choice(len(weights), n_vectors, replace=False)
                weights = weights[indices]

            return weights
        else:
            # For other dimensions, use random
            return np.random.dirichlet(np.ones(self.n_objectives), n_vectors)

    def calculate_neighborhoods(self, weights):
        """Calculate T closest weight vectors for each weight"""
        distances = cdist(weights, weights)
        neighborhoods = []

        for i in range(len(weights)):
            neighbors = np.argsort(distances[i])[:self.n_neighbors + 1]
            neighborhoods.append(neighbors[neighbors != i])

        return neighborhoods

    def evaluate_objectives(self, chromosome):
        """Evaluate three objectives with proper scaling"""
        indices = np.where(chromosome == 1)[0]

        if len(indices) == 0:
            return np.array([0, 0, self.max_size])

        # LU: Linked Usage (co-occurrence strength)
        lu_scores = self.rel_matrix[self.main_package_idx, indices].toarray().flatten()
        lu_score = np.sum(lu_scores)

        # Apply diminishing returns for very high values
        if lu_score > 5000:
            lu_score = 5000 + np.log1p(lu_score - 5000) * 100

        # SS: Semantic Similarity
        if len(indices) > 0:
            # Get similarities for each selected package
            direct_similarities = []
            for idx in indices:
                sim = self.sim_matrix[self.main_package_idx, idx]
                direct_similarities.append(sim)
            direct_similarities = np.array(direct_similarities)

            # Internal coherence
            if len(indices) > 1:
                selected_embeddings = self.embeddings[indices]
                centroid = np.mean(selected_embeddings, axis=0)
                coherence_scores = cosine_similarity(selected_embeddings, [centroid]).flatten()
                internal_coherence = np.mean(coherence_scores)
            else:
                internal_coherence = 0.5

            # Weighted combination
            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weights = weights / weights.sum()
            weighted_sim = np.average(direct_similarities, weights=weights)
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        # RSS: Recommended Set Size (with penalty for extremes)
        rss_score = len(indices)
        if len(indices) < self.ideal_size:
            rss_score += (self.ideal_size - len(indices)) * 0.5
        elif len(indices) > self.ideal_size * 1.5:
            rss_score += (len(indices) - self.ideal_size * 1.5) * 0.3

        return np.array([-lu_score, -ss_score, rss_score])

    def normalize_objectives(self, objectives):
        """Normalize objectives to [0, 1] for stable decomposition"""
        # Avoid division by zero
        range_vals = self.obj_max - self.obj_min
        range_vals[range_vals == 0] = 1.0

        norm_obj = (objectives - self.obj_min) / range_vals
        return np.clip(norm_obj, 0, 1)

    def decompose(self, objectives, weight):
        """Enhanced decomposition with normalization"""
        # Normalize objectives for stable computation
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(self.z)

        if self.decomposition == 'weighted_sum':
            return np.sum(weight * norm_obj)

        elif self.decomposition == 'tchebycheff':
            # Tchebycheff approach with stability check
            diff = np.abs(norm_obj - norm_z)
            weighted_diff = weight * diff
            # Add small epsilon to prevent exact zeros
            weighted_diff = np.where(weight > 0, weighted_diff, -1e10)
            return np.max(weighted_diff)

        elif self.decomposition == 'pbi':
            # Penalty-based boundary intersection
            diff = norm_obj - norm_z
            d1 = np.abs(np.dot(diff, weight)) / (np.linalg.norm(weight) + 1e-10)
            d2 = np.linalg.norm(diff - d1 * weight / (np.linalg.norm(weight) + 1e-10))
            return d1 + self.theta * d2

        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def smart_initialization(self, strategy='adaptive'):
        """Enhanced initialization with problem knowledge"""
        chromosome = np.zeros(self.n_packages, dtype=np.int8)

        # Adaptive size based on strategy
        if strategy == 'small':
            size = random.randint(2, 4)
        elif strategy == 'medium':
            size = random.randint(5, 8)
        elif strategy == 'large':
            size = random.randint(9, 12)
        elif strategy == 'adaptive':
            # Use distribution that favors ideal size
            size = int(np.random.normal(self.ideal_size, 2))
            size = np.clip(size, self.min_size, self.max_size)
        else:
            size = random.randint(3, 10)

        # Combine different candidate sources
        candidates = []

        # Prioritize co-occurrence candidates
        if len(self.cooccur_candidates) > 0:
            n_cooccur = min(size // 2 + 1, len(self.cooccur_candidates))
            cooccur_scores = self.rel_matrix[self.main_package_idx, self.cooccur_candidates].toarray().flatten()
            if cooccur_scores.sum() > 0:
                probs = cooccur_scores / cooccur_scores.sum()
                selected_cooccur = np.random.choice(self.cooccur_candidates, n_cooccur,
                                                   replace=False, p=probs)
                candidates.extend(selected_cooccur)

        # Add semantic candidates
        if len(self.semantic_candidates) > 0:
            n_semantic = min(size // 3, len(self.semantic_candidates))
            candidates.extend(self.semantic_candidates[:n_semantic])

        # Fill with cluster candidates if needed
        if len(candidates) < size and len(self.cluster_candidates) > 0:
            remaining = size - len(candidates)
            cluster_sample = np.random.choice(self.cluster_candidates,
                                            min(remaining, len(self.cluster_candidates)),
                                            replace=False)
            candidates.extend(cluster_sample)

        # Ensure we have enough candidates
        candidates = list(set(candidates))[:size]

        if len(candidates) > 0:
            chromosome[candidates] = 1

        return chromosome

    def differential_evolution(self, target, indices):
        """Enhanced DE operator for discrete problem"""
        if len(indices) < 3:
            return target.copy()

        # Select three different solutions
        r = np.random.choice(indices, 3, replace=False)
        x1 = self.population[r[0]]['chromosome']
        x2 = self.population[r[1]]['chromosome']
        x3 = self.population[r[2]]['chromosome']

        # Create donor vector
        donor = target.copy()

        # Differential mutation for binary problem
        diff = (x2 != x3).astype(int)
        mutation_mask = np.random.random(self.n_packages) < 0.5

        # Apply differential
        donor = np.where(mutation_mask & (diff == 1), x1, donor)

        # Crossover
        crossover_mask = np.random.random(self.n_packages) < self.crossover_rate
        offspring = np.where(crossover_mask, donor, target)

        return offspring

    def mutation(self, chromosome):
        """Adaptive mutation with problem knowledge"""
        mutated = chromosome.copy()

        for i in range(self.n_packages):
            if i == self.main_package_idx:
                continue

            if random.random() < self.mutation_rate:
                # Smart mutation: prefer high-value packages
                if mutated[i] == 0:
                    # Consider adding based on value
                    cooccur = self.rel_matrix[self.main_package_idx, i]
                    prob_add = min(0.5, cooccur / 1000) if cooccur > 0 else 0.05
                    if random.random() < prob_add:
                        mutated[i] = 1
                else:
                    # Remove with lower probability for good packages
                    mutated[i] = 1 - mutated[i]

        return mutated

    def repair_solution(self, chromosome):
        """Repair solution to satisfy constraints"""
        active_indices = np.where(chromosome == 1)[0]

        # Fix size constraints
        if len(active_indices) < self.min_size:
            # Add packages with highest value
            candidates = np.where(chromosome == 0)[0]
            scores = self.rel_matrix[self.main_package_idx, candidates].toarray().flatten()
            best_candidates = candidates[np.argsort(scores)[::-1]]
            n_add = self.min_size - len(active_indices)
            chromosome[best_candidates[:n_add]] = 1

        elif len(active_indices) > self.max_size:
            # Remove packages with lowest contribution
            scores = self.rel_matrix[self.main_package_idx, active_indices].toarray().flatten()
            worst_indices = active_indices[np.argsort(scores)]
            n_remove = len(active_indices) - self.max_size
            chromosome[worst_indices[:n_remove]] = 0

        return chromosome

    def update_reference_point(self, objectives):
        """Adaptive reference point update"""
        # Update ideal point (best values seen)
        self.z = np.minimum(self.z, objectives)

        # Update bounds for normalization
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

    def update_nadir_point(self, objectives):
        """Update nadir point for normalization"""
        self.nadir = np.maximum(self.nadir, objectives)

    def update_archive(self, solution):
        """Maintain external archive of non-dominated solutions"""
        # Check if solution is non-dominated
        dominated = False
        to_remove = []

        for i, archived in enumerate(self.archive):
            if self.dominates(archived['objectives'], solution['objectives']):
                dominated = True
                break
            elif self.dominates(solution['objectives'], archived['objectives']):
                to_remove.append(i)

        if not dominated:
            # Remove dominated solutions
            for i in reversed(to_remove):
                del self.archive[i]

            # Add new solution
            self.archive.append(solution.copy())

            # Limit archive size using crowding distance
            if len(self.archive) > self.max_archive_size:
                self.reduce_archive()

    def reduce_archive(self):
        """Reduce archive size using crowding distance"""
        # Calculate crowding distance
        objectives = np.array([sol['objectives'] for sol in self.archive])
        n = len(self.archive)

        crowding_distance = np.zeros(n)

        for obj_idx in range(self.n_objectives):
            sorted_indices = np.argsort(objectives[:, obj_idx])
            crowding_distance[sorted_indices[0]] = np.inf
            crowding_distance[sorted_indices[-1]] = np.inf

            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = objectives[sorted_indices[i + 1], obj_idx] - objectives[sorted_indices[i - 1], obj_idx]
                    crowding_distance[sorted_indices[i]] += distance / obj_range

        # Keep solutions with highest crowding distance
        keep_indices = np.argsort(crowding_distance)[::-1][:self.max_archive_size]
        self.archive = [self.archive[i] for i in keep_indices]

    def adapt_weights(self, generation):
        """Adapt weight vectors based on population distribution"""
        if generation % 10 != 0 or generation == 0:
            return

        # Calculate utility of each subproblem
        for i in range(self.pop_size):
            if generation > 10:
                improvement = np.linalg.norm(self.delta[i])
                self.utility[i] = 1.0 + improvement

        # Identify poorly performing subproblems
        mean_utility = np.mean(self.utility)
        poor_subproblems = self.utility < mean_utility * 0.5

        # Adjust weights for poor subproblems
        if np.any(poor_subproblems):
            # Find crowded regions in objective space
            objectives = np.array([self.population[i]['objectives'] for i in range(self.pop_size)])

            for i in np.where(poor_subproblems)[0]:
                # Move weight toward less crowded region
                distances = cdist([self.weights[i]], self.weights)[0]
                sparse_region = np.argmax(distances)

                # Adjust weight toward sparse region
                direction = self.weights[sparse_region] - self.weights[i]
                self.weights[i] += 0.1 * direction
                self.weights[i] = np.clip(self.weights[i], 0, 1)
                self.weights[i] /= np.sum(self.weights[i])

            # Recalculate neighborhoods after weight adjustment
            self.neighbors = self.calculate_neighborhoods(self.weights)

    def update_parameters(self, generation):
        """Dynamically adjust algorithm parameters"""
        progress = generation / self.max_gen

        # Reduce theta over time for better convergence
        self.theta = self.initial_theta * (1 - 0.5 * progress)

        # Adjust mutation rate
        if generation > 0 and generation % 5 == 0:
            # Calculate diversity
            objectives = np.array([self.population[i]['objectives'] for i in range(self.pop_size)])
            diversity = np.std(objectives, axis=0).mean()

            # Increase mutation if diversity is low
            if diversity < 0.1:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            else:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

    def calculate_metrics(self, population):
        """Calculate quality metrics for current population"""
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
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(pareto_objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(pareto_objectives)
        metrics['spread'] = self.metrics_calculator.spread(pareto_objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(pareto_objectives)

        return metrics

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (for minimization)"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def get_metrics_history(self):
        """Return the metrics history if tracking was enabled"""
        return self.metrics_history if self.track_metrics else None

    def run(self):
        """Main MOEA/D loop with enhancements"""
        print(f"\nStarting Enhanced MOEA/D-AWA...")
        print("="*60)

        # Initialize population
        self.population = []
        strategies = ['small', 'medium', 'large', 'adaptive']

        print("Initializing population with diverse strategies...")
        for i in range(self.pop_size):
            strategy = strategies[i % len(strategies)]
            chromosome = self.smart_initialization(strategy)
            chromosome = self.repair_solution(chromosome)
            objectives = self.evaluate_objectives(chromosome)

            self.population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

            self.update_reference_point(objectives)
            self.update_nadir_point(objectives)
            self.update_archive({'chromosome': chromosome, 'objectives': objectives})

        print(f"Population initialized with {len(self.population)} solutions")
        print(f"Initial archive size: {len(self.archive)}")

        # Evolution loop
        for generation in range(self.max_gen):
            # Store old objectives for utility calculation
            old_objectives = [sol['objectives'].copy() for sol in self.population]

            # Update parameters dynamically
            self.update_parameters(generation)

            # Evolve population
            for i in range(self.pop_size):
                # Select mating pool (neighbors or random)
                if random.random() < 0.9:
                    indices = self.neighbors[i]
                else:
                    indices = list(range(self.pop_size))

                # Generate offspring
                offspring = self.differential_evolution(
                    self.population[i]['chromosome'], indices)
                offspring = self.mutation(offspring)
                offspring = self.repair_solution(offspring)

                offspring_obj = self.evaluate_objectives(offspring)

                # Update reference points
                self.update_reference_point(offspring_obj)
                self.update_nadir_point(offspring_obj)

                # Update archive
                self.update_archive({'chromosome': offspring, 'objectives': offspring_obj})

                # Update neighboring solutions
                c = 0
                max_updates = max(2, int(self.n_neighbors * self.update_rate))

                for j in indices:
                    if c >= max_updates:
                        break

                    # Check if offspring is better for subproblem j
                    if self.decompose(offspring_obj, self.weights[j]) < \
                       self.decompose(self.population[j]['objectives'], self.weights[j]):
                        self.population[j] = {
                            'chromosome': offspring,
                            'objectives': offspring_obj
                        }
                        c += 1

            # Update utility for adaptive weights
            for i in range(self.pop_size):
                self.delta[i] = old_objectives[i] - self.population[i]['objectives']

            # Adapt weight vectors
            self.adapt_weights(generation)

            # Calculate metrics if tracking
            if self.track_metrics:
                metrics = self.calculate_metrics(self.population)
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics:
                            self.metrics_history[key].append(metrics[key])

            # Progress report
            if generation % 10 == 0:
                best_lu = min([ind['objectives'][0] for ind in self.population])
                best_ss = min([ind['objectives'][1] for ind in self.population])
                best_rss = min([ind['objectives'][2] for ind in self.population])

                print(f"Generation {generation}: Best LU={-best_lu:.2f}, "
                      f"SS={-best_ss:.4f}, RSS={best_rss:.1f}")
                print(f"  Archive size: {len(self.archive)}, "
                      f"Mutation rate: {self.mutation_rate:.3f}, Theta: {self.theta:.2f}")

                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, "
                          f"Spacing={metrics.get('spacing', 0):.4f}, "
                          f"Diversity={metrics.get('diversity', 0):.4f}")

        # Final results from archive
        pareto_front = self.get_pareto_front()

        # Print final metrics
        if self.track_metrics and self.metrics_history['hypervolume']:
            print("\n" + "="*60)
            print("FINAL METRICS SUMMARY")
            print("-"*60)
            print(f"Final Hypervolume: {self.metrics_history['hypervolume'][-1]:.4f}")
            print(f"Final Spacing: {self.metrics_history['spacing'][-1]:.4f}")
            print(f"Final Diversity: {self.metrics_history['diversity'][-1]:.4f}")

            if len(self.metrics_history['hypervolume']) > 1:
                initial_hv = self.metrics_history['hypervolume'][0]
                final_hv = self.metrics_history['hypervolume'][-1]
                if initial_hv > 0:
                    improvement = (final_hv - initial_hv) / initial_hv * 100
                    print(f"Hypervolume Improvement: {improvement:+.1f}%")
            print("="*60)

        return pareto_front

    def get_pareto_front(self):
        """Extract non-dominated solutions from archive and population"""
        # Combine archive and current population
        all_solutions = self.archive.copy() + self.population

        # Find non-dominated solutions
        pareto_front = []
        for i, sol_i in enumerate(all_solutions):
            dominated = False
            for j, sol_j in enumerate(all_solutions):
                if i != j and self.dominates(sol_j['objectives'], sol_i['objectives']):
                    dominated = True
                    break

            if not dominated:
                # Check if already in pareto front (avoid duplicates)
                duplicate = False
                for existing in pareto_front:
                    if np.array_equal(existing['chromosome'], sol_i['chromosome']):
                        duplicate = True
                        break

                if not duplicate:
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
                'objectives': sol['objectives']
            })

        # Sort by multiple criteria
        recommendations = sorted(recommendations,
                               key=lambda x: (-x['linked_usage'], -x['semantic_similarity'], x['size']))

        return recommendations


def main():
    """Test the improved implementation"""
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    # Check if metrics tracking is requested
    track_metrics = '--metrics' in sys.argv or '--track-metrics' in sys.argv

    print(f"Enhanced MOEA/D-AWA - Library Recommendation for '{package_name}'")
    print("="*60)

    moead = MOEAD_VNS_Improved(package_name, pop_size=100, n_neighbors=20, max_gen=50,
                               decomposition='tchebycheff', track_metrics=track_metrics)

    solutions = moead.run()

    print(f"\nFound {len(solutions)} Pareto-optimal solutions")
    print("="*60)

    # Show top recommendations
    recommendations = moead.get_recommendations(solutions)
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. Size {rec['size']}: {', '.join(rec['packages'][:10])}")
        print(f"   LU={rec['linked_usage']:.1f}, SS={rec['semantic_similarity']:.3f}")


if __name__ == '__main__':
    main()