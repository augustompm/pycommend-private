"""
NSGA-II for PyCommend - Multi-Objective Library Recommendation
Note: This is standard NSGA-II without VNS components (historical naming)
"""

import numpy as np
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys
import os
import time

# Add path for quality metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class NSGA2:
    """
    NSGA-II for library recommendation with 3 objectives:
    1. LU (Linked Usage): Maximize co-occurrence in real projects
    2. SS (Semantic Similarity): Maximize weighted topical coherence
    3. RSS (Recommended Set Size): Minimize set size
    """

    def __init__(self, main_package, pop_size=100, max_gen=50, track_metrics=False):
        self.main_package = main_package
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5  # Ideal recommendation size
        self.track_metrics = track_metrics

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()

        # Initialize quality metrics if tracking is enabled
        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.reference_set = None
            self.metrics_history = {
                'hypervolume': [],
                'igd_plus': [],
                'spacing': [],
                'diversity': []
            }

        print(f"PyCommend VNS initialized for '{main_package}'")
        print(f"Using 3 objectives: LU (Linked Usage), SS (Semantic Similarity), RSS (Set Size)")
        if self.track_metrics:
            print("Quality metrics tracking: ENABLED (Hypervolume, IGD+, Spacing, Diversity)")

    def load_all_data(self):
        """Load all required data matrices"""
        print("Loading data matrices...")

        # Load co-occurrence matrix (for LU objective)
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            relationships_data = pickle.load(f)
        self.rel_matrix = relationships_data['matrix']
        self.package_names = relationships_data['package_names']
        self.n_packages = len(self.package_names)

        # Load similarity matrix (for SS objective)
        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            similarity_data = pickle.load(f)
        self.sim_matrix = similarity_data['similarity_matrix']

        # Load SBERT embeddings (for semantic clustering and SS)
        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        self.embeddings = embeddings_data['embeddings']

        # Find main package index
        if self.main_package not in self.package_names:
            raise ValueError(f"Package '{self.main_package}' not found in dataset")
        self.main_package_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic clustering for better initialization"""
        print("Initializing semantic components...")

        # K-means clustering on embeddings
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.sum(self.cluster_labels == self.target_cluster)
        print(f"Target package in cluster {self.target_cluster} with {cluster_members} members")

    def compute_candidate_pools(self):
        """Precompute candidate pools for smart initialization"""
        main_idx = self.main_package_idx

        # Pool 1: Top co-occurring packages
        cooccur_scores = self.rel_matrix[main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0][:200]

        # Pool 2: Semantically similar packages
        target_embedding = self.embeddings[main_idx]
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        # Pool 3: Same cluster packages
        self.cluster_candidates = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = self.cluster_candidates[self.cluster_candidates != main_idx]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

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

        # Objective 1: LU (Linked Usage) - Maximize
        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        # Add bonus for highly connected packages
        threshold = np.percentile(self.rel_matrix[main_idx].data, 75) if self.rel_matrix[main_idx].data.size > 0 else 1.0
        strong_links = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        # Objective 2: SS (Weighted Semantic Similarity) - Maximize
        if len(indices) > 0:
            # Method 1: Direct similarity to main package
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            # Method 2: Coherence within the set
            if len(indices) > 1:
                selected_embeddings = self.embeddings[indices]
                centroid = np.mean(selected_embeddings, axis=0)
                coherence_scores = cosine_similarity(selected_embeddings, [centroid]).flatten()
                internal_coherence = np.mean(coherence_scores)
            else:
                internal_coherence = 0.5

            # Weighted combination
            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        # Objective 3: RSS (Recommended Set Size) - Minimize
        # Prefer sizes close to ideal_size
        rss_score = len(indices)
        if len(indices) < self.ideal_size:
            rss_score += (self.ideal_size - len(indices)) * 0.5  # Penalty for too small
        elif len(indices) > self.ideal_size * 1.5:
            rss_score += (len(indices) - self.ideal_size * 1.5) * 0.3  # Penalty for too large

        # Return objectives (negate LU and SS for minimization in NSGA-II)
        return np.array([-lu_score, -ss_score, rss_score])

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solutions using domain knowledge"""
        chromosome = np.zeros(self.n_packages, dtype=np.int8)

        # Determine size
        if strategy == 'small':
            size = random.randint(2, 4)
        elif strategy == 'medium':
            size = random.randint(5, 7)
        elif strategy == 'large':
            size = random.randint(8, 12)
        else:
            size = random.randint(3, 10)

        # Select packages based on strategy
        if strategy == 'cooccur' and len(self.cooccur_candidates) > 0:
            # Use co-occurrence based selection
            weights = self.rel_matrix[self.main_package_idx, self.cooccur_candidates].toarray().flatten()
            if weights.sum() > 0:
                weights = weights / weights.sum()
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False, p=weights)
            else:
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False)

        elif strategy == 'semantic' and len(self.semantic_candidates) > 0:
            # Use semantic similarity
            selected = self.semantic_candidates[:min(size, len(self.semantic_candidates))]

        elif strategy == 'cluster' and len(self.cluster_candidates) > 0:
            # Use cluster members
            selected = np.random.choice(self.cluster_candidates,
                                      min(size, len(self.cluster_candidates)),
                                      replace=False)

        else:  # hybrid or fallback
            candidates = []

            # Mix different strategies
            if len(self.cooccur_candidates) > 0:
                candidates.extend(self.cooccur_candidates[:size//2])
            if len(self.semantic_candidates) > 0:
                candidates.extend(self.semantic_candidates[:size//3])
            if len(self.cluster_candidates) > 0:
                sample_size = min(size//4, len(self.cluster_candidates))
                candidates.extend(np.random.choice(self.cluster_candidates, sample_size, replace=False))

            if len(candidates) > 0:
                candidates = list(set(candidates))
                selected = np.random.choice(candidates, min(size, len(candidates)), replace=False)
            else:
                # Fallback to random
                valid_indices = list(range(self.n_packages))
                valid_indices.remove(self.main_package_idx)
                selected = np.random.choice(valid_indices, size, replace=False)

        chromosome[selected] = 1
        return chromosome

    def initialize_population(self):
        """Initialize population with diverse strategies"""
        print("Initializing population...")
        population = []

        # Use different strategies for diversity
        strategies = {
            'small': int(0.2 * self.pop_size),     # Small sets (2-4)
            'medium': int(0.3 * self.pop_size),    # Medium sets (5-7)
            'large': int(0.2 * self.pop_size),     # Large sets (8-12)
            'cooccur': int(0.1 * self.pop_size),   # Co-occurrence based
            'semantic': int(0.1 * self.pop_size),  # Semantic based
            'hybrid': int(0.1 * self.pop_size)     # Mixed strategy
        }

        for strategy, count in strategies.items():
            for _ in range(count):
                chromosome = self.smart_initialization(strategy)
                objectives = self.evaluate_objectives(chromosome)
                population.append({
                    'chromosome': chromosome,
                    'objectives': objectives,
                    'rank': None,
                    'crowding_distance': 0
                })

        # Fill remaining with hybrid
        remaining = self.pop_size - len(population)
        for _ in range(remaining):
            chromosome = self.smart_initialization('hybrid')
            objectives = self.evaluate_objectives(chromosome)
            population.append({
                'chromosome': chromosome,
                'objectives': objectives,
                'rank': None,
                'crowding_distance': 0
            })

        print(f"Population initialized with {len(population)} individuals")
        return population

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (for minimization)"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def fast_non_dominated_sort(self, P):
        """Fast non-dominated sorting"""
        n = len(P)
        S = [[] for _ in range(n)]
        N = [0] * n
        F = []
        rank = [0] * n

        for p in range(n):
            for q in range(n):
                if self.dominates(P[p]['objectives'], P[q]['objectives']):
                    S[p].append(q)
                elif self.dominates(P[q]['objectives'], P[p]['objectives']):
                    N[p] += 1

            if N[p] == 0:
                rank[p] = 0
                F.append([])
                F[0].append(p)

        i = 0
        while F[i]:
            Q = []
            for p in F[i]:
                for q in S[p]:
                    N[q] -= 1
                    if N[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)

        for idx, r in enumerate(rank):
            P[idx]['rank'] = r

        return F[:-1]

    def crowding_distance_assignment(self, I):
        """Assign crowding distance to individuals"""
        l = len(I)
        for i in range(l):
            I[i]['crowding_distance'] = 0

        n_obj = len(I[0]['objectives'])

        for m in range(n_obj):
            I = sorted(I, key=lambda x: x['objectives'][m])

            I[0]['crowding_distance'] = float('inf')
            I[l-1]['crowding_distance'] = float('inf')

            f_min = I[0]['objectives'][m]
            f_max = I[l-1]['objectives'][m]

            if f_max - f_min == 0:
                continue

            for i in range(1, l-1):
                distance = I[i+1]['objectives'][m] - I[i-1]['objectives'][m]
                I[i]['crowding_distance'] += distance / (f_max - f_min)

    def tournament_selection(self, population):
        """Binary tournament selection"""
        if not population:
            raise ValueError("Cannot select from empty population")

        if len(population) == 1:
            return population[0]

        p1 = random.choice(population)
        p2 = random.choice(population)

        if 'rank' not in p1 or 'rank' not in p2 or p1['rank'] is None or p2['rank'] is None:
            return p1 if random.random() < 0.5 else p2

        if p1['rank'] < p2['rank']:
            return p1
        elif p2['rank'] < p1['rank']:
            return p2
        else:
            if p1['crowding_distance'] > p2['crowding_distance']:
                return p1
            else:
                return p2

    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        child = parent1['chromosome'].copy()
        mask = np.random.random(self.n_packages) < 0.5
        child[mask] = parent2['chromosome'][mask]

        # Ensure minimum size
        if np.sum(child) < self.min_size:
            candidates = np.where(child == 0)[0]
            if len(candidates) > 0:
                add_indices = np.random.choice(candidates,
                                             min(self.min_size - np.sum(child), len(candidates)),
                                             replace=False)
                child[add_indices] = 1

        # Ensure maximum size
        if np.sum(child) > self.max_size:
            active = np.where(child == 1)[0]
            remove_indices = np.random.choice(active,
                                            np.sum(child) - self.max_size,
                                            replace=False)
            child[remove_indices] = 0

        return child

    def generate_reference_set(self, n_points=100):
        """
        Generate a reference set for IGD+ calculation
        Creates a well-distributed set of achievable but challenging points
        """
        # Use initial population to understand the objective space
        temp_pop = []
        for _ in range(50):
            chromosome = self.smart_initialization('hybrid')
            objectives = self.evaluate_objectives(chromosome)
            temp_pop.append(objectives)

        temp_pop = np.array(temp_pop)

        # Calculate bounds based on actual achievable objectives
        # LU and SS are negative (maximize), RSS is positive (minimize)
        lu_best = np.min(temp_pop[:, 0]) * 1.2  # 20% better than best found
        lu_worst = np.max(temp_pop[:, 0]) * 0.8
        ss_best = np.min(temp_pop[:, 1]) * 1.2  # 20% better
        ss_worst = np.max(temp_pop[:, 1]) * 0.8
        rss_best = np.min(temp_pop[:, 2]) * 0.8  # 20% better (smaller)
        rss_worst = np.max(temp_pop[:, 2]) * 1.2

        # Generate reference points using a grid
        n_per_dim = int(np.cbrt(n_points)) + 1
        lu_range = np.linspace(lu_best, lu_worst, n_per_dim)
        ss_range = np.linspace(ss_best, ss_worst, n_per_dim)
        rss_range = np.linspace(rss_best, rss_worst, n_per_dim)

        ref_points = []
        for lu in lu_range:
            for ss in ss_range:
                for rss in rss_range:
                    ref_points.append([lu, ss, rss])

        ref_points = np.array(ref_points)

        # Add some extreme points representing ideal solutions
        ideal_points = [
            [lu_best, ss_best, rss_best],  # Best in all objectives
            [lu_best, ss_worst, rss_best],  # Trade-off points
            [lu_worst, ss_best, rss_best],
            [lu_best, ss_best, rss_worst],
        ]

        ref_points = np.vstack([ref_points, ideal_points])

        # Keep only non-dominated points
        non_dominated_mask = []
        for i in range(len(ref_points)):
            is_dominated = False
            for j in range(len(ref_points)):
                if i != j and self.dominates(ref_points[j], ref_points[i]):
                    is_dominated = True
                    break
            non_dominated_mask.append(not is_dominated)

        ref_points = ref_points[non_dominated_mask]

        # Limit size
        if len(ref_points) > n_points:
            # Keep diverse subset
            indices = np.random.choice(len(ref_points), n_points, replace=False)
            ref_points = ref_points[indices]

        self.reference_set = ref_points

    def update_reference_set(self, population):
        """
        Update reference set with better solutions found
        """
        if self.reference_set is None:
            # Initialize with uniform reference set
            self.generate_reference_set()
            return

        # Get current Pareto front
        fronts = self.fast_non_dominated_sort(population)
        if fronts and fronts[0]:
            current_pareto = [population[i] for i in fronts[0]]
            current_objectives = np.array([ind['objectives'] for ind in current_pareto])

            # Update reference set with better solutions
            combined = np.vstack([self.reference_set, current_objectives])

            # Keep only non-dominated solutions
            non_dominated_mask = []
            for i in range(len(combined)):
                is_dominated = False
                for j in range(len(combined)):
                    if i != j and self.dominates(combined[j], combined[i]):
                        is_dominated = True
                        break
                non_dominated_mask.append(not is_dominated)

            # Update reference set only if we found better solutions
            new_ref = combined[non_dominated_mask]
            if len(new_ref) > 0:
                self.reference_set = new_ref

                # Limit size to control computation
                if len(self.reference_set) > 200:
                    # Keep diverse subset using crowding distance
                    indices = np.random.choice(len(self.reference_set), 200, replace=False)
                    self.reference_set = self.reference_set[indices]

    def calculate_metrics(self, population):
        """
        Calculate quality metrics for current population
        """
        fronts = self.fast_non_dominated_sort(population)
        if not fronts or not fronts[0]:
            return None

        # Get Pareto front
        pareto_indices = fronts[0]
        pareto_objectives = np.array([population[i]['objectives'] for i in pareto_indices])

        metrics = {}

        # Calculate Hypervolume
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(pareto_objectives)

        # Calculate IGD+ if reference set exists
        if self.reference_set is not None and len(self.reference_set) > 0:
            metrics['igd_plus'] = self.metrics_calculator.igd_plus(pareto_objectives, self.reference_set)
        else:
            metrics['igd_plus'] = None

        # Calculate Spacing
        metrics['spacing'] = self.metrics_calculator.spacing(pareto_objectives)

        # Calculate Diversity
        metrics['diversity'] = self.metrics_calculator.diversity(pareto_objectives)

        return metrics

    def mutation(self, chromosome):
        """Bit-flip mutation with domain knowledge"""
        mutated = chromosome.copy()
        mutation_rate = 0.1

        for i in range(self.n_packages):
            if i == self.main_package_idx:
                continue

            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]

        # Ensure constraints
        if np.sum(mutated) < self.min_size:
            candidates = np.where(mutated == 0)[0]
            candidates = [c for c in candidates if c != self.main_package_idx]
            if candidates:
                add_indices = np.random.choice(candidates,
                                             min(self.min_size - np.sum(mutated), len(candidates)),
                                             replace=False)
                mutated[add_indices] = 1

        if np.sum(mutated) > self.max_size:
            active = np.where(mutated == 1)[0]
            remove_indices = np.random.choice(active,
                                            np.sum(mutated) - self.max_size,
                                            replace=False)
            mutated[remove_indices] = 0

        return mutated

    def run(self):
        """Main NSGA-II loop"""
        print(f"\nStarting NSGA-II for PyCommend...")
        print("="*60)

        population = self.initialize_population()
        best_objectives_history = []

        # Initialize reference set if tracking metrics
        if self.track_metrics:
            self.update_reference_set(population)

        for generation in range(self.max_gen):
            # Create offspring
            offspring_population = []
            for _ in range(self.pop_size):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                child_chromosome = self.crossover(parent1, parent2)
                child_chromosome = self.mutation(child_chromosome)

                objectives = self.evaluate_objectives(child_chromosome)

                offspring_population.append({
                    'chromosome': child_chromosome,
                    'objectives': objectives,
                    'rank': None,
                    'crowding_distance': 0
                })

            # Combine populations
            population = population + offspring_population
            fronts = self.fast_non_dominated_sort(population)

            # Environmental selection
            new_population = []
            for front_idx, front in enumerate(fronts):
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    remaining = self.pop_size - len(new_population)
                    if remaining > 0:
                        front_individuals = [population[i] for i in front]
                        self.crowding_distance_assignment(front_individuals)
                        front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)
                        new_population.extend(front_individuals[:remaining])
                    break

            # Ensure population size
            if len(new_population) < self.pop_size:
                while len(new_population) < self.pop_size:
                    new_individual = self.smart_initialization('hybrid')
                    objectives = self.evaluate_objectives(new_individual)
                    new_population.append({
                        'chromosome': new_individual,
                        'objectives': objectives,
                        'rank': None,
                        'crowding_distance': 0
                    })

            population = new_population[:self.pop_size]

            # Calculate metrics if tracking is enabled
            if self.track_metrics:
                metrics = self.calculate_metrics(population)
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

                    # Update reference set periodically
                    if generation % 5 == 0:
                        self.update_reference_set(population)

            # Print progress
            if generation % 10 == 0 and population:
                current_fronts = self.fast_non_dominated_sort(population)
                if current_fronts and len(current_fronts) > 0 and len(current_fronts[0]) > 0:
                    pareto_front = [population[i] for i in current_fronts[0]]
                    if pareto_front:
                        best = min(pareto_front, key=lambda x: x['objectives'][0])
                        print(f"Generation {generation}: Pareto size={len(pareto_front)}")
                        print(f"  Best: LU={-best['objectives'][0]:.2f}, "
                              f"SS={-best['objectives'][1]:.4f}, RSS={best['objectives'][2]:.1f}")

                        # Print metrics if tracking
                        if self.track_metrics and metrics:
                            print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, ", end="")
                            if metrics.get('igd_plus') is not None:
                                print(f"IGD+={metrics['igd_plus']:.4f}, ", end="")
                            print(f"Spacing={metrics.get('spacing', 0):.4f}, "
                                  f"Diversity={metrics.get('diversity', 0):.4f}")

                        best_objectives_history.append(best['objectives'])

        # Get final Pareto front
        final_fronts = self.fast_non_dominated_sort(population)
        if final_fronts and final_fronts[0]:
            pareto_solutions = [population[i] for i in final_fronts[0]]
        else:
            pareto_solutions = population[:min(10, len(population))]

        # Print final metrics summary if tracking
        if self.track_metrics and self.metrics_history['hypervolume']:
            print("\n" + "="*60)
            print("FINAL METRICS SUMMARY")
            print("-"*60)
            print(f"Final Hypervolume: {self.metrics_history['hypervolume'][-1]:.4f}")
            if self.metrics_history['igd_plus'] and self.metrics_history['igd_plus'][-1] is not None:
                print(f"Final IGD+: {self.metrics_history['igd_plus'][-1]:.4f}")
                # Calculate improvement
                if len(self.metrics_history['igd_plus']) > 1:
                    initial = self.metrics_history['igd_plus'][0]
                    final = self.metrics_history['igd_plus'][-1]
                    if initial > 0:
                        improvement = (initial - final) / initial * 100
                        print(f"IGD+ Improvement: {improvement:.1f}%")
            print(f"Final Spacing: {self.metrics_history['spacing'][-1]:.4f}")
            print(f"Final Diversity: {self.metrics_history['diversity'][-1]:.4f}")
            print("="*60)

        return pareto_solutions

    def get_metrics_history(self):
        """Return the metrics history if tracking was enabled"""
        if self.track_metrics:
            return self.metrics_history
        else:
            return None

    def get_recommendations(self, solutions):
        """Extract package recommendations from solutions"""
        recommendations = []

        for sol in solutions:
            indices = np.where(sol['chromosome'] == 1)[0]
            packages = [self.package_names[idx] for idx in indices]

            # Calculate actual objective values
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

        # Sort by different criteria
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

    print(f"PyCommend VNS - Library Recommendation for '{package_name}'")
    print("="*60)

    # Run NSGA-II
    nsga2 = NSGA2_VNS(package_name, pop_size=100, max_gen=50, track_metrics=track_metrics)
    solutions = nsga2.run()

    print(f"\nFound {len(solutions)} Pareto-optimal solutions")
    print("="*60)

    # Get recommendations
    recommendations = nsga2.get_recommendations(solutions)

    # Show different sizes of recommendations
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