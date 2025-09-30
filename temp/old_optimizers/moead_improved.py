"""
MOEA/D Improved for PyCommend VNS - Enhanced Convergence
Based on Zhang & Li (2007) with parameter optimization for better convergence
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


class MOEAD_VNS_Improved:
    """
    Improved MOEA/D with better convergence properties
    Changes:
    1. Increased update rate (20% instead of 10%)
    2. Elitism mechanism to preserve best solutions
    3. Adaptive acceptance threshold
    4. Better initialization
    """

    def __init__(self, main_package, pop_size=100, n_neighbors=20, max_gen=50,
                 decomposition='tchebycheff', theta=5.0, track_metrics=False,
                 update_rate=0.2, elitism_rate=0.1, acceptance_threshold=0.99):
        self.main_package = main_package
        self.pop_size = pop_size
        self.n_neighbors = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.theta = theta
        self.n_objectives = 3
        self.track_metrics = track_metrics

        self.update_rate = update_rate
        self.elitism_rate = elitism_rate
        self.acceptance_threshold = acceptance_threshold

        self.elite_size = max(1, int(pop_size * elitism_rate))
        self.elite_population = []

        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

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

        print(f"MOEA/D-VNS Improved initialized for '{main_package}'")
        print(f"Update rate: {update_rate}, Elitism: {elitism_rate}, Acceptance: {acceptance_threshold}")

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
        """Generate uniformly distributed weight vectors"""
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
                        weights.append(w / np.sum(w))

        while len(weights) < self.pop_size:
            w = np.random.rand(self.n_objectives)
            weights.append(w / np.sum(w))

        return np.array(weights[:self.pop_size])

    def compute_neighborhoods(self):
        """Compute neighborhood structure based on weight vectors"""
        distances = cdist(self.weights, self.weights, 'euclidean')
        self.neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        self.B = self.neighbors

    def evaluate_objectives(self, chromosome):
        """Evaluate 3 objectives"""
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
                mean_embeddings = np.mean([self.embeddings[idx] for idx in indices], axis=0)
                internal_coherence = np.mean([
                    cosine_similarity([self.embeddings[idx]], [mean_embeddings])[0, 0]
                    for idx in indices
                ])
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        rss_score = len(indices)
        size_penalty = abs(len(indices) - self.ideal_size) * 0.05
        rss_score = rss_score * (1 + size_penalty)

        return np.array([-lu_score, -ss_score, rss_score])

    def decompose(self, objectives, weight):
        """Decomposition function for scalarization"""
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
        """Initialize a chromosome with domain knowledge"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(self.min_size, min(10, self.max_size))

        if strategy == 'cooccurrence' and len(self.cooccur_candidates) > 0:
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            selected = np.random.choice(self.cooccur_candidates[:50], n_select, replace=False)
            chromosome[selected] = 1

        elif strategy == 'semantic' and len(self.semantic_candidates) > 0:
            n_select = min(target_size - 1, len(self.semantic_candidates))
            selected = np.random.choice(self.semantic_candidates[:50], n_select, replace=False)
            chromosome[selected] = 1

        elif strategy == 'cluster' and len(self.cluster_candidates) > 0:
            n_select = min(target_size - 1, len(self.cluster_candidates))
            selected = np.random.choice(self.cluster_candidates, n_select, replace=False)
            chromosome[selected] = 1

        else:
            n_cooccur = min((target_size - 1) // 2, len(self.cooccur_candidates))
            n_semantic = min(target_size - 1 - n_cooccur, len(self.semantic_candidates))

            if n_cooccur > 0:
                selected_cooccur = np.random.choice(self.cooccur_candidates[:30], n_cooccur, replace=False)
                chromosome[selected_cooccur] = 1

            if n_semantic > 0:
                selected_semantic = np.random.choice(self.semantic_candidates[:30], n_semantic, replace=False)
                chromosome[selected_semantic] = 1

        return chromosome

    def differential_evolution(self, target, indices):
        """Differential Evolution operator"""
        if len(indices) < 3:
            return target.copy()

        candidates = random.sample(list(indices), min(3, len(indices)))
        r1, r2, r3 = candidates[:3]

        F = 0.5 + 0.3 * random.random()
        CR = 0.7 + 0.2 * random.random()

        mutant = self.population[r1]['chromosome'].copy()
        diff = self.population[r2]['chromosome'] - self.population[r3]['chromosome']
        mutant = mutant + F * diff
        mutant = np.clip(mutant, 0, 1)

        trial = target.copy()
        for j in range(len(target)):
            if random.random() < CR:
                trial[j] = mutant[j]

        trial = (trial > 0.5).astype(int)
        return trial

    def mutation(self, chromosome):
        """Mutation operator"""
        mutated = chromosome.copy()
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or random.random() > 0.3:
            return mutated

        mutation_type = random.choice(['add', 'remove', 'swap'])

        if mutation_type == 'add' and len(indices) < self.max_size:
            candidates = self.cooccur_candidates[:30] if len(self.cooccur_candidates) > 0 else []
            valid_candidates = [c for c in candidates if mutated[c] == 0]
            if valid_candidates:
                new_idx = random.choice(valid_candidates)
                mutated[new_idx] = 1

        elif mutation_type == 'remove' and len(indices) > self.min_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable:
                remove_idx = random.choice(removable)
                mutated[remove_idx] = 0

        elif mutation_type == 'swap':
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable and len(self.semantic_candidates) > 0:
                remove_idx = random.choice(removable)
                candidates = [c for c in self.semantic_candidates[:30] if mutated[c] == 0]
                if candidates:
                    add_idx = random.choice(candidates)
                    mutated[remove_idx] = 0
                    mutated[add_idx] = 1

        return mutated

    def repair_solution(self, chromosome):
        """Repair invalid solutions"""
        indices = np.where(chromosome == 1)[0]

        if self.main_package_idx not in indices:
            chromosome[self.main_package_idx] = 1
            indices = np.where(chromosome == 1)[0]

        if len(indices) > self.max_size:
            scores = [self.rel_matrix[self.main_package_idx, idx] for idx in indices]
            sorted_indices = [x for _, x in sorted(zip(scores, indices), reverse=True)]
            keep = sorted_indices[:self.max_size]
            chromosome = np.zeros(self.n_packages)
            chromosome[keep] = 1

        elif len(indices) < self.min_size:
            n_add = self.min_size - len(indices)
            candidates = self.cooccur_candidates[:50] if len(self.cooccur_candidates) > 0 else []
            valid_candidates = [c for c in candidates if chromosome[c] == 0]
            if len(valid_candidates) >= n_add:
                add_indices = random.sample(valid_candidates, n_add)
                chromosome[add_indices] = 1

        return chromosome

    def update_reference_point(self, objectives):
        """Update ideal point"""
        self.z = np.minimum(self.z, objectives)

    def update_nadir_point(self, objectives):
        """Update nadir point"""
        self.nadir = np.maximum(self.nadir, objectives)

    def update_elite(self):
        """Maintain elite population"""
        all_solutions = self.population + self.elite_population

        all_solutions.sort(key=lambda x: self.decompose(x['objectives'],
                                                       np.ones(self.n_objectives)/self.n_objectives))

        self.elite_population = all_solutions[:self.elite_size]

    def calculate_metrics(self, population):
        """Calculate quality metrics"""
        if not self.track_metrics or not population:
            return None

        objectives = np.array([ind['objectives'] for ind in population])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(objectives)
        metrics['spread'] = self.metrics_calculator.spread(objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(objectives)

        return metrics

    def run(self):
        """Main MOEA/D loop with improved convergence"""
        print(f"\nStarting MOEA/D Improved for PyCommend VNS...")
        print("="*60)

        print("Initializing population with enhanced strategies...")
        self.population = []
        strategies = ['cooccurrence', 'semantic', 'cluster', 'hybrid']

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

        best_hv = 0
        stagnation_counter = 0

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
                max_updates = max(1, int(len(indices) * self.update_rate))

                for j in indices:
                    if c >= max_updates:
                        break

                    old_fitness = self.decompose(self.population[j]['objectives'], self.weights[j])
                    new_fitness = self.decompose(offspring_obj, self.weights[j])

                    if new_fitness < old_fitness * self.acceptance_threshold:
                        self.population[j] = {
                            'chromosome': offspring,
                            'objectives': offspring_obj
                        }
                        c += 1

            self.update_elite()

            if generation % 5 == 0 and self.elite_population:
                for i in range(min(2, len(self.elite_population))):
                    worst_idx = np.argmax([self.decompose(ind['objectives'], self.weights[j])
                                         for j, ind in enumerate(self.population)])
                    self.population[worst_idx] = self.elite_population[i].copy()

            if self.track_metrics:
                metrics = self.calculate_metrics(self.population)
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics:
                            self.metrics_history[key].append(metrics[key])

                    current_hv = metrics.get('hypervolume', 0)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            if generation % 10 == 0:
                best_lu = min([ind['objectives'][0] for ind in self.population])
                best_ss = min([ind['objectives'][1] for ind in self.population])
                best_rss = min([ind['objectives'][2] for ind in self.population])
                print(f"Generation {generation}: Best LU={-best_lu:.2f}, "
                      f"SS={-best_ss:.4f}, RSS={best_rss:.1f}")

                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, "
                          f"Spacing={metrics.get('spacing', 0):.4f}")

            if stagnation_counter > 10:
                self.acceptance_threshold = max(0.95, self.acceptance_threshold - 0.01)
                self.update_rate = min(0.3, self.update_rate + 0.02)
                stagnation_counter = 0

        pareto_front = self.get_pareto_front()

        if self.track_metrics and self.metrics_history['hypervolume']:
            print("\n" + "="*60)
            print("FINAL METRICS SUMMARY")
            print("-"*60)
            print(f"Final Hypervolume: {self.metrics_history['hypervolume'][-1]:.4f}")
            print(f"Final Spacing: {self.metrics_history['spacing'][-1]:.4f}")
            print(f"Final Spread: {self.metrics_history['spread'][-1]:.4f}")
            print(f"Final Diversity: {self.metrics_history['diversity'][-1]:.4f}")

            if len(self.metrics_history['hypervolume']) > 1:
                initial_hv = self.metrics_history['hypervolume'][0]
                final_hv = self.metrics_history['hypervolume'][-1]
                improvement = ((final_hv - initial_hv) / (initial_hv + 1e-10)) * 100
                print(f"Hypervolume Improvement: {improvement:.1f}%")
            print("="*60)

        return pareto_front

    def get_pareto_front(self):
        """Extract Pareto front"""
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

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

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

    track_metrics = '--metrics' in sys.argv or '--track-metrics' in sys.argv

    print(f"Testing {package_name} recommendation...")

    moead = MOEAD_VNS_Improved(package_name, pop_size=100, max_gen=50,
                               decomposition='tchebycheff', track_metrics=track_metrics,
                               update_rate=0.25, elitism_rate=0.1, acceptance_threshold=0.98)

    solutions = moead.run()

    print(f"\nFound {len(solutions)} Pareto optimal solutions")

    if solutions:
        print("\nTop 3 solutions by different criteria:")

        solutions_by_lu = sorted(solutions, key=lambda x: x['objectives'][0])[:3]
        print("\nBy Linked Usage (LU):")
        for i, sol in enumerate(solutions_by_lu, 1):
            print(f"  {i}. {sol['packages'][:5]} (LU={-sol['objectives'][0]:.2f})")

        solutions_by_ss = sorted(solutions, key=lambda x: x['objectives'][1])[:3]
        print("\nBy Semantic Similarity (SS):")
        for i, sol in enumerate(solutions_by_ss, 1):
            print(f"  {i}. {sol['packages'][:5]} (SS={-sol['objectives'][1]:.4f})")

        solutions_by_rss = sorted(solutions, key=lambda x: x['objectives'][2])[:3]
        print("\nBy Set Size (RSS):")
        for i, sol in enumerate(solutions_by_rss, 1):
            print(f"  {i}. {sol['packages'][:5]} (Size={sol['objectives'][2]:.0f})")