"""
MOEA/D-VNS with Proper Normalization
Based on Zhang & Li (2007) with normalization for objectives with different scales
This version ensures positive convergence through proper objective normalization
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


class MOEAD_VNS_Normalized:
    """
    MOEA/D with proper normalization for convergence
    Key improvement: Normalizes objectives to [0,1] before decomposition
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

        self.external_archive = []
        self.archive_limit = 100

        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

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

        print(f"MOEA/D-VNS Normalized initialized for '{main_package}'")
        print(f"Using decomposition: {decomposition} with normalization")
        print(f"Objective ranges properly configured for convergence")

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
        """Initialize semantic clustering"""
        print("Initializing semantic components...")

        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        self.threshold = 1.0

    def compute_candidate_pools(self):
        """Precompute candidate pools"""
        main_idx = self.main_package_idx

        cooccur_scores = self.rel_matrix[main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0][:200]

        target_embedding = self.embeddings[main_idx]
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        self.cluster_candidates = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = self.cluster_candidates[self.cluster_candidates != main_idx]

        if self.rel_matrix[main_idx].data.size > 0:
            self.threshold = np.percentile(self.rel_matrix[main_idx].data, 75)

    def setup_moead(self):
        """Setup MOEA/D components"""
        self.generate_weight_vectors()
        self.compute_neighborhoods()
        self.z = np.array([1.0, 1.0, 0.0])
        self.nadir = np.array([0.0, 0.0, 1.0])

    def generate_weight_vectors(self):
        """Generate uniformly distributed weight vectors"""
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

        self.weights = np.array(weights[:self.pop_size])
        print(f"Generated {len(self.weights)} weight vectors")

    def compute_neighborhoods(self):
        """Compute neighborhood based on weight vectors"""
        distances = cdist(self.weights, self.weights, 'euclidean')
        self.neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]

    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0, 1] range
        Critical for proper decomposition with different scales
        """
        norm_obj = np.zeros_like(objectives)

        for i in range(len(objectives)):
            if self.obj_max[i] - self.obj_min[i] != 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / (self.obj_max[i] - self.obj_min[i])
            else:
                norm_obj[i] = 0.5

        norm_obj = np.clip(norm_obj, 0, 1)
        return norm_obj

    def update_bounds(self, objectives):
        """
        Update objective bounds dynamically
        """
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

    def evaluate_objectives(self, chromosome):
        """Evaluate 3 objectives"""
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        strong_links = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > self.threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            if len(indices) > 1:
                internal_coherence = np.mean(direct_similarities) * 0.8
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

        objectives = np.array([-lu_score, -ss_score, rss_score])

        self.update_bounds(objectives)

        return objectives

    def decompose(self, objectives, weight):
        """
        Decomposition function with normalized objectives
        This is the key fix for convergence
        """
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.z.copy()

        if self.decomposition == 'weighted_sum':
            return np.sum(weight * norm_obj)

        elif self.decomposition == 'tchebycheff':
            return np.max(weight * np.abs(norm_obj - norm_z))

        elif self.decomposition == 'pbi':
            d1 = np.abs(np.dot(norm_obj - norm_z, weight)) / np.linalg.norm(weight)
            d2 = np.linalg.norm((norm_obj - norm_z) - d1 * weight / np.linalg.norm(weight))
            return d1 + self.theta * d2

        else:
            return np.sum(weight * norm_obj)

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solution with domain knowledge"""
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

    def mutation(self, chromosome):
        """Mutation operator"""
        mutated = chromosome.copy()
        indices = np.where(chromosome == 1)[0]

        if random.random() < 0.3:
            mutation_type = random.choice(['add', 'remove', 'swap'])

            if mutation_type == 'add' and len(indices) < self.max_size:
                candidates = self.cooccur_candidates[:30] if len(self.cooccur_candidates) > 0 else []
                valid = [c for c in candidates if mutated[c] == 0]
                if valid:
                    mutated[random.choice(valid)] = 1

            elif mutation_type == 'remove' and len(indices) > self.min_size:
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    mutated[random.choice(removable)] = 0

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

    def crossover(self, parent1, parent2):
        """Crossover operator"""
        offspring = np.zeros(self.n_packages)

        for i in range(self.n_packages):
            if random.random() < 0.5:
                offspring[i] = parent1[i]
            else:
                offspring[i] = parent2[i]

        return offspring

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

    def update_external_archive(self, solution, objectives):
        """Update external archive with quality-based criterion"""
        for i, (sol, obj) in enumerate(list(self.external_archive)):
            if self.dominates(objectives, obj):
                self.external_archive[i] = (solution, objectives)
                return True
            elif self.dominates(obj, objectives):
                return False

        self.external_archive.append((solution, objectives))

        if len(self.external_archive) > self.archive_limit:
            objectives_array = np.array([obj for _, obj in self.external_archive])

            distances = []
            for i in range(len(self.external_archive)):
                min_dist = float('inf')
                for j in range(len(self.external_archive)):
                    if i != j:
                        dist = np.linalg.norm(objectives_array[i] - objectives_array[j])
                        if dist < min_dist:
                            min_dist = dist
                distances.append((i, min_dist))

            distances.sort(key=lambda x: x[1])
            del self.external_archive[distances[0][0]]

        return True

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def calculate_metrics(self, population):
        """Calculate quality metrics"""
        if not self.track_metrics or not population:
            return None

        if self.external_archive:
            objectives = np.array([obj for _, obj in self.external_archive])
        else:
            objectives = np.array([ind['objectives'] for ind in population])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(objectives)
        metrics['spread'] = self.metrics_calculator.spread(objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(objectives)

        return metrics

    def run(self):
        """Main MOEA/D loop with normalization"""
        print(f"\nStarting MOEA/D-VNS Normalized...")
        print("="*60)

        print("Initializing population with diverse strategies...")
        self.population = []
        strategies = ['cooccurrence', 'semantic', 'hybrid']

        for i in range(self.pop_size):
            strategy = strategies[i % len(strategies)]
            chromosome = self.smart_initialization(strategy)
            objectives = self.evaluate_objectives(chromosome)

            self.population.append({
                'chromosome': chromosome,
                'objectives': objectives
            })

            norm_obj = self.normalize_objectives(objectives)
            self.z = np.minimum(self.z, norm_obj)
            self.nadir = np.maximum(self.nadir, norm_obj)

            self.update_external_archive(chromosome, objectives)

        print(f"Population initialized with {len(self.population)} solutions")
        print(f"Objective bounds: LU=[{self.obj_min[0]:.1f}, {self.obj_max[0]:.1f}], "
              f"SS=[{self.obj_min[1]:.3f}, {self.obj_max[1]:.3f}], "
              f"RSS=[{self.obj_min[2]:.1f}, {self.obj_max[2]:.1f}]")

        best_hv = 0
        no_improvement = 0

        for generation in range(self.max_gen):
            for i in range(self.pop_size):
                if random.random() < 0.9:
                    indices = self.neighbors[i]
                else:
                    indices = list(range(self.pop_size))

                parent_indices = random.sample(list(indices), min(2, len(indices)))
                parent1 = self.population[parent_indices[0]]['chromosome']
                parent2 = self.population[parent_indices[-1]]['chromosome'] if len(parent_indices) > 1 else parent1

                offspring = self.crossover(parent1, parent2)
                offspring = self.mutation(offspring)
                offspring = self.repair_solution(offspring)

                offspring_obj = self.evaluate_objectives(offspring)

                norm_obj = self.normalize_objectives(offspring_obj)
                self.z = np.minimum(self.z, norm_obj)
                self.nadir = np.maximum(self.nadir, norm_obj)

                self.update_external_archive(offspring, offspring_obj)

                max_updates = max(2, self.n_neighbors // 5)
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

            if generation % 5 == 0 and self.external_archive:
                for _ in range(min(3, len(self.external_archive))):
                    archive_sol, archive_obj = random.choice(self.external_archive)

                    fitness_values = [self.decompose(ind['objectives'], self.weights[j])
                                    for j, ind in enumerate(self.population)]
                    worst_idx = np.argmax(fitness_values)

                    self.population[worst_idx] = {
                        'chromosome': archive_sol.copy(),
                        'objectives': archive_obj.copy()
                    }

            if self.track_metrics:
                metrics = self.calculate_metrics(self.population)
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics:
                            self.metrics_history[key].append(metrics[key])

                    current_hv = metrics.get('hypervolume', 0)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1

            if generation % 10 == 0:
                if self.external_archive:
                    best_lu = min([obj[0] for _, obj in self.external_archive])
                    best_ss = min([obj[1] for _, obj in self.external_archive])
                    best_rss = min([obj[2] for _, obj in self.external_archive])
                else:
                    best_lu = min([ind['objectives'][0] for ind in self.population])
                    best_ss = min([ind['objectives'][1] for ind in self.population])
                    best_rss = min([ind['objectives'][2] for ind in self.population])

                print(f"Generation {generation}: Best LU={-best_lu:.2f}, "
                      f"SS={-best_ss:.4f}, RSS={best_rss:.1f}")

                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, "
                          f"Archive={len(self.external_archive)}, No_improv={no_improvement}")

        pareto_front = self.get_pareto_front()

        if self.track_metrics and self.metrics_history['hypervolume']:
            print("\n" + "="*60)
            print("FINAL METRICS SUMMARY")
            print("-"*60)
            print(f"Final Hypervolume: {self.metrics_history['hypervolume'][-1]:.4f}")
            print(f"Archive Size: {len(self.external_archive)}")

            if len(self.metrics_history['hypervolume']) > 1:
                initial_hv = self.metrics_history['hypervolume'][0]
                final_hv = self.metrics_history['hypervolume'][-1]
                improvement = ((final_hv - initial_hv) / (initial_hv + 1e-10)) * 100
                print(f"Hypervolume Improvement: {improvement:+.1f}%")

                if improvement > 0:
                    print("SUCCESS: Positive convergence achieved with normalization!")
            print("="*60)

        return pareto_front

    def get_pareto_front(self):
        """Extract Pareto front from external archive"""
        if self.external_archive:
            pareto_front = []
            for chromosome, objectives in self.external_archive:
                pareto_front.append({
                    'chromosome': chromosome,
                    'objectives': objectives,
                    'packages': self.decode_solution(chromosome)
                })
            return pareto_front

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

    track_metrics = '--metrics' in sys.argv or '--track-metrics' in sys.argv

    print(f"Testing {package_name} recommendation with normalized MOEA/D...")

    moead = MOEAD_VNS_Normalized(package_name, pop_size=100, max_gen=50,
                                 decomposition='tchebycheff', track_metrics=track_metrics)

    solutions = moead.run()

    print(f"\nFound {len(solutions)} Pareto optimal solutions")

    if solutions and len(solutions) > 0:
        print("\nTop solutions:")
        for i, sol in enumerate(solutions[:5], 1):
            print(f"  {i}. {sol['packages'][:5]} "
                  f"(LU={-sol['objectives'][0]:.2f}, "
                  f"SS={-sol['objectives'][1]:.4f}, "
                  f"RSS={sol['objectives'][2]:.0f})")