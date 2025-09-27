"""
NSGA-II v5 with Full SBERT Integration for PyCommend
4 objectives including semantic coherence using raw embeddings
"""

import numpy as np
import pickle
import random
import time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class NSGA2_V5:
    def __init__(self, package_name, pop_size=100, max_gen=100):
        """
        Initialize NSGA-II v5 with full semantic capabilities

        Args:
            package_name: Target package for recommendations
            pop_size: Population size
            max_gen: Maximum generations
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.max_gen = max_gen

        self.load_all_data()

        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{package_name}' not found in dataset")

        self.main_package_idx = self.pkg_to_idx[self.package_name]
        self.n_packages = len(self.package_names)
        self.n_objectives = 4

        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / self.n_packages
        self.eta_c = 20
        self.eta_m = 100

        self.min_size = 3
        self.max_size = 15

        self.initialize_semantic_components()

        print(f"NSGA-II v5 initialized for '{package_name}'")
        print(f"Using 4 objectives with full SBERT integration")
        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Semantic clusters: {self.n_clusters}")

    def load_all_data(self):
        """Load all three data sources including embeddings"""
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)
        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)
        self.sim_matrix = sim_data['similarity_matrix']

        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            emb_data = pickle.load(f)
        self.embeddings = emb_data['embeddings']

        print("All data loaded: relationships, similarity, and embeddings")

    def initialize_semantic_components(self):
        """Initialize semantic clustering and candidate pools"""
        print("Initializing semantic components...")

        self.n_clusters = 200
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)
        self.cluster_labels = self.kmeans.fit_predict(self.embeddings)

        self.target_cluster = self.cluster_labels[self.main_package_idx]
        cluster_members = np.where(self.cluster_labels == self.target_cluster)[0]
        print(f"Target package in cluster {self.target_cluster} with {len(cluster_members)} members")

        self.precompute_candidate_pools()

    def precompute_candidate_pools(self):
        """Precompute different candidate pools for initialization"""
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

    def evaluate_objectives(self, chromosome):
        """
        Evaluate 4 objectives including semantic coherence
        """
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size:
            return np.array([float('inf')] * 4)

        colink_score = 0
        for idx in indices:
            colink_score += self.rel_matrix[main_idx, idx]

        threshold = np.percentile(self.rel_matrix[main_idx].data, 75) if self.rel_matrix[main_idx].data.size > 0 else 1.0
        bonus = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > threshold]) * 0.1
        f1 = -(colink_score * (1 + bonus))

        similarities = [self.sim_matrix[main_idx, idx] for idx in indices]
        weights = 1.0 / (1.0 + np.arange(len(similarities)))
        f2 = -np.average(similarities, weights=weights/weights.sum())

        if len(indices) > 1:
            selected_embeddings = self.embeddings[indices]
            centroid = np.mean(selected_embeddings, axis=0)
            coherence_scores = cosine_similarity(selected_embeddings, [centroid]).flatten()
            coherence = np.mean(coherence_scores)
        else:
            coherence = 0.5
        f3 = -coherence

        size_penalty = abs(len(indices) - 7) * 0.1
        f4 = len(indices) + size_penalty

        return np.array([f1, f2, f3, f4])

    def smart_initialization(self, strategy='hybrid'):
        """
        Generate smart initial solution using multiple strategies
        """
        size = random.randint(self.min_size, self.max_size)

        if strategy == 'cooccur':
            candidates = self.cooccur_candidates[:min(100, len(self.cooccur_candidates))]
            if len(candidates) >= size:
                selected = np.random.choice(candidates, size, replace=False)
            else:
                selected = candidates

        elif strategy == 'semantic':
            candidates = self.semantic_candidates[:min(100, len(self.semantic_candidates))]
            if len(candidates) >= size:
                selected = np.random.choice(candidates, size, replace=False)
            else:
                selected = candidates

        elif strategy == 'cluster':
            if len(self.cluster_candidates) >= size:
                selected = np.random.choice(self.cluster_candidates, size, replace=False)
            else:
                selected = self.cluster_candidates

        else:
            cooccur_size = int(0.4 * size)
            semantic_size = int(0.4 * size)
            diverse_size = size - cooccur_size - semantic_size

            selected = []

            if len(self.cooccur_candidates) >= cooccur_size:
                selected.extend(np.random.choice(self.cooccur_candidates[:50], cooccur_size, replace=False))

            semantic_pool = np.setdiff1d(self.semantic_candidates[:50], selected)
            if len(semantic_pool) >= semantic_size:
                selected.extend(np.random.choice(semantic_pool, semantic_size, replace=False))

            other_clusters = np.where(self.cluster_labels != self.target_cluster)[0]
            diverse_pool = np.setdiff1d(other_clusters, selected)
            if len(diverse_pool) >= diverse_size:
                selected.extend(np.random.choice(diverse_pool, diverse_size, replace=False))

            selected = np.unique(selected)

        chromosome = np.zeros(self.n_packages, dtype=np.int8)
        chromosome[selected] = 1

        return chromosome

    def initialize_population(self):
        """
        Initialize population with multiple strategies
        """
        print("Initializing population with hybrid strategies...")
        population = []

        strategies = {
            'hybrid': int(0.4 * self.pop_size),
            'cooccur': int(0.2 * self.pop_size),
            'semantic': int(0.2 * self.pop_size),
            'cluster': int(0.2 * self.pop_size)
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

    def fast_non_dominated_sort(self, P):
        """Fast non-dominated sorting for 4 objectives"""
        S = [[] for _ in range(len(P))]
        n = [0] * len(P)
        F = [[]]

        for p_idx in range(len(P)):
            S[p_idx] = []
            n[p_idx] = 0

            for q_idx in range(len(P)):
                if p_idx == q_idx:
                    continue

                if self.dominates(P[p_idx]['objectives'], P[q_idx]['objectives']):
                    S[p_idx].append(q_idx)
                elif self.dominates(P[q_idx]['objectives'], P[p_idx]['objectives']):
                    n[p_idx] += 1

            if n[p_idx] == 0:
                P[p_idx]['rank'] = 0
                F[0].append(p_idx)

        i = 0
        while F[i]:
            Q = []
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        P[q]['rank'] = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)

        F.pop()
        return F

    def dominates(self, obj1, obj2):
        """Check domination for 4 objectives"""
        at_least_one_better = False
        for i in range(self.n_objectives):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                at_least_one_better = True
        return at_least_one_better

    def crowding_distance_assignment(self, I):
        """Crowding distance for 4 objectives"""
        l = len(I)
        for i in I:
            i['crowding_distance'] = 0

        for m in range(self.n_objectives):
            I.sort(key=lambda x: x['objectives'][m])

            I[0]['crowding_distance'] = float('inf')
            I[l-1]['crowding_distance'] = float('inf')

            f_max = I[l-1]['objectives'][m]
            f_min = I[0]['objectives'][m]

            if f_max - f_min == 0:
                continue

            for i in range(1, l-1):
                distance = I[i+1]['objectives'][m] - I[i-1]['objectives'][m]
                I[i]['crowding_distance'] += distance / (f_max - f_min)

    def tournament_selection(self, population):
        """Binary tournament selection"""
        p1 = random.choice(population)
        p2 = random.choice(population)

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

        if random.random() < self.crossover_prob:
            mask = np.random.rand(self.n_packages) < 0.5
            child[mask] = parent2['chromosome'][mask]

        return child

    def mutation(self, chromosome):
        """Bit flip mutation with repair"""
        mutated = chromosome.copy()

        for i in range(self.n_packages):
            if i == self.main_package_idx:
                continue

            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]

        current_size = np.sum(mutated)
        if current_size < self.min_size:
            available = np.where(mutated == 0)[0]
            available = available[available != self.main_package_idx]
            if len(available) > 0:
                to_add = min(self.min_size - current_size, len(available))
                indices = np.random.choice(available, to_add, replace=False)
                mutated[indices] = 1
        elif current_size > self.max_size:
            active = np.where(mutated == 1)[0]
            to_remove = current_size - self.max_size
            indices = np.random.choice(active, to_remove, replace=False)
            mutated[indices] = 0

        return mutated

    def run(self):
        """Execute NSGA-II v5 with 4 objectives"""
        print("\nStarting NSGA-II v5 with full SBERT integration...")
        print("="*60)

        population = self.initialize_population()
        best_objectives_history = []

        for generation in range(self.max_gen):
            fronts = self.fast_non_dominated_sort(population)

            new_population = []
            for front_idx, front in enumerate(fronts):
                front_individuals = [population[i] for i in front]
                self.crowding_distance_assignment(front_individuals)

                for idx, i in enumerate(front):
                    population[i] = front_individuals[idx]

                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    front_individuals = [population[i] for i in front]
                    front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)
                    new_population.extend(front_individuals[:self.pop_size - len(new_population)])
                    break

            offspring_population = []
            while len(offspring_population) < self.pop_size:
                parent1 = self.tournament_selection(new_population)
                parent2 = self.tournament_selection(new_population)

                child_chromosome = self.crossover(parent1, parent2)
                child_chromosome = self.mutation(child_chromosome)

                objectives = self.evaluate_objectives(child_chromosome)

                offspring_population.append({
                    'chromosome': child_chromosome,
                    'objectives': objectives,
                    'rank': None,
                    'crowding_distance': 0
                })

            population = new_population + offspring_population
            fronts = self.fast_non_dominated_sort(population)

            new_population = []
            for front_idx, front in enumerate(fronts):
                front_individuals = [population[i] for i in front]
                self.crowding_distance_assignment(front_individuals)

                for idx, i in enumerate(front):
                    population[i] = front_individuals[idx]

                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    front_individuals = [population[i] for i in front]
                    front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)
                    new_population.extend(front_individuals[:self.pop_size - len(new_population)])
                    break

            population = new_population

            if generation % 10 == 0:
                if fronts and len(fronts) > 0:
                    if len(fronts[0]) > 0:
                        pareto_front = [population[i] for i in fronts[0]]
                        if pareto_front:
                            best = min(pareto_front, key=lambda x: x['objectives'][0])
                            print(f"Generation {generation}: Pareto size={len(pareto_front)}")
                            print(f"  Best: F1={-best['objectives'][0]:.2f}, F2={-best['objectives'][1]:.4f}, "
                                  f"F3={-best['objectives'][2]:.4f}, F4={best['objectives'][3]:.1f}")
                            best_objectives_history.append(best['objectives'])

        final_fronts = self.fast_non_dominated_sort(population)
        if final_fronts and final_fronts[0]:
            pareto_solutions = [population[i] for i in final_fronts[0]]
        else:
            pareto_solutions = population[:min(10, len(population))]

        return pareto_solutions

    def get_recommendations(self, solutions):
        """Extract package recommendations from solutions"""
        recommendations = []

        for sol in solutions:
            indices = np.where(sol['chromosome'] == 1)[0]
            packages = [self.package_names[i] for i in indices]

            main_idx = self.main_package_idx
            scores = {}
            for idx in indices:
                pkg_name = self.package_names[idx]
                colink = float(self.rel_matrix[main_idx, idx])
                similarity = float(self.sim_matrix[main_idx, idx])

                pkg_embedding = self.embeddings[idx]
                main_embedding = self.embeddings[main_idx]
                semantic_score = cosine_similarity([pkg_embedding], [main_embedding])[0, 0]

                scores[pkg_name] = {
                    'colink': colink,
                    'similarity': similarity,
                    'semantic': float(semantic_score),
                    'combined': float(colink * 0.4 + similarity * 0.3 + semantic_score * 0.3)
                }

            recommendations.append({
                'packages': packages,
                'objectives': sol['objectives'].tolist(),
                'scores': scores
            })

        return recommendations


def main():
    """Test NSGA-II v5"""
    import argparse

    parser = argparse.ArgumentParser(description='NSGA-II v5 with Full SBERT Integration')
    parser.add_argument('--package', type=str, default='numpy',
                       help='Package name for recommendations')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations')

    args = parser.parse_args()

    try:
        nsga2 = NSGA2_V5(args.package, pop_size=args.pop_size, max_gen=args.generations)

        start_time = time.time()
        solutions = nsga2.run()
        execution_time = time.time() - start_time

        print(f"\nExecution completed in {execution_time:.2f} seconds")
        print(f"Found {len(solutions)} Pareto optimal solutions")

        print("\nTop Recommendations:")
        print("="*60)

        if solutions:
            best_solution = min(solutions, key=lambda x: x['objectives'][0])
            indices = np.where(best_solution['chromosome'] == 1)[0]

            print(f"Best solution for {args.package}:")
            for idx in indices[:10]:
                pkg_name = nsga2.package_names[idx]
                colink = nsga2.rel_matrix[nsga2.main_package_idx, idx]
                similarity = nsga2.sim_matrix[nsga2.main_package_idx, idx]

                pkg_emb = nsga2.embeddings[idx]
                main_emb = nsga2.embeddings[nsga2.main_package_idx]
                semantic = cosine_similarity([pkg_emb], [main_emb])[0, 0]

                print(f"  - {pkg_name}: colink={colink:.2f}, sim={similarity:.4f}, semantic={semantic:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()