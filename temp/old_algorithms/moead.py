"""
MOEA/D implementation for PyCommend
Multi-Objective Evolutionary Algorithm based on Decomposition
Based on Zhang & Li (2007) IEEE Transactions on Evolutionary Computation
"""

import numpy as np
import pickle
import random
import time
import os
import pandas as pd
from scipy.spatial.distance import cdist
from pathlib import Path


class MOEAD:
    def __init__(self, package_name, pop_size=100, n_neighbors=20, max_gen=100,
                 decomposition='tchebycheff', cr=1.0, f_scale=0.5, eta_m=20):
        """
        Initialize MOEA/D for package recommendation

        Args:
            package_name: Target package for recommendations
            pop_size: Number of subproblems/weight vectors (N)
            n_neighbors: Size of neighborhood T
            max_gen: Maximum generations
            decomposition: 'tchebycheff', 'weighted_sum', or 'pbi'
            cr: Crossover rate for DE operator
            f_scale: Scaling factor for DE operator
            eta_m: Distribution index for polynomial mutation
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.T = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.cr = cr
        self.f_scale = f_scale
        self.eta_m = eta_m

        self.load_data()

        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{package_name}' not found in dataset")

        self.n_packages = len(self.package_names)
        self.min_size = 3
        self.max_size = 15
        self.mutation_prob = 1.0 / self.n_packages
        self.neighbor_selection_prob = 0.9
        self.nr = 2
        self.colink_threshold = 0.01

        self.weights = self.generate_weight_vectors(pop_size, 3)
        self.B = self.define_neighborhoods()
        self.population = [self.create_individual() for _ in range(pop_size)]
        self.objectives = [self.evaluate_objectives(ind) for ind in self.population]
        self.z_star = np.min(self.objectives, axis=0)
        self.z_nadir = np.max(self.objectives, axis=0)
        self.archive = []
        self.theta = 5.0

        print(f"MOEA/D initialized for package '{package_name}'")
        print(f"Population: {pop_size}, Neighbors: {self.T}, Generations: {max_gen}")
        print(f"Decomposition: {decomposition}, CR: {cr}, F: {f_scale}")

    def load_data(self):
        """
        Load relationship and similarity matrices
        """
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)

        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)

        self.sim_matrix = sim_data['similarity_matrix']

        print("Data loaded successfully!")

    def generate_weight_vectors(self, n, m):
        """
        Generate uniformly distributed weight vectors using simplex-lattice design
        Based on Das and Dennis's systematic approach

        Args:
            n: Number of weight vectors
            m: Number of objectives
        """
        weights = []

        if m == 2:
            for i in range(n):
                w1 = i / (n - 1)
                w2 = 1 - w1
                weights.append([w1, w2])
        elif m == 3:
            h = 1
            while self.combination(h + m - 1, m - 1) < n:
                h += 1

            for i in range(h + 1):
                for j in range(h + 1 - i):
                    k = h - i - j
                    if k >= 0:
                        w = np.array([i, j, k]) / h
                        weights.append(w.tolist())
                        if len(weights) >= n:
                            break
                if len(weights) >= n:
                    break

            if len(weights) < n:
                for i in range(1, h):
                    if len(weights) >= n:
                        break
                    weights.append([i/h, (h-i)/h, 0])
                    if len(weights) >= n:
                        break
                    weights.append([i/h, 0, (h-i)/h])
                    if len(weights) >= n:
                        break
                    weights.append([0, i/h, (h-i)/h])
        else:
            for _ in range(n):
                w = np.random.dirichlet([1] * m)
                weights.append(w.tolist())

        if len(weights) > n:
            weights = weights[:n]
        elif len(weights) < n:
            while len(weights) < n:
                w = np.random.dirichlet([1] * m)
                weights.append(w.tolist())

        return np.array(weights)

    def combination(self, n, k):
        """
        Calculate n choose k
        """
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    def define_neighborhoods(self):
        """
        Define T closest neighbors for each weight vector
        """
        distances = cdist(self.weights, self.weights, metric='euclidean')

        B = []
        for i in range(self.pop_size):
            neighbors_idx = np.argsort(distances[i])[:self.T+1]
            neighbors = [idx for idx in neighbors_idx if idx != i][:self.T]
            B.append(neighbors)

        return B

    def create_individual(self):
        """
        Create a random individual (binary vector)
        """
        chromosome = np.zeros(self.n_packages, dtype=np.int8)

        size = random.randint(self.min_size, self.max_size)

        main_idx = self.pkg_to_idx[self.package_name]
        available = [i for i in range(self.n_packages) if i != main_idx]
        selected = random.sample(available, min(size, len(available)))

        chromosome[selected] = 1

        return chromosome

    def evaluate_objectives(self, chromosome):
        """
        Evaluate three objectives for a solution with threshold filtering

        Returns:
            [f1, f2, f3] where all are to be minimized
        """
        main_idx = self.pkg_to_idx[self.package_name]
        selected_indices = [i for i in range(self.n_packages) if chromosome[i] == 1]
        all_indices = selected_indices + [main_idx]

        size = len(selected_indices)
        if size < self.min_size:
            return [float('inf'), float('inf'), float('inf')]

        total_linked = 0.0
        count = 0
        strong_connections = set()

        for i in range(len(all_indices)):
            for j in range(i+1, len(all_indices)):
                link_value = self.rel_matrix[all_indices[i], all_indices[j]]
                total_linked += link_value
                count += 1

                if link_value > self.colink_threshold:
                    strong_connections.add(all_indices[i])
                    strong_connections.add(all_indices[j])

        avg_linked = total_linked / count if count > 0 else 0.0

        if main_idx in strong_connections:
            strong_connections.remove(main_idx)

        proportion_strong = len(strong_connections) / size if size > 0 else 0

        if proportion_strong > 0:
            f1 = -avg_linked * (1 + np.log1p(proportion_strong))
        else:
            f1 = -avg_linked

        total_sim = 0.0
        for idx in selected_indices:
            total_sim += self.sim_matrix[main_idx, idx]

        f2 = -total_sim / len(selected_indices) if selected_indices else 0.0

        f3 = size

        return np.array([f1, f2, f3])

    def decompose(self, objectives, weight):
        """
        Decomposition function based on selected method
        """
        if self.decomposition == 'tchebycheff':
            return self.tchebycheff(objectives, weight)
        elif self.decomposition == 'weighted_sum':
            return self.weighted_sum(objectives, weight)
        elif self.decomposition == 'pbi':
            return self.pbi(objectives, weight)
        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def tchebycheff(self, objectives, weight):
        """
        Tchebycheff decomposition function
        g^te(x|λ,z*) = max{λ_i * |f_i(x) - z*_i|}
        """
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(self.z_star)

        weighted_diff = weight * np.abs(norm_obj - norm_z)
        return np.max(weighted_diff)

    def weighted_sum(self, objectives, weight):
        """
        Weighted sum decomposition
        g^ws(x|λ) = Σ λ_i * f_i(x)
        """
        norm_obj = self.normalize_objectives(objectives)
        return np.sum(weight * norm_obj)

    def pbi(self, objectives, weight):
        """
        Penalty-based Boundary Intersection (PBI)
        g^pbi(x|λ,z*) = d1 + θ * d2
        """
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(self.z_star)

        diff = norm_obj - norm_z
        norm_weight = weight / np.linalg.norm(weight)
        d1 = np.abs(np.dot(diff, norm_weight))
        d2 = np.linalg.norm(diff - d1 * norm_weight)

        return d1 + self.theta * d2

    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0,1] range
        """
        if np.all(self.z_nadir == self.z_star):
            return objectives

        range_obj = self.z_nadir - self.z_star
        range_obj[range_obj == 0] = 1.0

        return (objectives - self.z_star) / range_obj

    def genetic_operator(self, parent1, parent2, parent3=None):
        """
        DE/rand/2 operator with polynomial mutation
        Based on MOEA/D-DE variant
        """
        if parent3 is None:
            offspring = self.uniform_crossover(parent1, parent2)
        else:
            offspring = self.de_operator(parent1, parent2, parent3)

        offspring = self.polynomial_mutation(offspring)
        offspring = self.repair_chromosome(offspring)

        return offspring

    def uniform_crossover(self, parent1, parent2):
        """
        Standard uniform crossover
        """
        offspring = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if random.random() < self.cr:
                offspring[i] = parent2[i]
            else:
                offspring[i] = parent1[i]
        return offspring

    def de_operator(self, parent1, parent2, parent3):
        """
        Differential Evolution operator
        v = parent1 + F * (parent2 - parent3)
        """
        p1_cont = parent1.astype(float)
        p2_cont = parent2.astype(float)
        p3_cont = parent3.astype(float)

        v = p1_cont + self.f_scale * (p2_cont - p3_cont)

        offspring = np.zeros_like(parent1, dtype=float)
        for i in range(len(parent1)):
            if random.random() < self.cr:
                offspring[i] = v[i]
            else:
                offspring[i] = p1_cont[i]

        offspring = (offspring > 0.5).astype(np.int8)

        return offspring

    def polynomial_mutation(self, offspring):
        """
        Polynomial mutation operator
        """
        for i in range(len(offspring)):
            if i == self.pkg_to_idx[self.package_name]:
                continue

            if random.random() < self.mutation_prob:
                offspring[i] = 1 - offspring[i]

        return offspring

    def repair_chromosome(self, chromosome):
        """
        Repair chromosome to satisfy size constraints
        """
        selected = np.sum(chromosome)

        if selected < self.min_size:
            main_idx = self.pkg_to_idx[self.package_name]
            zero_indices = [i for i in range(len(chromosome))
                          if chromosome[i] == 0 and i != main_idx]
            if zero_indices:
                n_add = min(self.min_size - selected, len(zero_indices))
                add_indices = random.sample(zero_indices, n_add)
                chromosome[add_indices] = 1

        elif selected > self.max_size:
            one_indices = np.where(chromosome == 1)[0]
            n_remove = selected - self.max_size
            if len(one_indices) > n_remove:
                remove_indices = random.sample(list(one_indices), n_remove)
                chromosome[remove_indices] = 0

        return chromosome

    def update_reference_point(self, objectives):
        """
        Update ideal point z* and nadir point
        """
        self.z_star = np.minimum(self.z_star, objectives)
        self.z_nadir = np.maximum(self.z_nadir, objectives)

    def update_archive(self, solution, objectives):
        """
        Update external archive with non-dominated solutions
        """
        to_remove = []
        is_dominated = False

        for i, (arch_sol, arch_obj) in enumerate(self.archive):
            if self.dominates(objectives, arch_obj):
                to_remove.append(i)
            elif self.dominates(arch_obj, objectives):
                is_dominated = True
                break

        for i in reversed(to_remove):
            del self.archive[i]

        if not is_dominated:
            self.archive.append((solution.copy(), objectives.copy()))

            if len(self.archive) > self.pop_size * 2:
                self.archive = self.maintain_archive(self.archive, self.pop_size)

    def dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2 (minimization)
        """
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                better_in_any = True
        return better_in_any

    def maintain_archive(self, archive, max_size):
        """
        Maintain archive using crowding distance
        """
        if len(archive) <= max_size:
            return archive

        objectives = np.array([obj for _, obj in archive])
        n_obj = objectives.shape[1]
        n_sol = len(archive)

        distances = np.zeros(n_sol)

        for m in range(n_obj):
            sorted_idx = np.argsort(objectives[:, m])

            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')

            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
            if obj_range > 0:
                for i in range(1, n_sol - 1):
                    distances[sorted_idx[i]] += (
                        objectives[sorted_idx[i+1], m] - objectives[sorted_idx[i-1], m]
                    ) / obj_range

        sorted_idx = np.argsort(-distances)
        return [archive[i] for i in sorted_idx[:max_size]]

    def run(self):
        """
        Main MOEA/D algorithm loop
        """
        start_time = time.time()

        print(f"Starting MOEA/D optimization...")

        for i in range(self.pop_size):
            self.update_archive(self.population[i], self.objectives[i])

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                if random.random() < self.neighbor_selection_prob:
                    pool = self.B[i]
                else:
                    pool = list(range(self.pop_size))

                if self.decomposition == 'tchebycheff' and random.random() < 0.5:
                    if len(pool) >= 3:
                        parents_idx = random.sample(pool, 3)
                    else:
                        parents_idx = random.choices(range(self.pop_size), k=3)

                    parent1 = self.population[parents_idx[0]]
                    parent2 = self.population[parents_idx[1]]
                    parent3 = self.population[parents_idx[2]]

                    offspring = self.genetic_operator(parent1, parent2, parent3)
                else:
                    if len(pool) >= 2:
                        parents_idx = random.sample(pool, 2)
                    else:
                        parents_idx = random.choices(range(self.pop_size), k=2)

                    parent1 = self.population[parents_idx[0]]
                    parent2 = self.population[parents_idx[1]]

                    offspring = self.genetic_operator(parent1, parent2)

                offspring_obj = self.evaluate_objectives(offspring)

                self.update_reference_point(offspring_obj)

                n_updates = 0
                update_order = np.random.permutation(self.B[i])

                for j in update_order:
                    if n_updates >= self.nr:
                        break

                    current_fitness = self.decompose(self.objectives[j], self.weights[j])
                    offspring_fitness = self.decompose(offspring_obj, self.weights[j])

                    if offspring_fitness < current_fitness:
                        self.population[j] = offspring.copy()
                        self.objectives[j] = offspring_obj.copy()
                        n_updates += 1

                self.update_archive(offspring, offspring_obj)

            if gen % 10 == 0 or gen == self.max_gen - 1:
                elapsed = time.time() - start_time
                avg_f1 = -np.mean([obj[0] for obj in self.objectives])
                avg_f2 = -np.mean([obj[1] for obj in self.objectives])
                avg_f3 = np.mean([obj[2] for obj in self.objectives])

                print(f"Generation {gen}/{self.max_gen} ({elapsed:.2f}s)")
                print(f"  Archive size: {len(self.archive)}")
                print(f"  Avg F1 (linked): {avg_f1:.4f}")
                print(f"  Avg F2 (similarity): {avg_f2:.4f}")
                print(f"  Avg F3 (size): {avg_f3:.2f}")

        elapsed = time.time() - start_time
        print(f"MOEA/D completed in {elapsed:.2f}s")
        print(f"Final archive: {len(self.archive)} solutions")

        return self.archive

    def format_results(self, solutions):
        """
        Format solutions into DataFrame
        """
        results = []
        unique_sets = {}

        for sol, obj in solutions:
            f1 = -obj[0]
            f2 = -obj[1]
            f3 = obj[2]

            selected_indices = np.where(sol == 1)[0]
            packages = sorted([self.package_names[i] for i in selected_indices])

            package_key = ','.join(packages)

            if package_key not in unique_sets or (f1 + f2 - f3) > unique_sets[package_key]['score']:
                unique_sets[package_key] = {
                    'linked_usage': f1,
                    'semantic_similarity': f2,
                    'size': f3,
                    'recommended_packages': package_key,
                    'score': f1 + f2 - f3
                }

        for i, data in enumerate(unique_sets.values(), 1):
            data['solution_id'] = i
            del data['score']
            results.append(data)

        df = pd.DataFrame(results)

        df = df.sort_values(by=['size', 'semantic_similarity', 'linked_usage'],
                           ascending=[True, False, False])

        df['solution_id'] = range(1, len(df) + 1)

        print(f"Found {len(df)} unique solutions")

        return df

    def save_results(self, df, output_dir='results'):
        """
        Save results to CSV
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{self.package_name}_moead_recommendations.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


def main():
    """
    Main function for testing
    """
    import argparse

    parser = argparse.ArgumentParser(description='MOEA/D for package recommendation')
    parser.add_argument('package', type=str, help='Package name')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--neighbors', type=int, default=20, help='Neighborhood size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    try:
        moead = MOEAD(args.package, args.pop_size, args.neighbors, args.generations)
        solutions = moead.run()

        results_df = moead.format_results(solutions)
        moead.save_results(results_df, args.output_dir)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()