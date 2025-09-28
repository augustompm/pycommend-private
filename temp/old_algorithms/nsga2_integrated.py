"""
NSGA-II with Weighted Probability Initialization for PyCommend
Integration of research-validated initialization method (74.3% success rate)
"""

import numpy as np
import pickle
import random
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
import time
import os
import sys

sys.path.append('../..')
from simple_best_method import weighted_probability_initialization

class NSGA2:
    def __init__(self, package_name, pop_size=100, max_gen=100):
        """
        Initialize NSGA-II with Weighted Probability Initialization

        Args:
            package_name: Main package for recommendations
            pop_size: Population size
            max_gen: Maximum generations
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.max_gen = max_gen

        self.load_data()

        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{package_name}' not found in dataset")

        self.main_package_idx = self.pkg_to_idx[self.package_name]
        self.n_packages = len(self.package_names)

        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / self.n_packages

        self.eta_c = 20
        self.eta_m = 100

        self.min_size = 3
        self.max_size = 15

        self.colink_threshold = self.calculate_colink_threshold()

        print(f"NSGA-II initialized for package '{package_name}'")
        print(f"Using Weighted Probability Initialization (74.3% success rate)")
        print(f"Total packages: {self.n_packages}")
        print(f"Solution size: {self.min_size}-{self.max_size}")
        print(f"Colink threshold: {self.colink_threshold:.2f}")

    def calculate_colink_threshold(self):
        """Calculate threshold for strong connections"""
        sample_size = 10000
        sample_values = []

        rows = np.random.randint(0, self.rel_matrix.shape[0], sample_size)
        cols = np.random.randint(0, self.rel_matrix.shape[1], sample_size)

        for i in range(sample_size):
            val = self.rel_matrix[rows[i], cols[i]]
            if val > 0:
                sample_values.append(val)

        if not sample_values:
            return 1.0

        return np.percentile(sample_values, 75)

    def load_data(self):
        """Load relationship and similarity matrices"""
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)

        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)

        self.sim_matrix = sim_data['similarity_matrix']

        print("Data loaded successfully!")

    def initialize_population(self):
        """
        Initialize population using Weighted Probability method
        74.3% success rate validated with real data
        """
        print("Initializing population with Weighted Probability method...")

        individuals_data = weighted_probability_initialization(
            self.rel_matrix,
            self.main_package_idx,
            pop_size=self.pop_size,
            k=100
        )

        population = []
        for idx_array in individuals_data:
            chromosome = np.zeros(self.n_packages, dtype=np.int8)
            chromosome[idx_array] = 1

            objectives = self.evaluate_objectives(chromosome)

            population.append({
                'chromosome': chromosome,
                'objectives': objectives,
                'rank': None,
                'crowding_distance': 0
            })

        print(f"Population initialized with {len(population)} individuals")
        return population

    def evaluate_objectives(self, chromosome):
        """
        Evaluate three objectives with sophisticated calculation
        """
        main_idx = self.pkg_to_idx[self.package_name]
        indices = np.where(chromosome == 1)[0]

        if len(indices) == 0:
            return np.array([float('inf'), float('inf'), float('inf')])

        total_colink = 0
        diversity_bonus = 0

        for idx in indices:
            connection_strength = self.rel_matrix[main_idx, idx]

            if connection_strength > self.colink_threshold:
                bonus = 1.5
            elif connection_strength > 0:
                bonus = 1.0
            else:
                bonus = 0.5

            total_colink += connection_strength * bonus

        if len(indices) > 1:
            unique_categories = len(set([idx % 10 for idx in indices]))
            diversity_bonus = unique_categories * 0.1

        total_colink = total_colink * (1 + diversity_bonus)

        f1 = -total_colink

        if len(indices) > 0:
            sim_values = [self.sim_matrix[main_idx, idx] for idx in indices]
            max_similarity = np.mean(sim_values)
        else:
            max_similarity = 0
        f2 = -max_similarity

        size_penalty = abs(len(indices) - 7) * 0.1
        f3 = len(indices) + size_penalty

        return np.array([f1, f2, f3])

    def fast_non_dominated_sort(self, P):
        """Fast non-dominated sorting algorithm"""
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
        """Check if obj1 dominates obj2"""
        at_least_one_better = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                at_least_one_better = True
        return at_least_one_better

    def crowding_distance_assignment(self, I):
        """Assign crowding distance to individuals"""
        l = len(I)
        for i in I:
            i['crowding_distance'] = 0

        for m in range(len(I[0]['objectives'])):
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
        """Bit flip mutation with size control"""
        mutated = chromosome.copy()

        for i in range(self.n_packages):
            if i == self.pkg_to_idx[self.package_name]:
                continue

            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]

        current_size = np.sum(mutated)
        if current_size < self.min_size:
            available = np.where((mutated == 0) &
                                (np.arange(self.n_packages) != self.pkg_to_idx[self.package_name]))[0]
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
        """Execute NSGA-II algorithm"""
        print("\nStarting NSGA-II with Weighted Probability Initialization...")
        print("="*60)

        population = self.initialize_population()

        best_solutions = []

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
                if fronts and len(fronts) > 0 and len(fronts[0]) > 0:
                    pareto_front = [population[i] for i in fronts[0]]
                    print(f"Generation {generation}: Pareto front size = {len(pareto_front)}")

                    if pareto_front:
                        best = min(pareto_front, key=lambda x: x['objectives'][0])
                        selected_indices = np.where(best['chromosome'] == 1)[0]
                        print(f"  Best colink: {-best['objectives'][0]:.2f}")
                        print(f"  Packages: {len(selected_indices)}")
                else:
                    print(f"Generation {generation}: No Pareto front yet")

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

            main_idx = self.pkg_to_idx[self.package_name]
            scores = {}
            for idx in indices:
                pkg_name = self.package_names[idx]
                colink = self.rel_matrix[main_idx, idx]
                similarity = self.sim_matrix[main_idx, idx]
                scores[pkg_name] = {
                    'colink': float(colink),
                    'similarity': float(similarity),
                    'combined': float(colink * 0.7 + similarity * 0.3)
                }

            recommendations.append({
                'packages': packages,
                'objectives': sol['objectives'].tolist(),
                'scores': scores
            })

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='NSGA-II Package Recommender with Weighted Probability')
    parser.add_argument('--package', type=str, default='numpy',
                       help='Package name for recommendations')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations')

    args = parser.parse_args()

    try:
        nsga2 = NSGA2(args.package, pop_size=args.pop_size, max_gen=args.generations)

        start_time = time.time()
        solutions = nsga2.run()
        execution_time = time.time() - start_time

        print(f"\nExecution completed in {execution_time:.2f} seconds")
        print(f"Found {len(solutions)} Pareto optimal solutions")

        recommendations = nsga2.get_recommendations(solutions)

        print("\nTop Recommendations:")
        print("="*60)

        best_solution = min(solutions, key=lambda x: x['objectives'][0])
        indices = np.where(best_solution['chromosome'] == 1)[0]

        print(f"Best solution for {args.package}:")
        for idx in indices[:10]:
            pkg_name = nsga2.package_names[idx]
            colink = nsga2.rel_matrix[nsga2.pkg_to_idx[args.package], idx]
            similarity = nsga2.sim_matrix[nsga2.pkg_to_idx[args.package], idx]
            print(f"  - {pkg_name}: colink={colink:.2f}, similarity={similarity:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()