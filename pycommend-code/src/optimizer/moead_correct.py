"""
MOEA/D implementation for PyCommend - Corrected Version
Multi-Objective Evolutionary Algorithm based on Decomposition
Following Zhang & Li (2007) paper strictly
"""

import numpy as np
import pickle
import random
import time
import os
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import comb
from itertools import combinations_with_replacement


class MOEAD_Correct:
    def __init__(self, package_name, pop_size=100, n_neighbors=20, max_gen=100,
                 decomposition='tchebycheff', delta=0.9, nr=2):
        """
        Initialize MOEA/D following original paper

        Args:
            package_name: Target package for recommendations
            pop_size: Number of subproblems (weight vectors)
            n_neighbors: Size of neighborhood T
            max_gen: Maximum generations
            decomposition: 'tchebycheff' or 'pbi' (Penalty-based Boundary Intersection)
            delta: Probability of selecting parents from neighborhood
            nr: Maximum number of solutions replaced by offspring
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.T = min(n_neighbors, pop_size - 1)
        self.max_gen = max_gen
        self.decomposition = decomposition
        self.delta = delta
        self.nr = nr

        # Load data
        self.load_data()

        # Verify package exists
        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Package '{package_name}' not found in dataset")

        # Problem parameters
        self.n_packages = len(self.package_names)
        self.n_objectives = 3
        self.min_size = 3
        self.max_size = 15

        # Genetic operator parameters (from paper)
        self.CR = 1.0  # Crossover rate
        self.F = 0.5   # Differential evolution parameter
        self.pm = 1.0 / self.n_packages  # Mutation probability
        self.eta_m = 20  # Polynomial mutation distribution index

        # Generate uniformly distributed weight vectors
        self.weights = self.generate_uniform_weights(pop_size, self.n_objectives)
        self.pop_size = len(self.weights)  # Adjust if needed

        # Calculate euclidean distances between weight vectors
        self.weight_distances = cdist(self.weights, self.weights)

        # Define neighborhood for each subproblem
        self.B = self.compute_neighborhoods()

        # Initialize population
        self.population = self.initialize_population()

        # Evaluate initial population
        self.objectives = np.array([self.evaluate_objectives(ind) for ind in self.population])

        # Initialize reference point (ideal point)
        self.z = np.min(self.objectives, axis=0)

        # Initialize nadir point (for normalization)
        self.z_nad = np.max(self.objectives, axis=0)

        # External population (EP) for storing non-dominated solutions
        self.external_population = []
        self.update_external_population(self.population, self.objectives)

        print(f"MOEA/D initialized correctly for package '{package_name}'")
        print(f"Population: {self.pop_size}, Neighbors: {self.T}, Generations: {max_gen}")
        print(f"Decomposition: {decomposition}, Delta: {delta}, Nr: {nr}")

    def load_data(self):
        """Load relationship and similarity matrices"""
        # Load relationship matrix
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)

        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}

        # Load similarity matrix
        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)

        self.sim_matrix = sim_data['similarity_matrix']

        print("Data loaded successfully")

    def generate_uniform_weights(self, n_points, n_obj):
        """
        Generate uniformly distributed weight vectors using Das-Dennis approach

        Args:
            n_points: Desired number of weight vectors
            n_obj: Number of objectives
        """
        # Calculate appropriate H for uniform distribution
        H = 1
        while comb(H + n_obj - 1, n_obj - 1) < n_points:
            H += 1

        # Generate reference directions
        weights = []
        for combo in combinations_with_replacement(range(H + 1), n_obj - 1):
            weight = []
            prev = 0
            for val in combo:
                weight.append(val - prev)
                prev = val
            weight.append(H - prev)

            # Normalize to sum to 1
            weight = np.array(weight) / H
            weights.append(weight)

        weights = np.array(weights)

        # If we have too many, sample randomly
        if len(weights) > n_points:
            indices = np.random.choice(len(weights), n_points, replace=False)
            weights = weights[indices]

        # If we need more, add random weights
        while len(weights) < n_points:
            w = np.random.dirichlet(np.ones(n_obj))
            weights = np.vstack([weights, w])

        return weights[:n_points]

    def compute_neighborhoods(self):
        """
        Compute T closest weight vectors for each subproblem
        """
        B = []
        for i in range(self.pop_size):
            # Sort distances and get T closest neighbors (excluding self)
            distances = self.weight_distances[i].copy()
            distances[i] = np.inf  # Exclude self
            neighbors = np.argsort(distances)[:self.T]
            B.append(neighbors)

        return B

    def initialize_population(self):
        """
        Initialize population with problem-specific heuristics
        """
        population = []
        main_idx = self.pkg_to_idx[self.package_name]

        for i in range(self.pop_size):
            chromosome = np.zeros(self.n_packages, dtype=np.int8)

            # Use different initialization strategies
            if i < self.pop_size // 3:
                # Strategy 1: Select packages with high co-usage with main package
                co_usage = self.rel_matrix[main_idx].copy()
                co_usage[main_idx] = -1  # Exclude self

                # Select top packages by co-usage
                size = random.randint(self.min_size, self.max_size)
                top_indices = np.argsort(co_usage)[-size:]
                chromosome[top_indices] = 1

            elif i < 2 * self.pop_size // 3:
                # Strategy 2: Select packages with high similarity
                similarity = self.sim_matrix[main_idx].copy()
                similarity[main_idx] = -1  # Exclude self

                size = random.randint(self.min_size, self.max_size)
                top_indices = np.argsort(similarity)[-size:]
                chromosome[top_indices] = 1

            else:
                # Strategy 3: Random selection
                size = random.randint(self.min_size, self.max_size)
                available = [j for j in range(self.n_packages) if j != main_idx]
                selected = random.sample(available, size)
                chromosome[selected] = 1

            # Ensure main package is not selected
            chromosome[main_idx] = 0

            population.append(chromosome)

        return population

    def evaluate_objectives(self, chromosome):
        """
        Evaluate three objectives (all to be minimized)

        F1: Negative average co-usage (maximize co-usage -> minimize negative)
        F2: Negative average similarity (maximize similarity -> minimize negative)
        F3: Set size (minimize)
        """
        main_idx = self.pkg_to_idx[self.package_name]
        selected_indices = np.where(chromosome == 1)[0]

        if len(selected_indices) < self.min_size:
            # Apply penalty instead of infinity
            penalty = 1000.0 * (self.min_size - len(selected_indices))
            return np.array([penalty, penalty, penalty])

        if len(selected_indices) > self.max_size:
            # Apply penalty for exceeding max size
            penalty = 100.0 * (len(selected_indices) - self.max_size)
            return np.array([penalty, penalty, len(selected_indices)])

        # F1: Negative average pairwise co-usage
        all_indices = np.append(selected_indices, main_idx)
        co_usage_sum = 0.0
        count = 0

        for i in range(len(all_indices)):
            for j in range(i + 1, len(all_indices)):
                co_usage_sum += self.rel_matrix[all_indices[i], all_indices[j]]
                count += 1

        f1 = -co_usage_sum / count if count > 0 else 0.0

        # F2: Negative average similarity to main package
        similarity_sum = np.sum(self.sim_matrix[main_idx, selected_indices])
        f2 = -similarity_sum / len(selected_indices)

        # F3: Set size
        f3 = len(selected_indices)

        return np.array([f1, f2, f3])

    def normalize_objectives(self, objectives):
        """
        Normalize objectives using ideal and nadir points
        """
        z_range = self.z_nad - self.z
        z_range[z_range == 0] = 1.0  # Avoid division by zero
        return (objectives - self.z) / z_range

    def tchebycheff_decomposition(self, objectives, weight):
        """
        Weighted Tchebycheff approach
        g^te(x|w,z*) = max{w_i|f_i(x) - z*_i|}
        """
        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(self.z)

        # Calculate weighted Tchebycheff
        return np.max(weight * np.abs(norm_obj - norm_z))

    def pbi_decomposition(self, objectives, weight, theta=5.0):
        """
        Penalty-based Boundary Intersection (PBI) approach
        """
        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)
        norm_z = self.normalize_objectives(self.z)

        # Calculate d1 and d2
        diff = norm_obj - norm_z
        weight_norm = np.linalg.norm(weight)

        if weight_norm == 0:
            return np.linalg.norm(diff)

        d1 = np.abs(np.dot(diff, weight)) / weight_norm
        d2 = np.linalg.norm(diff - (d1 * weight / weight_norm))

        return d1 + theta * d2

    def evaluate_subproblem(self, objectives, weight):
        """
        Evaluate solution for a subproblem
        """
        if self.decomposition == 'tchebycheff':
            return self.tchebycheff_decomposition(objectives, weight)
        elif self.decomposition == 'pbi':
            return self.pbi_decomposition(objectives, weight)
        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def differential_evolution_crossover(self, idx, pool):
        """
        Differential evolution crossover operator
        """
        if len(pool) < 3:
            pool = list(range(self.pop_size))

        # Select three different parents
        parents = random.sample(pool, 3)

        # Get parent chromosomes
        x_r1 = self.population[parents[0]]
        x_r2 = self.population[parents[1]]
        x_r3 = self.population[parents[2]]
        x_current = self.population[idx]

        # Create offspring
        offspring = x_current.copy()

        # Differential evolution
        for i in range(len(offspring)):
            if i == self.pkg_to_idx[self.package_name]:
                continue  # Skip main package

            if random.random() < self.CR:
                # DE/rand/1 strategy
                val = x_r1[i] + self.F * (x_r2[i] - x_r3[i])
                # Binary discretization
                offspring[i] = 1 if random.random() < self.sigmoid(val) else 0

        return offspring

    def sigmoid(self, x):
        """Sigmoid function for binary discretization"""
        return 1.0 / (1.0 + np.exp(-x))

    def polynomial_mutation(self, chromosome):
        """
        Polynomial mutation for binary encoding
        """
        mutated = chromosome.copy()
        main_idx = self.pkg_to_idx[self.package_name]

        for i in range(len(mutated)):
            if i == main_idx:
                continue

            if random.random() < self.pm:
                # Flip bit
                mutated[i] = 1 - mutated[i]

        return mutated

    def repair_solution(self, chromosome):
        """
        Repair solution to satisfy constraints
        """
        selected = np.sum(chromosome)
        main_idx = self.pkg_to_idx[self.package_name]

        # Ensure main package is not selected
        chromosome[main_idx] = 0

        if selected < self.min_size:
            # Add packages based on relationship strength
            zero_indices = np.where(chromosome == 0)[0]
            zero_indices = zero_indices[zero_indices != main_idx]

            if len(zero_indices) > 0:
                # Use co-usage as heuristic
                co_usage_scores = self.rel_matrix[main_idx, zero_indices]
                n_add = min(self.min_size - selected, len(zero_indices))
                add_indices = zero_indices[np.argsort(co_usage_scores)[-n_add:]]
                chromosome[add_indices] = 1

        elif selected > self.max_size:
            # Remove packages with lowest contribution
            one_indices = np.where(chromosome == 1)[0]
            n_remove = selected - self.max_size

            if len(one_indices) > n_remove:
                # Use combined score for removal
                scores = []
                for idx in one_indices:
                    co_usage = self.rel_matrix[main_idx, idx]
                    similarity = self.sim_matrix[main_idx, idx]
                    scores.append(co_usage + similarity)

                scores = np.array(scores)
                remove_indices = one_indices[np.argsort(scores)[:n_remove]]
                chromosome[remove_indices] = 0

        return chromosome

    def update_reference_point(self, objectives):
        """
        Update ideal and nadir points
        """
        self.z = np.minimum(self.z, objectives)
        self.z_nad = np.maximum(self.z_nad, objectives)

    def update_external_population(self, solutions, objectives_list):
        """
        Update external population with non-dominated solutions
        """
        for sol, obj in zip(solutions, objectives_list):
            sol = sol if isinstance(sol, np.ndarray) else np.array([sol])
            obj = obj if isinstance(obj, np.ndarray) else np.array([obj])

            if len(obj.shape) == 1:
                self.add_to_external_population(sol, obj)

    def add_to_external_population(self, solution, objectives):
        """
        Add solution to external population if non-dominated
        """
        # Check domination
        to_remove = []

        for i, (ep_sol, ep_obj) in enumerate(self.external_population):
            if self.dominates(objectives, ep_obj):
                to_remove.append(i)
            elif self.dominates(ep_obj, objectives):
                return  # Solution is dominated, don't add

        # Remove dominated solutions
        for i in reversed(to_remove):
            del self.external_population[i]

        # Add new solution
        self.external_population.append((solution.copy(), objectives.copy()))

        # Limit size using crowding distance
        if len(self.external_population) > self.pop_size * 2:
            self.prune_external_population()

    def dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2 (for minimization)
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def prune_external_population(self):
        """
        Prune external population using crowding distance
        """
        if len(self.external_population) <= self.pop_size:
            return

        objectives = np.array([obj for _, obj in self.external_population])

        # Calculate crowding distance
        n_points = len(objectives)
        n_obj = objectives.shape[1]
        crowding_distance = np.zeros(n_points)

        for m in range(n_obj):
            # Sort by objective m
            sorted_idx = np.argsort(objectives[:, m])

            # Boundary points get infinite distance
            crowding_distance[sorted_idx[0]] = np.inf
            crowding_distance[sorted_idx[-1]] = np.inf

            # Calculate distances for interior points
            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
            if obj_range > 0:
                for i in range(1, n_points - 1):
                    crowding_distance[sorted_idx[i]] += (
                        objectives[sorted_idx[i + 1], m] -
                        objectives[sorted_idx[i - 1], m]
                    ) / obj_range

        # Keep solutions with highest crowding distance
        keep_idx = np.argsort(crowding_distance)[-self.pop_size:]
        self.external_population = [self.external_population[i] for i in keep_idx]

    def run(self):
        """
        Main MOEA/D algorithm loop (following original paper)
        """
        start_time = time.time()

        print(f"Starting MOEA/D optimization (correct implementation)...")

        for gen in range(self.max_gen):
            # Permutation for random order
            perm = np.random.permutation(self.pop_size)

            for i in perm:
                # Step 2.1: Selection of mating pool
                if random.random() < self.delta:
                    pool = self.B[i].tolist()  # Neighborhood
                else:
                    pool = list(range(self.pop_size))  # Whole population

                # Step 2.2: Reproduction
                offspring = self.differential_evolution_crossover(i, pool)

                # Step 2.3: Mutation
                offspring = self.polynomial_mutation(offspring)

                # Step 2.4: Repair
                offspring = self.repair_solution(offspring)

                # Evaluation
                offspring_obj = self.evaluate_objectives(offspring)

                # Step 2.5: Update reference point
                self.update_reference_point(offspring_obj)

                # Step 2.6: Update neighboring solutions
                c = 0  # Counter for replacements

                # Random order for fairness
                update_indices = self.B[i].copy()
                np.random.shuffle(update_indices)

                for j in update_indices:
                    # Calculate fitness values
                    current_fitness = self.evaluate_subproblem(
                        self.objectives[j], self.weights[j]
                    )
                    offspring_fitness = self.evaluate_subproblem(
                        offspring_obj, self.weights[j]
                    )

                    # Replace if offspring is better
                    if offspring_fitness < current_fitness:
                        self.population[j] = offspring.copy()
                        self.objectives[j] = offspring_obj.copy()
                        c += 1

                    # Limit replacements
                    if c >= self.nr:
                        break

                # Step 2.7: Update external population
                self.add_to_external_population(offspring, offspring_obj)

            # Progress report
            if gen % 10 == 0 or gen == self.max_gen - 1:
                elapsed = time.time() - start_time

                # Calculate metrics
                avg_f1 = -np.mean(self.objectives[:, 0])
                avg_f2 = -np.mean(self.objectives[:, 1])
                avg_f3 = np.mean(self.objectives[:, 2])

                print(f"Generation {gen}/{self.max_gen} ({elapsed:.2f}s)")
                print(f"  EP size: {len(self.external_population)}")
                print(f"  Reference point: {self.z}")
                print(f"  Avg F1 (co-usage): {avg_f1:.4f}")
                print(f"  Avg F2 (similarity): {avg_f2:.4f}")
                print(f"  Avg F3 (size): {avg_f3:.2f}")

        elapsed = time.time() - start_time
        print(f"MOEA/D completed in {elapsed:.2f}s")
        print(f"Final external population: {len(self.external_population)} solutions")

        return self.external_population

    def format_results(self, solutions):
        """
        Format solutions into DataFrame
        """
        results = []
        unique_sets = {}

        for sol, obj in solutions:
            # Convert back to maximization for display
            f1 = -obj[0]  # Co-usage
            f2 = -obj[1]  # Similarity
            f3 = obj[2]   # Size

            # Get package names
            selected_indices = np.where(sol == 1)[0]
            packages = sorted([self.package_names[i] for i in selected_indices])

            # Create unique key
            package_key = ','.join(packages)

            # Keep best version of duplicate sets
            combined_score = f1 + f2 - 0.1 * f3

            if package_key not in unique_sets or combined_score > unique_sets[package_key]['score']:
                unique_sets[package_key] = {
                    'linked_usage': f1,
                    'semantic_similarity': f2,
                    'size': f3,
                    'recommended_packages': package_key,
                    'score': combined_score
                }

        # Convert to list
        for i, data in enumerate(unique_sets.values(), 1):
            data['solution_id'] = i
            del data['score']
            results.append(data)

        # Create DataFrame
        df = pd.DataFrame(results)

        if len(df) > 0:
            # Sort by objectives
            df = df.sort_values(
                by=['size', 'semantic_similarity', 'linked_usage'],
                ascending=[True, False, False]
            )

            # Renumber solutions
            df['solution_id'] = range(1, len(df) + 1)

        print(f"Found {len(df)} unique solutions")

        return df

    def save_results(self, df, output_dir='results'):
        """
        Save results to CSV
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{self.package_name}_moead_correct_recommendations.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='MOEA/D (Correct) for package recommendation')
    parser.add_argument('package', type=str, help='Package name')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--neighbors', type=int, default=20, help='Neighborhood size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--decomposition', type=str, default='tchebycheff',
                       choices=['tchebycheff', 'pbi'], help='Decomposition method')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    try:
        # Run MOEA/D
        moead = MOEAD_Correct(
            args.package,
            args.pop_size,
            args.neighbors,
            args.generations,
            args.decomposition
        )
        solutions = moead.run()

        # Format and save results
        results_df = moead.format_results(solutions)
        moead.save_results(results_df, args.output_dir)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()