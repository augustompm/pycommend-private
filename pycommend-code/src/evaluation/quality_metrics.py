"""
Quality Metrics for Multi-Objective Optimization
Implementation of key metrics: Hypervolume, IGD+, Spacing, Spread
"""

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
import warnings


class QualityMetrics:
    """
    Complete suite of quality metrics for multi-objective optimization
    """

    def __init__(self, reference_set=None, ideal_point=None, nadir_point=None):
        """
        Initialize metrics calculator

        Args:
            reference_set: Reference Pareto front for IGD calculations
            ideal_point: Ideal point for normalization
            nadir_point: Nadir point for normalization
        """
        self.reference_set = reference_set
        self.ideal_point = ideal_point
        self.nadir_point = nadir_point

    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0, 1] range
        """
        if self.ideal_point is None:
            self.ideal_point = np.min(objectives, axis=0)
        if self.nadir_point is None:
            self.nadir_point = np.max(objectives, axis=0)

        # Avoid division by zero
        range_obj = self.nadir_point - self.ideal_point
        range_obj[range_obj == 0] = 1.0

        return (objectives - self.ideal_point) / range_obj

    def filter_dominated(self, objectives):
        """
        Filter dominated solutions, keep only non-dominated
        """
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if j dominates i
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break

        return objectives[~is_dominated]

    def hypervolume(self, objectives, ref_point=None):
        """
        Calculate hypervolume indicator

        Args:
            objectives: Set of objective vectors
            ref_point: Reference point (default: 1.1 * max for each objective)

        Returns:
            Hypervolume value
        """
        if len(objectives) == 0:
            return 0.0

        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)

        # Set reference point if not provided
        if ref_point is None:
            ref_point = np.ones(norm_obj.shape[1]) * 1.1

        # Filter dominated solutions
        pareto_front = self.filter_dominated(norm_obj)

        if len(pareto_front) == 0:
            return 0.0

        # Use WFG algorithm for 2D and 3D, Monte Carlo for higher dimensions
        n_obj = pareto_front.shape[1]

        if n_obj == 2:
            return self._hv_2d(pareto_front, ref_point)
        elif n_obj == 3:
            return self._hv_3d(pareto_front, ref_point)
        else:
            return self._hv_monte_carlo(pareto_front, ref_point)

    def _hv_2d(self, points, ref_point):
        """
        Calculate 2D hypervolume exactly
        """
        # Sort points by first objective
        points = points[points[:, 0].argsort()]

        volume = 0.0
        prev_x = 0.0

        for point in points:
            if point[0] > ref_point[0] or point[1] > ref_point[1]:
                continue

            width = point[0] - prev_x
            height = ref_point[1] - point[1]

            if width > 0 and height > 0:
                volume += width * height

            prev_x = point[0]

        # Add last rectangle
        if prev_x < ref_point[0]:
            volume += (ref_point[0] - prev_x) * ref_point[1]

        return volume

    def _hv_3d(self, points, ref_point):
        """
        Calculate 3D hypervolume (simplified)
        """
        # Use inclusion-exclusion principle
        volume = 0.0

        for point in points:
            if np.any(point > ref_point):
                continue

            # Volume of box from point to reference
            box_volume = np.prod(ref_point - point)
            volume += box_volume

        # This is an approximation - exact 3D HV is complex
        # For exact calculation, use external library
        return volume / len(points)  # Average to avoid overcounting

    def _hv_monte_carlo(self, points, ref_point, n_samples=10000):
        """
        Monte Carlo approximation for high-dimensional hypervolume
        """
        # Generate random samples in the reference box
        samples = np.random.uniform(0, ref_point, (n_samples, len(ref_point)))

        # Count samples dominated by at least one point
        dominated_count = 0

        for sample in samples:
            for point in points:
                if np.all(point <= sample):
                    dominated_count += 1
                    break

        # Estimate hypervolume
        ref_volume = np.prod(ref_point)
        return (dominated_count / n_samples) * ref_volume

    def igd_plus(self, objectives, reference_set=None):
        """
        Calculate IGD+ (Inverted Generational Distance Plus)

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front

        Returns:
            IGD+ value (lower is better)
        """
        if reference_set is None:
            if self.reference_set is None:
                raise ValueError("Reference set required for IGD+")
            reference_set = self.reference_set

        # Normalize both sets
        norm_obj = self.normalize_objectives(objectives)
        norm_ref = self.normalize_objectives(reference_set)

        total_distance = 0.0

        for ref_point in norm_ref:
            min_distance = float('inf')

            for obj_point in norm_obj:
                # IGD+ distance (only counts where ref is worse)
                diff = np.maximum(ref_point - obj_point, 0)
                distance = np.linalg.norm(diff)

                if distance < min_distance:
                    min_distance = distance

            total_distance += min_distance

        return total_distance / len(norm_ref)

    def igd(self, objectives, reference_set=None):
        """
        Calculate standard IGD (for comparison)

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front

        Returns:
            IGD value (lower is better)
        """
        if reference_set is None:
            if self.reference_set is None:
                raise ValueError("Reference set required for IGD")
            reference_set = self.reference_set

        # Normalize both sets
        norm_obj = self.normalize_objectives(objectives)
        norm_ref = self.normalize_objectives(reference_set)

        # Calculate minimum distances
        distances = cdist(norm_ref, norm_obj)
        min_distances = np.min(distances, axis=1)

        return np.mean(min_distances)

    def spacing(self, objectives):
        """
        Calculate spacing metric (uniformity of distribution)

        Args:
            objectives: Set of objective vectors

        Returns:
            Spacing value (lower is better)
        """
        if len(objectives) < 2:
            return 0.0

        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)

        n = len(norm_obj)
        distances = []

        # Calculate minimum distance for each point
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(norm_obj[i] - norm_obj[j])
                    if dist < min_dist:
                        min_dist = dist
            distances.append(min_dist)

        # Calculate spacing
        mean_dist = np.mean(distances)
        if mean_dist == 0:
            return 0.0

        spacing = np.sqrt(np.sum((distances - mean_dist) ** 2) / (n - 1))
        return spacing

    def spread(self, objectives):
        """
        Calculate spread metric (distribution and extent)

        Args:
            objectives: Set of objective vectors

        Returns:
            Spread value (lower is better)
        """
        if len(objectives) < 3:
            return 1.0

        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)

        n = len(norm_obj)
        n_obj = norm_obj.shape[1]

        # Find extreme points for each objective
        extreme_points = []
        for i in range(n_obj):
            min_idx = np.argmin(norm_obj[:, i])
            max_idx = np.argmax(norm_obj[:, i])
            extreme_points.extend([min_idx, max_idx])

        extreme_points = list(set(extreme_points))

        # Calculate consecutive distances (simplified for any dimension)
        distances = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(norm_obj[i] - norm_obj[j])
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                distances.append(min_dist)

        if len(distances) == 0:
            return 1.0

        mean_dist = np.mean(distances)

        # Calculate distances to extremes
        df = dl = 0
        if len(extreme_points) >= 2:
            # Distance to first extreme
            df = np.min([np.linalg.norm(norm_obj[i] - norm_obj[extreme_points[0]])
                        for i in range(n) if i != extreme_points[0]])
            # Distance to last extreme
            dl = np.min([np.linalg.norm(norm_obj[i] - norm_obj[extreme_points[-1]])
                        for i in range(n) if i != extreme_points[-1]])

        # Calculate spread
        numerator = df + dl + np.sum(np.abs(distances - mean_dist))
        denominator = df + dl + (len(distances)) * mean_dist

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def diversity(self, objectives):
        """
        Calculate diversity metric

        Args:
            objectives: Set of objective vectors

        Returns:
            Diversity value (higher is better)
        """
        if len(objectives) < 2:
            return 0.0

        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)

        # Calculate pairwise distances
        distances = cdist(norm_obj, norm_obj)
        np.fill_diagonal(distances, 0)

        # Average distance to k nearest neighbors
        k = min(5, len(norm_obj) - 1)
        diversity_scores = []

        for i in range(len(norm_obj)):
            row_distances = distances[i]
            row_distances[i] = float('inf')
            k_nearest = np.sort(row_distances)[:k]
            diversity_scores.append(np.mean(k_nearest))

        return np.mean(diversity_scores)

    def maximum_spread(self, objectives):
        """
        Calculate maximum spread in each objective

        Args:
            objectives: Set of objective vectors

        Returns:
            Maximum spread value (higher is better)
        """
        # Normalize objectives
        norm_obj = self.normalize_objectives(objectives)

        # Calculate range in each objective
        spreads = []
        for i in range(norm_obj.shape[1]):
            obj_range = np.max(norm_obj[:, i]) - np.min(norm_obj[:, i])
            spreads.append(obj_range ** 2)

        return np.sqrt(np.sum(spreads))

    def evaluate_all(self, objectives, reference_set=None):
        """
        Calculate all metrics at once

        Args:
            objectives: Set of objective vectors
            reference_set: Reference Pareto front (optional)

        Returns:
            Dictionary with all metric values
        """
        results = {
            'n_solutions': len(objectives),
            'n_nondominated': len(self.filter_dominated(objectives))
        }

        # Calculate metrics that don't need reference set
        results['hypervolume'] = self.hypervolume(objectives)
        results['spacing'] = self.spacing(objectives)
        results['spread'] = self.spread(objectives)
        results['diversity'] = self.diversity(objectives)
        results['maximum_spread'] = self.maximum_spread(objectives)

        # Calculate metrics that need reference set
        if reference_set is not None or self.reference_set is not None:
            results['igd'] = self.igd(objectives, reference_set)
            results['igd_plus'] = self.igd_plus(objectives, reference_set)

        return results

    def generate_reference_set(self, n_points, n_objectives):
        """
        Generate uniform reference points for IGD calculation

        Args:
            n_points: Number of reference points
            n_objectives: Number of objectives

        Returns:
            Array of reference points
        """
        # Use Das-Dennis method
        if n_objectives == 2:
            # For 2 objectives, create uniform line
            weights = np.linspace(0, 1, n_points)
            ref_points = np.column_stack([weights, 1 - weights])

        elif n_objectives == 3:
            # For 3 objectives, create uniform simplex
            ref_points = []
            h = int(np.sqrt(n_points))

            for i in range(h):
                for j in range(h):
                    if i + j < h:
                        w1 = i / h
                        w2 = j / h
                        w3 = 1 - w1 - w2
                        ref_points.append([w1, w2, w3])

            ref_points = np.array(ref_points)

            # Add random points if needed
            while len(ref_points) < n_points:
                w = np.random.dirichlet(np.ones(n_objectives))
                ref_points = np.vstack([ref_points, w])

            ref_points = ref_points[:n_points]

        else:
            # For many objectives, use random sampling
            ref_points = np.random.dirichlet(np.ones(n_objectives), n_points)

        return ref_points


def compare_algorithms(objectives1, objectives2, name1="Algorithm 1", name2="Algorithm 2"):
    """
    Compare two algorithms using quality metrics

    Args:
        objectives1: Objectives from algorithm 1
        objectives2: Objectives from algorithm 2
        name1: Name of algorithm 1
        name2: Name of algorithm 2

    Returns:
        Comparison results dictionary
    """
    metrics = QualityMetrics()

    # Calculate metrics for both
    results1 = metrics.evaluate_all(objectives1)
    results2 = metrics.evaluate_all(objectives2)

    # Compare
    comparison = {
        'algorithm_1': name1,
        'algorithm_2': name2,
        'metrics': {}
    }

    for metric in results1.keys():
        value1 = results1[metric]
        value2 = results2[metric]

        # Determine winner based on metric type
        if metric in ['hypervolume', 'diversity', 'maximum_spread', 'n_solutions', 'n_nondominated']:
            # Higher is better
            winner = name1 if value1 > value2 else name2 if value2 > value1 else 'tie'
        else:
            # Lower is better
            winner = name1 if value1 < value2 else name2 if value2 < value1 else 'tie'

        comparison['metrics'][metric] = {
            name1: value1,
            name2: value2,
            'winner': winner,
            'difference': abs(value1 - value2)
        }

    return comparison