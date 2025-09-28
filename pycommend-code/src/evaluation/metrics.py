"""
Evaluation metrics for multi-objective optimization
"""

import numpy as np
from typing import List, Tuple


def calculate_hypervolume(solutions: List[List[float]], reference_point: List[float]) -> float:
    """
    Calculate hypervolume indicator for a set of solutions

    Args:
        solutions: List of objective values for each solution
        reference_point: Reference point for hypervolume calculation

    Returns:
        Hypervolume value
    """
    if not solutions:
        return 0.0

    solutions = np.array(solutions)
    ref = np.array(reference_point)

    # Sort solutions by first objective
    sorted_indices = np.argsort(-solutions[:, 0])
    sorted_solutions = solutions[sorted_indices]

    # Simple 3D hypervolume calculation
    volume = 0.0
    prev_x = ref[0]
    prev_y = ref[1]

    for sol in sorted_solutions:
        if np.all(sol > ref):  # Solution dominates reference point
            x_contrib = max(0, sol[0] - prev_x)
            y_contrib = max(0, sol[1] - prev_y)
            z_contrib = max(0, sol[2] - ref[2])

            volume += x_contrib * y_contrib * z_contrib

            prev_x = max(prev_x, sol[0])
            prev_y = max(prev_y, sol[1])

    # Normalize by reference volume
    ref_volume = np.prod(np.abs(ref))
    if ref_volume > 0:
        volume = volume / ref_volume

    return volume


def calculate_spacing(solutions: List[List[float]]) -> float:
    """
    Calculate spacing metric for solution distribution

    Args:
        solutions: List of objective values

    Returns:
        Spacing value (lower is better)
    """
    if len(solutions) < 2:
        return 0.0

    solutions = np.array(solutions)
    n = len(solutions)

    # Calculate minimum distance for each solution
    min_distances = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(solutions[i] - solutions[j])
                distances.append(dist)
        if distances:
            min_distances.append(min(distances))

    if not min_distances:
        return 0.0

    # Calculate spacing
    avg_dist = np.mean(min_distances)
    spacing = np.sqrt(np.sum((min_distances - avg_dist) ** 2) / len(min_distances))

    return spacing


def calculate_igd(solutions: List[List[float]], pareto_front: List[List[float]]) -> float:
    """
    Calculate Inverted Generational Distance

    Args:
        solutions: List of found solutions
        pareto_front: True Pareto front

    Returns:
        IGD value (lower is better)
    """
    if not solutions or not pareto_front:
        return float('inf')

    solutions = np.array(solutions)
    pareto = np.array(pareto_front)

    total_dist = 0.0
    for p in pareto:
        min_dist = float('inf')
        for s in solutions:
            dist = np.linalg.norm(p - s)
            min_dist = min(min_dist, dist)
        total_dist += min_dist

    return total_dist / len(pareto)