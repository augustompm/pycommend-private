"""
Unit tests for VNS neighborhoods
Evaluating and evolving each neighborhood for positive impact
Following rules.json - no shortcuts, real hypervolume calculation
"""

import sys
import os
import numpy as np
import time
from typing import Tuple, List

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS
from quality_metrics import QualityMetrics


class NeighborhoodTester:
    """
    Test and evolve VNS neighborhoods for real positive impact
    """

    def __init__(self, package='numpy'):
        self.movns = MOVNS_VNS(package, archive_size=10, max_iterations=1, track_metrics=False)
        self.metrics_calc = QualityMetrics()
        self.package = package
        self.neighborhoods = self.movns.define_neighborhoods()

    def evaluate_solution_quality(self, solution):
        """
        Calculate real quality metrics for a solution
        """
        objectives = self.movns.evaluate_objectives(solution)

        return {
            'lu': -objectives[0],
            'ss': -objectives[1],
            'rss': objectives[2],
            'objectives': objectives
        }

    def calculate_hypervolume_single(self, objectives):
        """
        Calculate hypervolume for a single solution
        """
        ref_point = np.array([0, 0, 100])
        objectives_array = objectives.reshape(1, -1)
        return self.metrics_calc.hypervolume(objectives_array, ref_point)

    def test_neighborhood_impact(self, neighborhood_idx, solution, num_samples=20):
        """
        Test impact of a specific neighborhood
        """
        neighborhood = self.neighborhoods[neighborhood_idx]
        original_quality = self.evaluate_solution_quality(solution)
        original_hv = self.calculate_hypervolume_single(original_quality['objectives'])

        improvements = []
        degradations = []
        neutrals = []

        for i in range(num_samples):
            neighbor = neighborhood(solution.copy())
            neighbor = self.movns.repair_solution(neighbor)

            neighbor_quality = self.evaluate_solution_quality(neighbor)
            neighbor_hv = self.calculate_hypervolume_single(neighbor_quality['objectives'])

            hv_diff = neighbor_hv - original_hv

            result = {
                'iteration': i,
                'hv_diff': hv_diff,
                'lu_diff': neighbor_quality['lu'] - original_quality['lu'],
                'ss_diff': neighbor_quality['ss'] - original_quality['ss'],
                'rss_diff': neighbor_quality['rss'] - original_quality['rss'],
                'solution': neighbor
            }

            if hv_diff > 0.001:
                improvements.append(result)
            elif hv_diff < -0.001:
                degradations.append(result)
            else:
                neutrals.append(result)

        return {
            'neighborhood_idx': neighborhood_idx,
            'improvements': improvements,
            'degradations': degradations,
            'neutrals': neutrals,
            'improvement_rate': len(improvements) / num_samples,
            'degradation_rate': len(degradations) / num_samples,
            'neutral_rate': len(neutrals) / num_samples,
            'avg_hv_improvement': np.mean([r['hv_diff'] for r in improvements]) if improvements else 0,
            'avg_hv_degradation': np.mean([r['hv_diff'] for r in degradations]) if degradations else 0
        }

    def evolve_neighborhood(self, neighborhood_idx):
        """
        Evolve a neighborhood to improve its impact
        """
        print(f"\nEvolving Neighborhood {neighborhood_idx + 1}")
        print("="*60)

        test_solutions = []
        for strategy in ['small', 'medium', 'large']:
            sol = self.movns.smart_initialization(strategy)
            test_solutions.append(sol)

        results = []
        for sol_idx, solution in enumerate(test_solutions):
            print(f"\nTesting with {['small', 'medium', 'large'][sol_idx]} solution:")
            result = self.test_neighborhood_impact(neighborhood_idx, solution)
            results.append(result)

            print(f"  Improvement rate: {result['improvement_rate']:.1%}")
            print(f"  Degradation rate: {result['degradation_rate']:.1%}")
            print(f"  Neutral rate: {result['neutral_rate']:.1%}")

            if result['improvements']:
                print(f"  Avg HV improvement: {result['avg_hv_improvement']:.4f}")
                best_improvement = max(result['improvements'], key=lambda x: x['hv_diff'])
                print(f"  Best HV improvement: {best_improvement['hv_diff']:.4f}")
                print(f"    LU change: {best_improvement['lu_diff']:.1f}")
                print(f"    SS change: {best_improvement['ss_diff']:.4f}")
                print(f"    RSS change: {best_improvement['rss_diff']:.0f}")

        avg_improvement_rate = np.mean([r['improvement_rate'] for r in results])

        return {
            'neighborhood_idx': neighborhood_idx,
            'avg_improvement_rate': avg_improvement_rate,
            'results': results
        }


def analyze_neighborhood_characteristics():
    """
    Analyze what each neighborhood actually does
    """
    print("\n" + "="*80)
    print("NEIGHBORHOOD CHARACTERISTICS ANALYSIS")
    print("="*80)

    tester = NeighborhoodTester('numpy')

    test_solution = tester.movns.smart_initialization('medium')
    original_size = np.sum(test_solution)
    print(f"\nOriginal solution size: {original_size}")

    for i, neighborhood in enumerate(tester.neighborhoods):
        print(f"\n--- Neighborhood {i+1} Analysis ---")

        changes = []
        for _ in range(10):
            neighbor = neighborhood(test_solution.copy())
            diff = np.sum(np.abs(neighbor - test_solution))
            new_size = np.sum(neighbor)
            size_change = new_size - original_size
            changes.append({
                'bits_changed': diff,
                'size_change': size_change
            })

        avg_bits = np.mean([c['bits_changed'] for c in changes])
        avg_size_change = np.mean([c['size_change'] for c in changes])

        print(f"  Average bits changed: {avg_bits:.1f}")
        print(f"  Average size change: {avg_size_change:+.1f}")

        if i == 0:
            print("  Type: Single bit flip")
        elif i == 1:
            print("  Type: Multi bit flip")
        elif i == 2:
            print("  Type: Segment exchange")
        elif i == 3:
            print("  Type: Smart adjustment")


def test_mobi_p_effectiveness():
    """
    Test MOBI/P local search effectiveness
    """
    print("\n" + "="*80)
    print("MOBI/P LOCAL SEARCH EFFECTIVENESS TEST")
    print("="*80)

    tester = NeighborhoodTester('numpy')

    for strategy in ['small', 'medium', 'large']:
        print(f"\nTesting MOBI/P with {strategy} solution:")

        solution = tester.movns.smart_initialization(strategy)
        original_quality = tester.evaluate_solution_quality(solution)
        original_hv = tester.calculate_hypervolume_single(original_quality['objectives'])

        print(f"  Original: LU={original_quality['lu']:.1f}, "
              f"SS={original_quality['ss']:.4f}, RSS={original_quality['rss']:.0f}")
        print(f"  Original HV: {original_hv:.4f}")

        for n_idx, neighborhood in enumerate(tester.neighborhoods[:2]):
            improved_solutions = tester.movns.mobi_p_local_search(solution, neighborhood, samples=10)

            if improved_solutions:
                best_sol, best_obj = improved_solutions[0]
                best_hv = tester.calculate_hypervolume_single(best_obj)

                print(f"\n  Neighborhood {n_idx+1} MOBI/P:")
                print(f"    Found {len(improved_solutions)} non-dominated solutions")
                print(f"    Best: LU={-best_obj[0]:.1f}, SS={-best_obj[1]:.4f}, RSS={best_obj[2]:.0f}")
                print(f"    Best HV: {best_hv:.4f} (change: {best_hv - original_hv:+.4f})")


def main():
    """
    Main testing routine following rules.json
    """
    print("="*80)
    print("VNS NEIGHBORHOODS UNIT TESTING")
    print("Following rules.json - No shortcuts, real execution")
    print("="*80)

    analyze_neighborhood_characteristics()

    tester = NeighborhoodTester('numpy')

    print("\n" + "="*80)
    print("TESTING EACH NEIGHBORHOOD")
    print("="*80)

    neighborhood_names = [
        "N1: Single Bit Flip",
        "N2: Multi Bit Flip",
        "N3: Segment Exchange",
        "N4: Smart Adjustment"
    ]

    all_results = []

    for idx, name in enumerate(neighborhood_names):
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print('='*60)

        result = tester.evolve_neighborhood(idx)
        all_results.append(result)

        if result['avg_improvement_rate'] < 0.2:
            print(f"\n[WARNING]: {name} has low improvement rate ({result['avg_improvement_rate']:.1%})")
            print("  Suggestion: This neighborhood needs redesign")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for idx, (name, result) in enumerate(zip(neighborhood_names, all_results)):
        status = "[OK]" if result['avg_improvement_rate'] >= 0.2 else "[FAIL]"
        print(f"{status} {name}: {result['avg_improvement_rate']:.1%} improvement rate")

    test_mobi_p_effectiveness()

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    for idx, (name, result) in enumerate(zip(neighborhood_names, all_results)):
        if result['avg_improvement_rate'] < 0.2:
            print(f"\n{name} needs improvement:")

            if idx == 0:
                print("  - Consider flipping bits based on contribution scores")
                print("  - Target low-contribution packages for removal")
                print("  - Add high co-occurrence packages")

            elif idx == 1:
                print("  - Use guided multi-flip based on objectives")
                print("  - Balance exploration with exploitation")
                print("  - Consider correlation between packages")

            elif idx == 2:
                print("  - Exchange based on semantic similarity")
                print("  - Ensure size constraints are respected")
                print("  - Use domain knowledge for exchanges")

            elif idx == 3:
                print("  - Refine ideal size calculation")
                print("  - Use better heuristics for package selection")
                print("  - Consider objective-specific adjustments")

    print("\nTesting complete - No shortcuts taken")


if __name__ == '__main__':
    main()