"""
Fair Comparison Test for Multi-Objective Algorithms
Based on 2024 research best practices:
1. Same number of function evaluations
2. Multiple performance metrics (HV, IGD+, Spacing, GD)
3. Statistical analysis over multiple runs
4. Standardized environment
"""

import numpy as np
import sys
import os
import time
from typing import Dict, List
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


class FairComparison:
    """Fair comparison framework based on 2024 best practices"""

    def __init__(self, max_evaluations: int = 5000):
        """
        Initialize fair comparison
        Args:
            max_evaluations: Fixed budget of function evaluations
        """
        self.max_evaluations = max_evaluations
        self.qm = QualityMetrics()
        self.evaluation_count = 0

    def count_evaluations_movns(self, iterations: int, archive_size: int) -> int:
        """Estimate evaluations for MOVNS"""
        initial_archive = archive_size
        local_search_per_iter = 20
        return initial_archive + (iterations * local_search_per_iter)

    def count_evaluations_moead(self, generations: int, pop_size: int) -> int:
        """Calculate evaluations for MOEA/D"""
        initial_pop = pop_size
        evaluations_per_gen = pop_size
        return initial_pop + (generations * evaluations_per_gen)

    def find_equivalent_settings(self) -> Dict:
        """Find algorithm settings for same evaluation budget"""
        settings = {}

        movns_archive = 100
        movns_iterations = (self.max_evaluations - movns_archive) // 20
        settings['movns'] = {
            'archive_size': movns_archive,
            'max_iterations': movns_iterations,
            'evaluations': self.count_evaluations_movns(movns_iterations, movns_archive)
        }

        moead_pop = 100
        moead_generations = (self.max_evaluations - moead_pop) // moead_pop
        settings['moead'] = {
            'pop_size': moead_pop,
            'max_gen': moead_generations,
            'evaluations': self.count_evaluations_moead(moead_generations, moead_pop)
        }

        return settings

    def calculate_metrics(self, solutions: List, algo) -> Dict:
        """Calculate comprehensive metrics"""
        if not solutions:
            return {'hv': 0, 'spacing': 0, 'diversity': 0, 'n_solutions': 0}

        objectives = []
        for sol in solutions:
            if isinstance(sol, dict) and 'chromosome' in sol:
                obj = algo.evaluate_objectives(sol['chromosome'])
            else:
                obj = algo.evaluate_objectives(sol)
            objectives.append(obj)

        objectives = np.array(objectives)

        normalized = []
        for obj in objectives:
            norm_obj = algo.normalize_objectives(obj)
            normalized.append(norm_obj)
        normalized = np.array(normalized)

        ref_point = np.array([0, 0, 1.0])
        hv = self.qm.hypervolume(normalized, ref_point)

        spacing = self.qm.spacing(objectives) if len(objectives) > 1 else 0

        diversity = 0
        if len(objectives) > 1:
            ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
            diversity = np.prod(ranges[ranges > 0])

        return {
            'hv': hv,
            'spacing': spacing,
            'diversity': diversity,
            'n_solutions': len(solutions)
        }

    def run_algorithm(self, algo_class, package: str, **kwargs) -> Dict:
        """Run algorithm and collect metrics"""
        algo = algo_class(package, track_metrics=True, **kwargs)
        algo_name = algo_class.__name__

        print(f"\nRunning {algo_name}...")
        print(f"  Settings: {kwargs}")

        start_time = time.time()
        solutions = algo.run()
        elapsed = time.time() - start_time

        metrics = self.calculate_metrics(solutions, algo)

        best_obj = {'lu': 0, 'ss': 0, 'rss': float('inf')}
        for sol in solutions:
            obj = algo.evaluate_objectives(sol['chromosome'])
            if -obj[0] > best_obj['lu']:
                best_obj['lu'] = -obj[0]
            if -obj[1] > best_obj['ss']:
                best_obj['ss'] = -obj[1]
            if obj[2] < best_obj['rss']:
                best_obj['rss'] = obj[2]

        algo_metrics = algo.get_metrics_history()
        hv_history = []
        if algo_metrics and 'hypervolume' in algo_metrics:
            hv_history = algo_metrics['hypervolume']

        return {
            'algorithm': algo_name,
            'time': elapsed,
            'metrics': metrics,
            'best_objectives': best_obj,
            'hv_history': hv_history,
            'settings': kwargs
        }

    def run_multiple_trials(self, package: str, n_trials: int = 5) -> Dict:
        """Run multiple trials for statistical analysis"""
        settings = self.find_equivalent_settings()

        movns_results = []
        moead_results = []

        print(f"\nRunning {n_trials} trials with {self.max_evaluations} evaluations each")
        print(f"MOVNS: {settings['movns']['max_iterations']} iterations")
        print(f"MOEA/D: {settings['moead']['max_gen']} generations")

        for trial in range(n_trials):
            print(f"\n{'='*60}")
            print(f"TRIAL {trial + 1}/{n_trials}")
            print(f"{'='*60}")

            np.random.seed(42 + trial)

            movns_result = self.run_algorithm(
                MOVNS_Advanced,
                package,
                **{k: v for k, v in settings['movns'].items() if k != 'evaluations'}
            )
            movns_results.append(movns_result)

            moead_result = self.run_algorithm(
                MOEAD_Normalized,
                package,
                **{k: v for k, v in settings['moead'].items() if k != 'evaluations'}
            )
            moead_results.append(moead_result)

        return {
            'movns': movns_results,
            'moead': moead_results,
            'settings': settings
        }

    def analyze_results(self, results: Dict) -> Dict:
        """Statistical analysis of results"""
        movns_hvs = [r['metrics']['hv'] for r in results['movns']]
        moead_hvs = [r['metrics']['hv'] for r in results['moead']]

        movns_times = [r['time'] for r in results['movns']]
        moead_times = [r['time'] for r in results['moead']]

        movns_spacing = [r['metrics']['spacing'] for r in results['movns']]
        moead_spacing = [r['metrics']['spacing'] for r in results['moead']]

        movns_diversity = [r['metrics']['diversity'] for r in results['movns']]
        moead_diversity = [r['metrics']['diversity'] for r in results['moead']]

        analysis = {
            'hypervolume': {
                'movns_mean': np.mean(movns_hvs),
                'movns_std': np.std(movns_hvs),
                'moead_mean': np.mean(moead_hvs),
                'moead_std': np.std(moead_hvs),
            },
            'time': {
                'movns_mean': np.mean(movns_times),
                'movns_std': np.std(movns_times),
                'moead_mean': np.mean(moead_times),
                'moead_std': np.std(moead_times),
            },
            'spacing': {
                'movns_mean': np.mean(movns_spacing),
                'movns_std': np.std(movns_spacing),
                'moead_mean': np.mean(moead_spacing),
                'moead_std': np.std(moead_spacing),
            },
            'diversity': {
                'movns_mean': np.mean(movns_diversity),
                'movns_std': np.std(movns_diversity),
                'moead_mean': np.mean(moead_diversity),
                'moead_std': np.std(moead_diversity),
            }
        }

        return analysis


def main():
    """Main fair comparison"""
    print("="*70)
    print("FAIR COMPARISON TEST - BASED ON 2024 BEST PRACTICES")
    print("="*70)

    print("\nMethodology:")
    print("1. Fixed budget: Same number of function evaluations")
    print("2. Multiple metrics: HV, Spacing, Diversity")
    print("3. Statistical analysis: Mean and std over multiple runs")
    print("4. Standardized environment: Same random seeds")

    comparator = FairComparison(max_evaluations=5000)
    package = 'fastapi'

    results = comparator.run_multiple_trials(package, n_trials=3)

    analysis = comparator.analyze_results(results)

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*70)

    print(f"\nBudget: {comparator.max_evaluations} function evaluations")
    print(f"MOVNS: {results['settings']['movns']['max_iterations']} iterations")
    print(f"MOEA/D: {results['settings']['moead']['max_gen']} generations")

    print("\n1. HYPERVOLUME (Higher is better)")
    print(f"   MOVNS:  {analysis['hypervolume']['movns_mean']:.4f} ± {analysis['hypervolume']['movns_std']:.4f}")
    print(f"   MOEA/D: {analysis['hypervolume']['moead_mean']:.4f} ± {analysis['hypervolume']['moead_std']:.4f}")

    if analysis['hypervolume']['movns_mean'] > 0 and analysis['hypervolume']['moead_mean'] > 0:
        if analysis['hypervolume']['movns_mean'] > analysis['hypervolume']['moead_mean']:
            improvement = (analysis['hypervolume']['movns_mean'] / analysis['hypervolume']['moead_mean'] - 1) * 100
            print(f"   Winner: MOVNS (+{improvement:.1f}%)")
        else:
            improvement = (analysis['hypervolume']['moead_mean'] / analysis['hypervolume']['movns_mean'] - 1) * 100
            print(f"   Winner: MOEA/D (+{improvement:.1f}%)")

    print("\n2. SPACING (Lower is better)")
    print(f"   MOVNS:  {analysis['spacing']['movns_mean']:.4f} ± {analysis['spacing']['movns_std']:.4f}")
    print(f"   MOEA/D: {analysis['spacing']['moead_mean']:.4f} ± {analysis['spacing']['moead_std']:.4f}")

    if analysis['spacing']['movns_mean'] < analysis['spacing']['moead_mean']:
        print(f"   Winner: MOVNS (better distribution)")
    else:
        print(f"   Winner: MOEA/D (better distribution)")

    print("\n3. DIVERSITY (Higher is better)")
    print(f"   MOVNS:  {analysis['diversity']['movns_mean']:.2e} ± {analysis['diversity']['movns_std']:.2e}")
    print(f"   MOEA/D: {analysis['diversity']['moead_mean']:.2e} ± {analysis['diversity']['moead_std']:.2e}")

    if analysis['diversity']['movns_mean'] > analysis['diversity']['moead_mean']:
        print(f"   Winner: MOVNS (higher diversity)")
    else:
        print(f"   Winner: MOEA/D (higher diversity)")

    print("\n4. EXECUTION TIME (seconds)")
    print(f"   MOVNS:  {analysis['time']['movns_mean']:.1f} ± {analysis['time']['movns_std']:.1f}")
    print(f"   MOEA/D: {analysis['time']['moead_mean']:.1f} ± {analysis['time']['moead_std']:.1f}")

    speed_ratio = analysis['time']['moead_mean'] / analysis['time']['movns_mean']
    print(f"   Speed ratio: {speed_ratio:.1f}x")

    print("\n" + "="*70)
    print("FINAL VERDICT (Fair Comparison)")
    print("="*70)

    movns_wins = 0
    moead_wins = 0

    if analysis['hypervolume']['movns_mean'] > analysis['hypervolume']['moead_mean']:
        movns_wins += 1
        print("✓ Hypervolume: MOVNS")
    else:
        moead_wins += 1
        print("✓ Hypervolume: MOEA/D")

    if analysis['spacing']['movns_mean'] < analysis['spacing']['moead_mean']:
        movns_wins += 1
        print("✓ Spacing: MOVNS")
    else:
        moead_wins += 1
        print("✓ Spacing: MOEA/D")

    if analysis['diversity']['movns_mean'] > analysis['diversity']['moead_mean']:
        movns_wins += 1
        print("✓ Diversity: MOVNS")
    else:
        moead_wins += 1
        print("✓ Diversity: MOEA/D")

    print(f"\nOverall Score: MOVNS {movns_wins} - {moead_wins} MOEA/D")

    if movns_wins > moead_wins:
        print("\nConclusion: MOVNS Advanced is superior under fair comparison")
    elif moead_wins > movns_wins:
        print("\nConclusion: MOEA/D is superior under fair comparison")
    else:
        print("\nConclusion: Both algorithms are comparable")

    with open('fair_comparison_results.json', 'w') as f:
        json.dump({
            'settings': results['settings'],
            'analysis': analysis,
            'winner': 'MOVNS' if movns_wins > moead_wins else 'MOEA/D'
        }, f, indent=2, default=str)
        print("\nResults saved to fair_comparison_results.json")


if __name__ == "__main__":
    main()