"""
Incremental MOVNS Improvements
Focus: Simple, testable improvements to increase HV
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized


class MOVNSImproved(MOVNS_V2):
    """
    Incremental improvements to MOVNS v2
    """
    
    def __init__(self, main_package, archive_size=100, max_iterations=40):
        super().__init__(main_package, archive_size, max_iterations,
                        k_max=4, track_metrics=True, min_no_improvement=10)
        
        self.intensification_rate = 0.3
        self.local_search_intensity = 15
        
    def improved_mobi_p(self, solution):
        """Improved MOBI/P with more neighbors"""
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)
        
        candidates = []
        
        indices = np.where(solution == 1)[0]
        
        if len(indices) < self.max_size:
            cooccur_scores = self.rel_matrix[self.main_package_idx].toarray().flatten()
            cooccur_scores[indices] = -np.inf
            top_candidates = np.argsort(cooccur_scores)[-20:]
            
            for candidate in top_candidates:
                if solution[candidate] == 0:
                    test_solution = solution.copy()
                    test_solution[candidate] = 1
                    test_obj = self.evaluate_objectives(test_solution)
                    
                    if not self.dominates(best_objectives, test_obj):
                        candidates.append((test_solution, test_obj))
                        if self.dominates(test_obj, best_objectives):
                            best_solution = test_solution
                            best_objectives = test_obj
        
        if len(indices) > self.min_size:
            for idx in indices[:5]:
                test_solution = solution.copy()
                test_solution[idx] = 0
                test_obj = self.evaluate_objectives(test_solution)
                
                if not self.dominates(best_objectives, test_obj):
                    candidates.append((test_solution, test_obj))
        
        if candidates:
            non_dominated = []
            for sol, obj in candidates:
                is_dominated = False
                for _, other_obj in candidates:
                    if self.dominates(other_obj, obj):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(sol)
            
            if non_dominated:
                return non_dominated[np.random.randint(len(non_dominated))]
        
        return best_solution
    
    def improved_shake(self, solution, intensity):
        """Improved shaking with guided perturbation"""
        shaken = solution.copy()
        indices = np.where(shaken == 1)[0]
        
        for _ in range(intensity):
            if np.random.rand() < 0.5 and len(indices) < self.max_size:
                candidates = self.semantic_candidates[:30]
                valid = [c for c in candidates if shaken[c] == 0]
                if valid:
                    shaken[np.random.choice(valid)] = 1
            elif len(indices) > self.min_size:
                idx_to_remove = np.random.choice(indices)
                shaken[idx_to_remove] = 0
            
            indices = np.where(shaken == 1)[0]
        
        return self.repair_solution(shaken)
    
    def run(self):
        """Improved main loop"""
        print(f"\nStarting Improved MOVNS for {self.main_package}...")
        
        no_improvement = 0
        best_hv = 0
        
        for iteration in range(self.max_iterations):
            if len(self.archive) > 0:
                if np.random.rand() < self.intensification_rate:
                    idx = np.random.randint(min(10, len(self.archive)))
                    current = self.archive[idx]['chromosome'].copy()
                else:
                    current = self.select_unexplored_solution()
            else:
                current = self.select_unexplored_solution()
            
            for k in range(self.k_max):
                x_prime = self.improved_shake(current, k + 1)
                
                for _ in range(self.local_search_intensity):
                    x_local = self.improved_mobi_p(x_prime)
                    if not np.array_equal(x_local, x_prime):
                        x_prime = x_local
                    else:
                        break
                
                x_obj = self.evaluate_objectives(x_prime)
                current_obj = self.evaluate_objectives(current)
                
                if self.dominates(x_obj, current_obj):
                    current = x_prime
                    break
            
            obj = self.evaluate_objectives(current)
            self.update_archive(current, obj)
            
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    current_hv = metrics.get('hypervolume', 0)
                    
                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    print(f"Iter {iteration}: Archive={len(self.archive)}, "
                          f"HV={current_hv:.4f}, Best={best_hv:.4f}")
                    
                    if no_improvement >= self.min_no_improvement:
                        print(f"Early stopping at iteration {iteration}")
                        break
        
        print(f"\nCompleted: {len(self.archive)} solutions, HV={best_hv:.4f}")
        return self.archive


def test_improvements():
    """Test incremental improvements"""
    print("="*60)
    print("TESTING INCREMENTAL MOVNS IMPROVEMENTS")
    print("="*60)
    
    results = {}
    
    print("\n1. Testing MOVNS v2 (baseline)...")
    algo = MOVNS_V2('fastapi', archive_size=100, max_iterations=30)
    start = time.time()
    solutions = algo.run()
    v2_time = time.time() - start
    
    metrics = algo.get_metrics_history()
    v2_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0
    results['v2'] = {'hv': v2_hv, 'time': v2_time, 'solutions': len(solutions)}
    print(f"Result: HV={v2_hv:.4f}, Time={v2_time:.1f}s, Solutions={len(solutions)}")
    
    print("\n2. Testing MOVNS Improved...")
    algo = MOVNSImproved('fastapi', archive_size=100, max_iterations=30)
    start = time.time()
    solutions = algo.run()
    imp_time = time.time() - start
    
    metrics = algo.get_metrics_history()
    imp_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0
    results['improved'] = {'hv': imp_hv, 'time': imp_time, 'solutions': len(solutions)}
    print(f"Result: HV={imp_hv:.4f}, Time={imp_time:.1f}s, Solutions={len(solutions)}")
    
    print("\n3. Testing MOEA/D (target)...")
    algo = MOEAD_Normalized('fastapi', pop_size=100, max_gen=30)
    start = time.time()
    solutions = algo.run()
    moead_time = time.time() - start
    
    metrics = algo.get_metrics_history()
    moead_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0
    results['moead'] = {'hv': moead_hv, 'time': moead_time, 'solutions': len(solutions)}
    print(f"Result: HV={moead_hv:.4f}, Time={moead_time:.1f}s, Solutions={len(solutions)}")
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nHypervolume:")
    print(f"  MOVNS v2:       {v2_hv:.4f}")
    print(f"  MOVNS Improved: {imp_hv:.4f}")
    print(f"  MOEA/D:         {moead_hv:.4f}")
    
    print(f"\nImprovements:")
    if v2_hv > 0:
        print(f"  Improved vs v2:     {(imp_hv/v2_hv - 1)*100:+.1f}%")
    else:
        print(f"  Improved vs v2:     N/A (v2 HV is 0)")

    if moead_hv > 0:
        print(f"  Improved vs MOEA/D: {(imp_hv/moead_hv - 1)*100:+.1f}%")
        print(f"  As % of MOEA/D:     {(imp_hv/moead_hv)*100:.1f}%")
    else:
        print(f"  Improved vs MOEA/D: N/A (MOEA/D HV is 0)")
    
    if imp_hv > v2_hv:
        print("\n✓ SUCCESS: Improvements work!")
        print("Key improvements:")
        print("  - More intensive local search")
        print("  - Better MOBI/P with more neighbors")
        print("  - Guided shaking")
        print("  - Intensification from elite archive")
    else:
        print("\n✗ No improvement achieved")
    
    return results


if __name__ == "__main__":
    results = test_improvements()