"""
MOVNS Efficient - Versão eficiente que mantém 80-100 soluções
Baseado no MOVNS v2 mas com inicialização rápida
"""

import numpy as np
import random
import sys
import os
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from optimizer.movns_v2 import MOVNS_V2


class MOVNS_Efficient(MOVNS_V2):
    """
    MOVNS Efficient - Rápido e eficaz
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):

        # Inicializar parent mas NÃO executar initialize_archive dele
        self.skip_initial_archive = True

        super().__init__(main_package, archive_size, max_iterations,
                        track_metrics=track_metrics)

        self.min_archive = 80

        # Fazer nossa própria inicialização rápida
        self.fast_initialize()

        print(f"MOVNS Efficient initialized with {len(self.archive)} solutions")

    def fast_initialize(self):
        """Inicialização rápida para 80 soluções"""

        # Limpar arquivo se já existir
        self.archive = []

        # Gerar 80 soluções rapidamente
        for i in range(self.min_archive):
            solution = np.zeros(len(self.package_names), dtype=int)

            # Variar estratégia
            if i < 30 and hasattr(self, 'cooccur_candidates'):
                # Baseado em co-ocorrência
                n_select = np.random.randint(3, 7)
                candidates = self.cooccur_candidates[:100]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
            elif i < 60 and hasattr(self, 'semantic_candidates'):
                # Baseado em semântica
                n_select = np.random.randint(3, 7)
                candidates = self.semantic_candidates[:100]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
            else:
                # Aleatório
                n_select = np.random.randint(2, 8)
                indices = np.random.choice(len(solution), n_select, replace=False)
                solution[indices] = 1

            solution[self.main_package_idx] = 1

            # Adicionar diretamente sem update_archive (mais rápido)
            objectives = self.evaluate_objectives(solution)
            self.archive.append({
                'chromosome': solution.copy(),
                'objectives': objectives.copy()
            })

        # Remover dominados apenas uma vez no final
        self.remove_dominated()

    def remove_dominated(self):
        """Remove soluções dominadas do arquivo"""
        non_dominated = []

        for i, sol_i in enumerate(self.archive):
            is_dominated = False
            for j, sol_j in enumerate(self.archive):
                if i != j and self.dominates(sol_j['objectives'], sol_i['objectives']):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(sol_i)

        self.archive = non_dominated

    def run(self) -> List[Dict]:
        """Execução principal eficiente"""
        print(f"\nStarting MOVNS Efficient for {self.main_package}...")
        print("="*60)

        best_hv = 0
        no_improvement = 0

        for iteration in range(self.max_iterations):

            # Garantir mínimo de soluções
            while len(self.archive) < self.min_archive:
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 8)
                indices = np.random.choice(len(solution), n_select, replace=False)
                solution[indices] = 1
                solution[self.main_package_idx] = 1
                objectives = self.evaluate_objectives(solution)
                self.archive.append({
                    'chromosome': solution.copy(),
                    'objectives': objectives.copy()
                })

            # VNS melhorado mas ainda eficiente
            if len(self.archive) > 0:
                # Selecionar 2 pais para intensificação
                num_parents = min(2, len(self.archive))
                if num_parents > 0:
                    parent_indices = np.random.choice(len(self.archive), num_parents, replace=False)
                else:
                    parent_indices = []

                for parent_idx in parent_indices:
                    # Re-check archive size as update_archive may remove solutions
                    if parent_idx >= len(self.archive):
                        continue
                    parent = self.archive[parent_idx]['chromosome'].copy()
                    parent_obj = self.evaluate_objectives(parent)
                    best = parent.copy()
                    best_obj = parent_obj.copy()

                    # Tentar 5 neighborhoods para melhor intensificação
                    for k in range(5):
                        neighbor = best.copy()

                        if k == 0:
                            # Add high cooccurrence package
                            if hasattr(self, 'cooccur_candidates'):
                                inactive = np.where(neighbor == 0)[0]
                                candidates = [i for i in self.cooccur_candidates[:50] if i in inactive]
                                if candidates:
                                    idx = np.random.choice(candidates)
                                    neighbor[idx] = 1
                        elif k == 1:
                            # Add high semantic similarity package
                            if hasattr(self, 'semantic_candidates'):
                                inactive = np.where(neighbor == 0)[0]
                                candidates = [i for i in self.semantic_candidates[:50] if i in inactive]
                                if candidates:
                                    idx = np.random.choice(candidates)
                                    neighbor[idx] = 1
                        elif k == 2:
                            # Remove low contribution package
                            active = np.where(neighbor == 1)[0]
                            if len(active) > 3:
                                removable = [i for i in active if i != self.main_package_idx]
                                if removable:
                                    idx = np.random.choice(removable)
                                    neighbor[idx] = 0
                        elif k == 3:
                            # Smart swap based on objectives
                            active = np.where(neighbor == 1)[0]
                            inactive = np.where(neighbor == 0)[0]
                            if len(active) > 2 and len(inactive) > 0:
                                # Remove random, add from candidates
                                remove_idx = np.random.choice([i for i in active if i != self.main_package_idx])
                                if hasattr(self, 'cooccur_candidates'):
                                    candidates = [i for i in self.cooccur_candidates[:100] if i in inactive]
                                    if candidates:
                                        add_idx = np.random.choice(candidates)
                                        neighbor[remove_idx] = 0
                                        neighbor[add_idx] = 1
                        else:
                            # Multi-flip for diversity
                            n_changes = np.random.randint(1, 4)
                            for _ in range(n_changes):
                                idx = np.random.randint(len(neighbor))
                                if idx != self.main_package_idx:
                                    neighbor[idx] = 1 - neighbor[idx]

                        # Avaliar e manter melhor
                        if not np.array_equal(neighbor, best):
                            neighbor_obj = self.evaluate_objectives(neighbor)

                            if self.dominates(neighbor_obj, best_obj):
                                best = neighbor
                                best_obj = neighbor_obj
                                self.update_archive(neighbor, neighbor_obj)
                            elif not self.dominates(best_obj, neighbor_obj):
                                # Adicionar para diversidade
                                self.update_archive(neighbor, neighbor_obj)

            # Calcular métricas
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in metrics:
                        if key in self.metrics_history:
                            self.metrics_history[key].append(metrics[key])

                    if 'hypervolume' in metrics:
                        current_hv = metrics['hypervolume']

                        if current_hv > best_hv:
                            best_hv = current_hv
                            no_improvement = 0
                        else:
                            no_improvement += 1

                        if iteration % 10 == 0:
                            print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                                  f"HV={current_hv:.4f}")

            # Early stopping
            if no_improvement >= 15:
                print(f"Converged at iteration {iteration}")
                break

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                for key in final_metrics:
                    if key in self.metrics_history:
                        self.metrics_history[key].append(final_metrics[key])
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_efficient.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_Efficient(
        package,
        archive_size=100,
        max_iterations=20,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    print(f"Target achieved: {80 <= len(solutions) <= 100}")


if __name__ == "__main__":
    main()