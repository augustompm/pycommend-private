"""
MOVNS Final - Calibrado para vencer HV + Spacing
Objetivo: Manter 80-100 soluções com boa distribuição
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import warnings
from optimizer.movns_v2 import MOVNS_V2

warnings.filterwarnings('ignore')


class MOVNS_Final(MOVNS_V2):
    """
    MOVNS Final calibrado:
    - Mantém 80-100 soluções no arquivo
    - Usa neighborhoods de diversidade
    - Preserva cálculo correto do HV
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):
        super().__init__(main_package, archive_size, max_iterations,
                         track_metrics=track_metrics)

        self.min_archive = 80  # Manter mínimo de 80 soluções
        self.target_archive = 100  # Alvo de 100 soluções

        print(f"MOVNS Final initialized - Calibrated for HV+Spacing")
        print(f"Target: {self.min_archive}-{self.target_archive} solutions")

    def n1_diversity_flip(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 1: Flip para diversidade"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(inactive) > 0 and len(active) < 10:
            # Adicionar pacote aleatório para diversidade
            add_idx = np.random.choice(inactive)
            neighbor[add_idx] = 1
        elif len(active) > 5:
            # Remover pacote aleatório
            remove_idx = np.random.choice(active)
            if remove_idx != self.main_package_idx:
                neighbor[remove_idx] = 0

        return neighbor

    def n2_cluster_exchange(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 2: Troca baseada em clusters para diversidade"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(active) > 3 and len(inactive) > 0:
            # Trocar elementos de clusters diferentes
            remove_idx = np.random.choice(active)
            if remove_idx != self.main_package_idx:
                neighbor[remove_idx] = 0

                # Adicionar de cluster diferente
                if hasattr(self, 'cluster_labels'):
                    remove_cluster = self.cluster_labels[remove_idx]
                    different_cluster = [idx for idx in inactive
                                       if self.cluster_labels[idx] != remove_cluster]
                    if different_cluster:
                        add_idx = np.random.choice(different_cluster)
                        neighbor[add_idx] = 1
                    else:
                        add_idx = np.random.choice(inactive)
                        neighbor[add_idx] = 1

        return neighbor

    def n3_size_variation(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 3: Variar tamanho da solução"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]
        current_size = len(active)

        # Variar tamanho entre 3 e 12
        if current_size < 5 and len(inactive) > 0:
            # Adicionar 2-3 elementos
            n_add = min(3, len(inactive))
            add_indices = np.random.choice(inactive, n_add, replace=False)
            neighbor[add_indices] = 1
        elif current_size > 8:
            # Remover 1-2 elementos
            n_remove = min(2, current_size - 3)
            candidates = [idx for idx in active if idx != self.main_package_idx]
            if len(candidates) >= n_remove:
                remove_indices = np.random.choice(candidates, n_remove, replace=False)
                neighbor[remove_indices] = 0

        return neighbor

    def generate_diverse_solution(self) -> np.ndarray:
        """Gerar solução diversa para preencher arquivo"""
        solution = np.zeros(len(self.package_names), dtype=int)

        # Tamanho variado para diversidade
        n_select = np.random.randint(3, 10)

        # Estratégias variadas
        strategy = np.random.choice(['random', 'cooccur', 'semantic', 'cluster'])

        if strategy == 'cooccur' and hasattr(self, 'cooccur_candidates'):
            candidates = self.cooccur_candidates[:100]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1
        elif strategy == 'semantic' and hasattr(self, 'semantic_candidates'):
            candidates = self.semantic_candidates[:100]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1
        elif strategy == 'cluster' and hasattr(self, 'cluster_labels'):
            cluster_id = np.random.choice(np.unique(self.cluster_labels))
            cluster_members = np.where(self.cluster_labels == cluster_id)[0]
            if len(cluster_members) >= n_select:
                indices = np.random.choice(cluster_members, n_select, replace=False)
                solution[indices] = 1
            else:
                indices = np.random.choice(len(solution), n_select, replace=False)
                solution[indices] = 1
        else:
            # Random
            indices = np.random.choice(len(solution), n_select, replace=False)
            solution[indices] = 1

        solution[self.main_package_idx] = 1
        return solution

    def run(self) -> List[Dict]:
        """Run MOVNS Final com foco em manter 80-100 soluções"""
        print(f"\nStarting MOVNS Final for {self.main_package}...")
        print("="*60)

        # Garantir arquivo grande desde o início
        print("Building diverse initial archive...")
        while len(self.archive) < self.min_archive:
            solution = self.generate_diverse_solution()
            objectives = self.evaluate_objectives(solution)
            self.update_archive(solution, objectives)

        print(f"Initial archive: {len(self.archive)} solutions")

        best_hv = 0
        no_improvement = 0

        for iteration in range(self.max_iterations):
            # Garantir tamanho mínimo do arquivo
            while len(self.archive) < self.min_archive:
                new_sol = self.generate_diverse_solution()
                new_obj = self.evaluate_objectives(new_sol)
                self.update_archive(new_sol, new_obj)

            # Selecionar solução do arquivo
            if len(self.archive) > 0:
                parent_idx = np.random.randint(len(self.archive))
                parent = self.archive[parent_idx]['chromosome'].copy()
            else:
                parent = self.generate_diverse_solution()

            # VNS com neighborhoods de diversidade
            k = 0
            k_max = 4

            while k < k_max:
                # Escolher neighborhood
                if k == 0:
                    neighbor = self.n1_diversity_flip(parent)
                elif k == 1:
                    neighbor = self.n2_cluster_exchange(parent)
                elif k == 2:
                    neighbor = self.n3_size_variation(parent)
                else:
                    # N4: Random multi-flip
                    neighbor = parent.copy()
                    n_flips = np.random.randint(1, 4)
                    for _ in range(n_flips):
                        idx = np.random.randint(len(neighbor))
                        if idx != self.main_package_idx:
                            neighbor[idx] = 1 - neighbor[idx]

                # Avaliar
                if not np.array_equal(neighbor, parent):
                    neighbor_obj = self.evaluate_objectives(neighbor)
                    parent_obj = self.evaluate_objectives(parent)

                    # Sempre adicionar ao arquivo para diversidade
                    self.update_archive(neighbor, neighbor_obj)

                    # Aceitar se melhor ou não-dominado
                    if self.dominates(neighbor_obj, parent_obj):
                        parent = neighbor
                        k = 0
                    elif not self.dominates(parent_obj, neighbor_obj):
                        # Aceitar com probabilidade para diversidade
                        if np.random.random() < 0.3:
                            parent = neighbor
                            k = 0
                        else:
                            k += 1
                    else:
                        k += 1
                else:
                    k += 1

            # Adicionar soluções diversas periodicamente
            if iteration % 3 == 0:
                for _ in range(2):
                    div_sol = self.generate_diverse_solution()
                    div_obj = self.evaluate_objectives(div_sol)
                    self.update_archive(div_sol, div_obj)

            # Calcular métricas
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics and 'hypervolume' in metrics:
                    current_hv = metrics['hypervolume']

                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    if iteration % 10 == 0:
                        print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                              f"HV={current_hv:.4f}, Best={best_hv:.4f}")

            # Early stopping
            if no_improvement >= 15:
                print(f"Converged at iteration {iteration}")
                break

        # Garantir tamanho final
        while len(self.archive) < self.min_archive:
            solution = self.generate_diverse_solution()
            objectives = self.evaluate_objectives(solution)
            self.update_archive(solution, objectives)

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_final.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_Final(
        package,
        archive_size=100,
        max_iterations=30,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    print(f"Target achieved: {80 <= len(solutions) <= 100}")


if __name__ == "__main__":
    main()