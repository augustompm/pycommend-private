"""
MOVNS Robust - Versão robusta para vencer HV + Spacing
Mantém 80-100 soluções de alta qualidade
"""

import numpy as np
import random
import sys
import os
from typing import List, Dict, Tuple
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Imports do parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from optimizer.movns_v2 import MOVNS_V2


class MOVNS_Robust(MOVNS_V2):
    """
    MOVNS Robust - Baseado no Advanced mas mantendo 80-100 soluções
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):

        super().__init__(main_package, archive_size, max_iterations,
                        track_metrics=track_metrics)

        # Parâmetros para manter muitas soluções
        self.min_archive = 80  # Nunca ter menos que 80 soluções
        self.target_archive = 100  # Alvo de 100 soluções

        # Parâmetros de busca local agressiva
        self.local_search_intensity = 2  # Reduzido para ser mais rápido
        self.perturbation_strength = 0.2  # Força da perturbação

        # Para diversidade
        self.diversity_memory = set()
        self.generation_strategies = ['cooccur', 'semantic', 'cluster', 'random', 'hybrid']

        print(f"MOVNS Robust initialized")
        print(f"Archive target: {self.min_archive}-{self.target_archive} solutions")
        print(f"Max iterations: {max_iterations}")

    def aggressive_initialization(self):
        """Inicialização agressiva para ter 80+ soluções desde o início"""
        print("Aggressive initialization...")

        # Coletar soluções primeiro, depois adicionar em batch
        initial_solutions = []

        # Estratégia 1: Co-ocorrência (10 soluções)
        if hasattr(self, 'cooccur_candidates'):
            for i in range(10):
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 8)
                candidates = self.cooccur_candidates[:100]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    initial_solutions.append(solution)

        # Estratégia 2: Similaridade semântica (10 soluções)
        if hasattr(self, 'semantic_candidates'):
            for i in range(10):
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 8)
                candidates = self.semantic_candidates[:100]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    obj = self.evaluate_objectives(solution)
                    self.update_archive(solution, obj)

        # Estratégia 3: Baseada em clusters (10 soluções)
        if hasattr(self, 'cluster_labels'):
            unique_clusters = np.unique(self.cluster_labels)
            for cluster_id in unique_clusters[:10]:
                solution = np.zeros(len(self.package_names), dtype=int)
                cluster_members = np.where(self.cluster_labels == cluster_id)[0]
                if len(cluster_members) > 0:
                    n_select = min(5, len(cluster_members))
                    indices = np.random.choice(cluster_members, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    obj = self.evaluate_objectives(solution)
                    self.update_archive(solution, obj)

        # Estratégia 4: Aleatória com tamanhos variados (10 soluções)
        for size in range(2, 7):
            for _ in range(2):
                solution = np.zeros(len(self.package_names), dtype=int)
                indices = np.random.choice(len(solution), size, replace=False)
                solution[indices] = 1
                solution[self.main_package_idx] = 1
                obj = self.evaluate_objectives(solution)
                self.update_archive(solution, obj)

        # Estratégia 5: Híbrida (preencher até 80)
        while len(self.archive) < self.min_archive:
            solution = self.generate_hybrid_solution()
            obj = self.evaluate_objectives(solution)
            self.update_archive(solution, obj)

        print(f"Initial archive: {len(self.archive)} solutions")

    def generate_hybrid_solution(self) -> np.ndarray:
        """Gera solução híbrida combinando estratégias"""
        solution = np.zeros(len(self.package_names), dtype=int)

        # Escolher estratégia aleatória
        strategy = random.choice(self.generation_strategies)
        n_select = np.random.randint(3, 10)

        if strategy == 'cooccur' and hasattr(self, 'cooccur_candidates'):
            candidates = self.cooccur_candidates[:150]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1

        elif strategy == 'semantic' and hasattr(self, 'semantic_candidates'):
            candidates = self.semantic_candidates[:150]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1

        elif strategy == 'cluster' and hasattr(self, 'cluster_labels'):
            # Escolher 2-3 clusters
            n_clusters = np.random.randint(2, 4)
            clusters = np.random.choice(np.unique(self.cluster_labels), n_clusters, replace=False)
            for cluster_id in clusters:
                cluster_members = np.where(self.cluster_labels == cluster_id)[0]
                if len(cluster_members) > 0:
                    n_from_cluster = min(3, len(cluster_members))
                    indices = np.random.choice(cluster_members, n_from_cluster, replace=False)
                    solution[indices] = 1

        elif strategy == 'hybrid':
            # Combinar múltiplas estratégias
            if hasattr(self, 'cooccur_candidates') and hasattr(self, 'semantic_candidates'):
                n_cooccur = n_select // 2
                n_semantic = n_select - n_cooccur

                cooccur_idx = np.random.choice(self.cooccur_candidates[:100],
                                              min(n_cooccur, len(self.cooccur_candidates[:100])),
                                              replace=False)
                semantic_idx = np.random.choice(self.semantic_candidates[:100],
                                               min(n_semantic, len(self.semantic_candidates[:100])),
                                               replace=False)
                solution[cooccur_idx] = 1
                solution[semantic_idx] = 1
        else:
            # Random fallback
            indices = np.random.choice(len(solution), n_select, replace=False)
            solution[indices] = 1

        solution[self.main_package_idx] = 1
        return solution

    def local_search_intensive(self, solution: np.ndarray) -> np.ndarray:
        """Busca local intensiva para melhorar solução"""
        best_solution = solution.copy()
        best_obj = self.evaluate_objectives(best_solution)

        for _ in range(self.local_search_intensity):
            # Tentar diferentes tipos de mudanças
            for change_type in ['add', 'remove', 'swap', 'multi']:
                neighbor = self.apply_change(best_solution, change_type)
                neighbor_obj = self.evaluate_objectives(neighbor)

                # Aceitar se melhor
                if self.dominates(neighbor_obj, best_obj):
                    best_solution = neighbor
                    best_obj = neighbor_obj

                # Adicionar ao arquivo se não-dominado
                elif not self.dominates(best_obj, neighbor_obj):
                    self.update_archive(neighbor, neighbor_obj)

        return best_solution

    def apply_change(self, solution: np.ndarray, change_type: str) -> np.ndarray:
        """Aplica mudança específica na solução"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if change_type == 'add' and len(inactive) > 0 and len(active) < 12:
            # Adicionar elemento
            idx = np.random.choice(inactive)
            neighbor[idx] = 1

        elif change_type == 'remove' and len(active) > 3:
            # Remover elemento
            candidates = [i for i in active if i != self.main_package_idx]
            if candidates:
                idx = np.random.choice(candidates)
                neighbor[idx] = 0

        elif change_type == 'swap' and len(active) > 2 and len(inactive) > 0:
            # Trocar elementos
            remove_idx = np.random.choice([i for i in active if i != self.main_package_idx])
            add_idx = np.random.choice(inactive)
            neighbor[remove_idx] = 0
            neighbor[add_idx] = 1

        elif change_type == 'multi':
            # Mudança múltipla
            n_changes = np.random.randint(1, 4)
            for _ in range(n_changes):
                idx = np.random.randint(len(neighbor))
                if idx != self.main_package_idx:
                    neighbor[idx] = 1 - neighbor[idx]

        return neighbor

    def run(self) -> List[Dict]:
        """Execução principal do MOVNS Robust"""
        print(f"\nStarting MOVNS Robust for {self.main_package}...")
        print("="*60)

        # Inicialização agressiva
        self.aggressive_initialization()

        best_hv = 0
        no_improvement = 0

        for iteration in range(self.max_iterations):

            # Garantir mínimo de soluções
            while len(self.archive) < self.min_archive:
                new_solution = self.generate_hybrid_solution()
                new_obj = self.evaluate_objectives(new_solution)
                self.update_archive(new_solution, new_obj)

            # Selecionar solução para melhorar
            if len(self.archive) > 0:
                parent_idx = np.random.randint(len(self.archive))
                current = self.archive[parent_idx]['chromosome'].copy()
            else:
                current = self.generate_hybrid_solution()

            # Busca local intensiva
            improved = self.local_search_intensive(current)

            # Se melhorou, adicionar
            if not np.array_equal(improved, current):
                obj = self.evaluate_objectives(improved)
                self.update_archive(improved, obj)

            # A cada 3 iterações, adicionar soluções diversas
            if iteration % 3 == 0:
                for _ in range(3):
                    diverse = self.generate_hybrid_solution()
                    diverse_obj = self.evaluate_objectives(diverse)
                    self.update_archive(diverse, diverse_obj)

            # Perturbação ocasional para escapar de mínimos locais
            if iteration % 5 == 0 and len(self.archive) > 0:
                perturbed_idx = np.random.randint(len(self.archive))
                perturbed = self.archive[perturbed_idx]['chromosome'].copy()

                # Aplicar perturbação forte
                n_changes = int(len(perturbed) * self.perturbation_strength)
                for _ in range(n_changes):
                    idx = np.random.randint(len(perturbed))
                    if idx != self.main_package_idx:
                        perturbed[idx] = 1 - perturbed[idx]

                perturbed_obj = self.evaluate_objectives(perturbed)
                self.update_archive(perturbed, perturbed_obj)

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
            if no_improvement >= 20:
                print(f"Converged at iteration {iteration}")
                break

        # Garantir tamanho final
        while len(self.archive) < self.min_archive:
            solution = self.generate_hybrid_solution()
            obj = self.evaluate_objectives(solution)
            self.update_archive(solution, obj)

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_robust.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_Robust(
        package,
        archive_size=100,
        max_iterations=30,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    if 80 <= len(solutions) <= 100:
        print("Archive target achieved!")


if __name__ == "__main__":
    main()