"""
Improved VNS neighborhoods based on test results
Each neighborhood is designed for positive impact on objectives
"""

import numpy as np


class ImprovedNeighborhoods:
    """
    Neighborhoods designed for real impact on PyCommend objectives
    """

    def __init__(self, movns):
        self.movns = movns
        self.rel_matrix = movns.rel_matrix
        self.main_package_idx = movns.main_package_idx
        self.cooccur_candidates = movns.cooccur_candidates
        self.semantic_candidates = movns.semantic_candidates
        self.n_packages = movns.n_packages

    def n1_smart_flip(self, solution):
        """
        Smart single flip: Add high-value or remove low-value package
        """
        s_new = solution.copy()
        active_indices = np.where(solution == 1)[0]
        inactive_indices = np.where(solution == 0)[0]

        if len(active_indices) > 0 and len(inactive_indices) > 0:
            if np.random.random() < 0.5 and len(active_indices) > 2:
                scores = []
                for idx in active_indices:
                    if idx != self.main_package_idx:
                        score = self.rel_matrix[self.main_package_idx, idx]
                        scores.append((idx, score))

                if scores:
                    scores.sort(key=lambda x: x[1])
                    remove_idx = scores[0][0]
                    s_new[remove_idx] = 0
            else:
                if len(self.cooccur_candidates) > 0:
                    candidates = [c for c in self.cooccur_candidates[:50]
                                if solution[c] == 0]
                    if candidates:
                        add_idx = np.random.choice(candidates)
                        s_new[add_idx] = 1

        return s_new

    def n2_exchange_pair(self, solution):
        """
        Exchange: Remove one weak package, add one strong package
        """
        s_new = solution.copy()
        active_indices = np.where(solution == 1)[0]

        if len(active_indices) > 2:
            active_no_main = [i for i in active_indices if i != self.main_package_idx]
            if active_no_main:
                scores = []
                for idx in active_no_main:
                    score = self.rel_matrix[self.main_package_idx, idx]
                    scores.append((idx, score))

                scores.sort(key=lambda x: x[1])
                remove_idx = scores[0][0]
                s_new[remove_idx] = 0

                candidates = [c for c in self.cooccur_candidates[:100]
                            if solution[c] == 0]
                if candidates:
                    add_idx = np.random.choice(candidates)
                    s_new[add_idx] = 1

        return s_new

    def n3_size_adjustment(self, solution):
        """
        Adjust size toward ideal (5-7 packages)
        """
        s_new = solution.copy()
        current_size = np.sum(solution)
        ideal_size = 5

        if current_size < ideal_size:
            n_add = min(2, ideal_size - current_size)
            candidates = [c for c in self.cooccur_candidates[:100]
                        if solution[c] == 0]
            if len(candidates) >= n_add:
                add_indices = np.random.choice(candidates, n_add, replace=False)
                s_new[add_indices] = 1

        elif current_size > ideal_size + 2:
            n_remove = min(2, current_size - ideal_size)
            active_indices = np.where(solution == 1)[0]
            active_no_main = [i for i in active_indices if i != self.main_package_idx]

            if len(active_no_main) >= n_remove:
                scores = []
                for idx in active_no_main:
                    score = self.rel_matrix[self.main_package_idx, idx]
                    scores.append((idx, score))

                scores.sort(key=lambda x: x[1])
                remove_indices = [s[0] for s in scores[:n_remove]]
                s_new[remove_indices] = 0

        return s_new

    def n4_semantic_refinement(self, solution):
        """
        Replace packages with semantically similar but better co-occurrence
        """
        s_new = solution.copy()
        active_indices = np.where(solution == 1)[0]

        if len(active_indices) > 2:
            active_no_main = [i for i in active_indices if i != self.main_package_idx]

            if active_no_main and len(self.semantic_candidates) > 0:
                target_idx = np.random.choice(active_no_main)
                s_new[target_idx] = 0

                candidates = [c for c in self.semantic_candidates[:50]
                            if solution[c] == 0]

                if candidates:
                    best_candidate = None
                    best_score = -1

                    for c in candidates[:10]:
                        score = self.rel_matrix[self.main_package_idx, c]
                        if score > best_score:
                            best_score = score
                            best_candidate = c

                    if best_candidate is not None:
                        s_new[best_candidate] = 1
                    else:
                        s_new[target_idx] = 1

        return s_new