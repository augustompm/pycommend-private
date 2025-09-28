# PROJECT V5 - NSGA-II → MOVNS Transformation for PyCommend-VNS

## 🎯 Core Concept
**Transform NSGA-II into MOVNS** by replacing genetic operators with VNS neighborhoods while **keeping 80% of the existing code**.

## 📚 Scientific Foundation

### Papers Analyzed (Downloaded)
1. **Dahite et al. (2022)** - MOVND/PI with MOBI/P strategy
2. **Pardo et al. (2024)** - MOGVNS for software
3. **Hassani et al. (2023)** - PVNS hybrid approach

### Key Decision: MOVND/PI Selected
- **MOBI/P strategy** ideal for discrete problems
- **65-70% faster** than genetic algorithms
- **Reuses 80% of NSGA-II code**
- Expected **+30-45% improvement** in hypervolume

## 🔄 Transformation Plan: NSGA-II → MOVNS

### What We Keep from NSGA-II (80%)
```python
✅ load_all_data()              # All data loading
✅ initialize_semantic_components()  # K-means clustering
✅ compute_candidate_pools()     # Candidate pools
✅ evaluate_objectives()         # LU, SS, RSS objectives (unchanged!)
✅ smart_initialization()        # All initialization strategies
✅ track_metrics                 # Quality metrics tracking
```

### What We Replace (20%)
```python
❌ fast_non_dominated_sort()    → ✅ Pareto archive management
❌ crowding_distance_assignment() → ✅ Archive diversity
❌ tournament_selection()        → ✅ Archive-based selection
❌ crossover()                   → ✅ VNS shaking
❌ mutation()                    → ✅ Neighborhood operators
❌ genetic loop                  → ✅ VNS loop with MOBI/P
```

## 📋 Implementation: movns_vns.py

### Step 1: Create MOVNS by Inheriting NSGA-II
```python
from nsga2_vns import NSGA2_VNS

class MOVNS_VNS(NSGA2_VNS):
    """
    MOVNS for PyCommend - Transforms NSGA-II into VNS
    Reuses 80% of NSGA-II infrastructure
    """

    def __init__(self, main_package, archive_size=100, max_gen=30):
        # Inherit ALL data loading and evaluation from NSGA-II
        super().__init__(main_package, pop_size=archive_size, max_gen=max_gen)

        # Replace population with archive
        self.archive = []
        self.archive_limit = archive_size

        # VNS specific
        self.k_max = 4  # Number of neighborhoods
```

### Step 2: Define Neighborhoods (Adapt from Mutation/Crossover)
```python
def _define_neighborhoods(self):
    """Transform NSGA-II operators into VNS neighborhoods"""

    def n1_single_bit_flip(solution):
        # From mutation: flip 1 random bit
        s_new = solution.copy()
        idx = np.random.randint(self.n_packages)
        s_new[idx] = 1 - s_new[idx]
        return s_new

    def n2_multi_bit_flip(solution):
        # From mutation: flip 2-3 bits
        s_new = solution.copy()
        n_flips = np.random.randint(2, 4)
        indices = np.random.choice(self.n_packages, n_flips, replace=False)
        s_new[indices] = 1 - s_new[indices]
        return s_new

    def n3_segment_exchange(solution):
        # From crossover: exchange segment
        s_new = solution.copy()
        size = np.sum(solution)
        if size > 2:
            # Remove some, add others
            active = np.where(solution == 1)[0]
            to_remove = np.random.choice(active, size//3)
            s_new[to_remove] = 0

            # Add from candidates
            candidates = self.cooccur_candidates[:50]
            to_add = np.random.choice(candidates, size//3)
            s_new[to_add] = 1
        return s_new

    def n4_smart_adjustment(solution):
        # Domain-specific: adjust to ideal size using co-occurrence
        s_new = solution.copy()
        current_size = np.sum(solution)

        if current_size < self.ideal_size:
            # Add top co-occurring packages
            candidates = self.cooccur_candidates
            for c in candidates:
                if s_new[c] == 0:
                    s_new[c] = 1
                    if np.sum(s_new) >= self.ideal_size:
                        break
        elif current_size > self.ideal_size * 1.5:
            # Remove weakest packages
            active = np.where(solution == 1)[0]
            scores = [self.rel_matrix[self.main_package_idx, i] for i in active]
            weakest = active[np.argsort(scores)[:current_size - self.ideal_size]]
            s_new[weakest] = 0

        return s_new

    return [n1_single_bit_flip, n2_multi_bit_flip,
            n3_segment_exchange, n4_smart_adjustment]
```

### Step 3: MOBI/P Local Search (Core Innovation)
```python
def mobi_p_local_search(self, solution, neighborhood):
    """
    Multi-Objective Best Improvement with Pareto
    From Dahite et al. (2022)
    """
    best_solution = solution
    best_objectives = self.evaluate_objectives(solution)  # Reuse from NSGA-II!
    pareto_set = []

    # Generate all neighbors in this neighborhood
    for _ in range(20):  # Sample size
        neighbor = neighborhood(solution)
        neighbor_obj = self.evaluate_objectives(neighbor)

        # Check dominance
        if self.dominates(neighbor_obj, best_objectives):
            best_solution = neighbor
            best_objectives = neighbor_obj
            pareto_set = [(neighbor, neighbor_obj)]
        elif not self.dominates(best_objectives, neighbor_obj):
            # Non-dominated, add to Pareto set
            pareto_set.append((neighbor, neighbor_obj))

    # Return non-dominated solutions
    return self.filter_non_dominated(pareto_set)
```

### Step 4: Main MOVNS Loop (Replace Genetic Loop)
```python
def run(self):
    """
    MOVNS main algorithm - replaces NSGA-II genetic loop
    """
    print(f"Starting MOVNS for {self.main_package}")

    # Step 1: Initialize using NSGA-II's smart initialization
    initial_pop = self.initialize_population()  # Reuse completely!

    # Step 2: Extract non-dominated as initial archive
    for ind in initial_pop:
        self.update_archive(ind['chromosome'], ind['objectives'])

    # Step 3: VNS main loop
    neighborhoods = self._define_neighborhoods()

    for iteration in range(self.max_gen):
        improved = False

        # For each solution in archive
        for sol_dict in self.archive[:]:  # Copy to allow modification
            solution = sol_dict['chromosome']
            k = 0

            # Variable Neighborhood Search
            while k < self.k_max:
                # Shaking
                s_prime = self.shake(solution, neighborhoods[k], intensity=k+1)

                # MOBI/P Local Search
                improved_solutions = self.mobi_p_local_search(s_prime, neighborhoods[k])

                # Update archive
                archive_updated = False
                for (new_sol, new_obj) in improved_solutions:
                    if self.update_archive(new_sol, new_obj):
                        archive_updated = True
                        improved = True

                # Neighborhood change
                if archive_updated:
                    k = 0  # Reset to first neighborhood
                else:
                    k += 1  # Next neighborhood

        # Archive management
        if len(self.archive) > self.archive_limit:
            self.truncate_archive()

        # Track metrics (reuse from NSGA-II)
        if self.track_metrics:
            self.update_metrics_history()

        print(f"Generation {iteration}: Archive size = {len(self.archive)}")

        # Early stopping if no improvement
        if not improved and iteration > 10:
            print("No improvement, stopping early")
            break

    return self.get_final_solutions()
```

### Step 5: Helper Methods
```python
def shake(self, solution, neighborhood, intensity):
    """Perturbation with adaptive intensity"""
    s = solution.copy()
    for _ in range(intensity):
        s = neighborhood(s)
    return s

def update_archive(self, solution, objectives):
    """Update Pareto archive"""
    # Remove dominated solutions
    self.archive = [s for s in self.archive
                   if not self.dominates(objectives, s['objectives'])]

    # Check if new solution is dominated
    for s in self.archive:
        if self.dominates(s['objectives'], objectives):
            return False

    # Add to archive
    self.archive.append({
        'chromosome': solution,
        'objectives': objectives
    })
    return True

def truncate_archive(self):
    """Keep archive size limited using diversity"""
    # Reuse crowding distance concept from NSGA-II
    self.crowding_distance_assignment(self.archive)
    self.archive.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)
    self.archive = self.archive[:self.archive_limit]
```

## 📊 Expected Results

### Performance Comparison

| Metric | NSGA-II (Current) | MOVNS (Expected) | Improvement |
|--------|------------------|------------------|-------------|
| **Hypervolume** | 0.1932 | 0.25-0.28 | +30-45% |
| **Time (s)** | 12.22 | 7-9 | -40% |
| **Convergence** | 30 generations | 15-20 iterations | -50% |
| **Solutions** | 50 | 30-40 | More focused |

### Why MOVNS Will Outperform NSGA-II

1. **Local Search Focus**: MOBI/P explores neighborhoods thoroughly
2. **Smart Neighborhoods**: Based on problem structure
3. **No Random Crossover**: Directed search instead
4. **Archive vs Population**: Focus on quality not quantity
5. **Early Convergence**: VNS converges faster than GA

## 🎓 Paper Structure

### Title
**"From Genetic to Neighborhood Search: Transforming NSGA-II into MOVNS for Python Package Recommendation"**

### Key Messages
1. **Transformation methodology**: How to convert GA → VNS
2. **80% code reuse**: Practical implementation
3. **MOBI/P effectiveness**: Local search for discrete MOO
4. **Real application**: PyCommend with 9,997 packages

### Experiments
1. **Baseline**: Original NSGA-II (HV = 0.1932)
2. **Transformed**: MOVNS (target HV = 0.25+)
3. **Comparison**: MOEA/D (HV = 0.1041)
4. **Ablation**: Test each neighborhood contribution

## ✅ Implementation Checklist

### Week 1: Core MOVNS
- [ ] Create movns_vns.py inheriting from NSGA2_VNS
- [ ] Implement 4 neighborhoods
- [ ] Implement MOBI/P local search
- [ ] Replace genetic loop with VNS loop

### Week 2: Testing
- [ ] Test with numpy, fastapi, django
- [ ] Compare hypervolume with NSGA-II
- [ ] Measure convergence speed
- [ ] Validate solution quality

### Week 3: Paper
- [ ] Write transformation methodology
- [ ] Document results
- [ ] Create figures
- [ ] Submit to ICVNS 2025

## 🚀 Quick Implementation

```bash
# 1. Copy NSGA-II as base
cd /e/pycommend/pycommend-code/src/optimizer
cp nsga2_vns.py movns_vns.py

# 2. Edit to transform into MOVNS
# - Keep all data loading
# - Keep evaluate_objectives
# - Replace genetic operators with neighborhoods
# - Replace main loop with VNS

# 3. Test
python -m src.optimizer.movns_vns --package numpy

# 4. Compare
python compare_algorithms_real.py --include-movns
```

## 🎯 Success Criteria

1. **MOVNS working**: Produces valid Pareto front
2. **Better than NSGA-II**: HV ≥ 0.25 (vs 0.1932)
3. **Faster convergence**: < 20 iterations
4. **Code reuse**: 80% from NSGA-II unchanged
5. **Paper ready**: Clear transformation story

---
*PROJECT V5 CORRECT - NSGA-II → MOVNS Transformation*
*Created: 2024-12-27*
*Target: ICVNS 2025*