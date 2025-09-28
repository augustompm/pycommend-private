# PVNS Algorithm Summary from 3PHEA Paper (2023)

**Paper**: Novel Hybrid Evolutionary Algorithm for Bi-objective Optimization Problems
**Authors**: Mohammad Reza Hassani, Seyed Taghi Akhavan Niaki, Mahdi Bashiri
**Published**: Scientific Reports, Nature, March 2023
**DOI**: 10.1038/s41598-023-31123-8

## Context: Three-Phase Hybrid Evolutionary Algorithm (3PHEA)

### Overall Architecture
3PHEA combines three powerful algorithms:
1. **Phase 1**: Lin-Kernighan Heuristic (LKH) - Find supported efficient solutions
2. **Phase 2**: Improved NSGA-II - Find non-supported efficient solutions
3. **Phase 3**: **Pareto Variable Neighborhood Search (PVNS)** - Improve population quality

## PVNS Algorithm (Phase 3)

### Core Concept
PVNS is a multi-objective adaptation of Variable Neighborhood Search that works with Pareto-optimal solutions to escape local optima through systematic neighborhood changes.

### Algorithm 6: Pareto Variable Neighborhood Search
```
Algorithm: PVNS (Pareto Variable Neighborhood Search)
Input:
  - Initial Pareto Set P from Phase 2
  - Neighborhood structures N₁, N₂, ..., Nₖₘₐₓ
  - Maximum iterations iterMax
  - Local search procedure LocalSearch()
Output: Improved Pareto Set P*

1. Initialization:
   P* = P  // Copy initial Pareto set
   iter = 0

2. Main Loop:
   while iter < iterMax:
      a. Select solution s from P* using tournament selection

      b. k = 1  // Start with first neighborhood

      c. while k ≤ kₘₐₓ:
         i. Shaking:
            s' = Shake(s, Nₖ)  // Random neighbor in kth neighborhood

         ii. Local Search:
            s'' = LocalSearch(s')  // Apply local search

         iii. Pareto Update:
            if s'' is non-dominated by P*:
               Add s'' to P*
               Remove dominated solutions from P*
               k = 1  // Restart from first neighborhood
            else:
               k = k + 1  // Move to next neighborhood

      d. Archive Management:
         if |P*| > MaxArchiveSize:
            TruncateArchive(P*)  // Remove crowded solutions

      e. iter = iter + 1

3. Return P*
```

### Key Components

#### 1. Neighborhood Structures for PVNS
The paper uses five neighborhood structures:
1. **2-opt**: Reverse a segment of the tour
2. **3-opt**: Remove 3 edges and reconnect
3. **Or-opt**: Move a chain of 1-3 consecutive nodes
4. **Exchange**: Swap two non-adjacent nodes
5. **Insertion**: Remove and reinsert a node

#### 2. Shaking Procedure
```
Shake(s, Nₖ):
  intensity = random(1, min(k*2, 10))  // Adaptive intensity
  for i = 1 to intensity:
    s = RandomMove(s, Nₖ)  // Apply random move in neighborhood k
  return s
```

#### 3. Local Search Component
```
LocalSearch(s):
  improved = true
  while improved:
    improved = false
    for each neighborhood N:
      s' = BestImprovement(s, N)  // Find best neighbor
      if Dominates(s', s) or IncomparableButBetter(s', s):
        s = s'
        improved = true
        break
  return s
```

#### 4. Pareto Dominance Check
```
Dominates(s1, s2):
  return all(s1.objectives ≤ s2.objectives) and
         any(s1.objectives < s2.objectives)
```

## Integration with 3PHEA

### Phase 3 Implementation Details
```
Phase3_PVNS(population):
  // Convert population to Pareto set
  ParetoSet = ExtractNonDominated(population)

  // Apply PVNS
  ImprovedSet = PVNS(ParetoSet)

  // Merge with original population
  FinalPopulation = Merge(population, ImprovedSet)

  // Keep only non-dominated
  return ExtractNonDominated(FinalPopulation)
```

### Performance Enhancements

#### Dynamic Neighborhood Selection
- Start with simple neighborhoods (2-opt)
- Progress to complex ones (3-opt, Or-opt)
- Adapt based on improvement frequency

#### Elite Edge Preservation
- Identify common edges in Pareto set
- Protect during perturbation
- Focus exploration on variable edges

#### Adaptive Parameters
```
UpdateParameters(iteration, maxIter):
  shakingIntensity = 1 + (iteration / maxIter) * 5
  neighborhoodProb = AdjustBasedOnSuccess()
  archiveSize = 100 + (iteration / maxIter) * 50
```

## Experimental Results

### Test Problems
- **Bi-objective TSP**: 20 instances
- **Sizes**: 100 to 1000 cities
- **Objectives**: Distance and time minimization

### Performance Metrics
| Metric | PVNS Alone | 3PHEA with PVNS | NSGA-II | MOEA/D |
|--------|------------|-----------------|---------|---------|
| HV | 0.876 | **0.943** | 0.892 | 0.834 |
| IGD | 0.0124 | **0.0087** | 0.0156 | 0.0201 |
| Spacing | 0.0234 | **0.0198** | 0.0267 | 0.0312 |
| CPU Time | 87s | 145s | 198s | 167s |

### Key Findings
1. **PVNS improves HV by 7.8%** over initial Pareto set
2. **Reduces IGD by 29.8%** indicating better convergence
3. **More uniform distribution** (better spacing metric)
4. **Computationally efficient** compared to pure evolutionary methods

## PyCommend Adaptation

### Applicable Concepts
1. **Multi-phase approach**: Combine with existing NSGA-II/MOEA/D
2. **Dynamic neighborhoods**: Adapt for package selection problem
3. **Elite preservation**: Keep successful package combinations
4. **Local search intensification**: Focus on promising regions

### Proposed Neighborhoods for PyCommend
1. **Add Package**: Add one related package
2. **Remove Package**: Remove one package
3. **Swap Package**: Replace with similar package
4. **Batch Add**: Add multiple co-occurring packages
5. **Size Reduction**: Remove lowest-value packages

### Implementation Suggestions
```python
def pvns_for_pycommend(pareto_set, max_iter=100):
    neighborhoods = [
        AddPackageNeighborhood(),
        RemovePackageNeighborhood(),
        SwapPackageNeighborhood(),
        BatchAddNeighborhood(),
        SizeReductionNeighborhood()
    ]

    improved_set = pareto_set.copy()

    for _ in range(max_iter):
        solution = tournament_select(improved_set)
        k = 0

        while k < len(neighborhoods):
            # Shaking
            perturbed = shake(solution, neighborhoods[k])

            # Local search
            improved = local_search(perturbed)

            # Update Pareto set
            if is_non_dominated(improved, improved_set):
                add_to_pareto(improved_set, improved)
                remove_dominated(improved_set)
                k = 0  # Restart
            else:
                k += 1

    return improved_set
```

## Key Takeaways

1. **PVNS excels at refining** existing Pareto approximations
2. **Systematic neighborhood exploration** prevents premature convergence
3. **Local search crucial** for exploitation
4. **Hybrid approaches superior** to single algorithms
5. **Adaptive mechanisms** improve robustness

## Citation
```bibtex
@article{hassani2023novel,
  title={Novel hybrid evolutionary algorithm for bi-objective optimization problems},
  author={Hassani, Mohammad Reza and Niaki, Seyed Taghi Akhavan and Bashiri, Mahdi},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={4764},
  year={2023},
  publisher={Nature Publishing Group}
}
```