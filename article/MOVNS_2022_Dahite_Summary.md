# MOVNS Algorithm Summary from Dahite et al. (2022)

**Paper**: Multi-Objective Model and Variable Neighborhood Search Algorithms for the Joint Maintenance Scheduling and Workforce Routing Problem
**Authors**: Lamiaa Dahite, Abdeslam Kadrani, Rachid Benmansour, Rym Nesrine Guibadj, Cyril Fonlupt
**Published**: Mathematics 2022, Volume 10, Issue 11

## Key Algorithms Presented

### 1. MOVND/P (Multi-objective VND based on Pareto Dominance)

**Algorithm 3 - Core Structure:**
- Designed as an intensified local search component for GVNS
- Uses novel MOBI/P (Multi-objective Best Improvement) strategy
- Tests if each neighbor is non-dominated with best solution found so far in neighborhood

**Key Features:**
- Starts with unique initial solution
- Archive A stores non-dominated solutions
- Uses counterAI to track archive improvements
- Neighborhood change based on improvement detection

### 2. MOVND/PI (Improved version)

**Algorithm 4 - Enhanced Structure:**
- More sophisticated solution selection criterion
- Randomly chooses from non-dominated solutions not previously explored
- Focuses on diversity by avoiding previously visited points
- Designed for standalone use, not just as component

### 3. MOGVNS/P (Multi-objective GVNS based on Pareto)

**Algorithm 5 - Main GVNS Algorithm:**
```
1. Initialize solution s₀ using bestInsertionHeuristic(0.5, 0.5)
2. Add s₀ to archive A
3. While archive improves and iteration < itermax:
   - For each neighborhood k:
     a. Apply shaking nS times to get s'
     b. Apply MOVND/P to s' to get PVND
     c. Update archive A with non-dominated solutions from PVND
     d. Check neighborhood change criteria
   - Select new s from unexplored solutions in A
```

**Key Components:**
- **Shaking Phase**: Diversification through random perturbations
- **Local Search**: Uses MOVND/P for intensification
- **Archive Management**: Maintains Pareto-optimal solutions
- **Restart Strategy**: Selects from unexplored archive solutions

### 4. MOGVNS/CDP (with Population)

**Algorithm 6 - Population-based variant:**
- Starts with population of M+1 solutions
- Each generated using different weights (w₁, wₗ)
- Applies MOGVNS/P to each population member
- Aims for better coverage through diverse starting points

### 5. MOGVNS/D (Decomposition-based)

**Algorithm 7 - Weighted Sum Approach:**
```
1. Generate initial population with weights
2. For each solution i:
   - Apply shaking
   - Apply single-objective VND
   - Update if improved according to g(s, w₁, wₗ)
3. Convert final population to Pareto set
```

## Novel Contributions

### MOBI/P Strategy (Multi-objective Best Improvement)
**Key Innovation**: Tests each neighbor against the dynamically updated best solution found so far in the current neighborhood exploration.

**Procedure**:
1. Initialize bestSolution as first neighbor
2. For each neighbor n:
   - If n dominates bestSolution: update bestSolution
   - If n is incomparable: add to candidate set
3. Return all non-dominated solutions found

### Neighborhood Structures Used
1. **Swap**: Exchange two operations (inter/intra-route)
2. **Insert**: Move operation to new position
3. **2-opt***: Inter-route edge exchange
4. **2-opt**: Intra-route edge reversal

### Performance Metrics
- **Hypervolume (HV)**: Measures convergence and coverage
- **Number of Non-dominated Points (NDP)**: Solution diversity
- **CPU Time**: Computational efficiency

## Problem-Specific Adaptations

### Three Objectives Considered:
1. **LU (Linked Usage)**: Maximize co-occurrence in projects
2. **SS (Semantic Similarity)**: Maximize topical coherence
3. **RSS (Set Size)**: Minimize recommendation set size

### PyCommend Integration:
- Smart initialization using domain knowledge
- Candidate pools: co-occurrence, semantic, cluster-based
- Size-aware mutation operators
- Semantic coherence preservation

## Results Summary

### Performance Improvements (vs Literature):
- **MOVND/P**:
  - HV: +85.71% (failure), +7.82% (maintenance)
  - CPU: -80.79% (failure), -81.56% (maintenance)

- **MOVND/PI**:
  - HV: +303.78% (failure), +104.12% (maintenance)
  - NDP: +304.90% (failure), +108.45% (maintenance)

- **MOGVNS/P**:
  - HV: +91.47% (failure), +52.29% (maintenance)
  - CPU: -47.74% (failure), -41.69% (maintenance)

### Algorithm Comparison:
- MOGVNS/P outperforms MOGVNS/D by 574-792% in HV
- MOVND/PI comparable to MOGVNS/P but 65-70% faster
- Population-based variants (CDP) show marginal improvements

## Implementation Details

### Parameters:
- itermax = max(1, ⌊n/2⌋) for MOVND/P and MOGVNS/P
- iterCmax = max(1, ⌊n/16⌋) for neighborhood change
- Population size M = 10 for decomposition approaches
- Shaking repetitions nS = 3

### Stopping Criteria:
1. Maximum iterations reached
2. No archive improvement detected
3. Neighborhood exploration limits

## Key Takeaways

1. **MOBI/P strategy** significantly improves multi-objective local search efficiency
2. **Pareto dominance** approach superior to weighted sum for discrete problems
3. **Archive management** crucial for maintaining solution diversity
4. **Smart initialization** with domain knowledge improves convergence
5. **MOVND/PI** offers best speed-quality tradeoff for PyCommend

## Citation
```bibtex
@article{dahite2022multi,
  title={Multi-Objective Model and Variable Neighborhood Search Algorithms for
         the Joint Maintenance Scheduling and Workforce Routing Problem},
  author={Dahite, Lamiaa and Kadrani, Abdeslam and Benmansour, Rachid and
          Guibadj, Rym Nesrine and Fonlupt, Cyril},
  journal={Mathematics},
  volume={10},
  number={11},
  pages={1807},
  year={2022},
  publisher={MDPI}
}
```