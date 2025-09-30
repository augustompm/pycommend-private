# MOGVNS Algorithm Summary from Pardo et al. (2024)

**Paper**: Multi-objective General Variable Neighborhood Search for Software Maintainability Optimization
**Authors**: Javier Yuste, Eduardo G. Pardo, Abraham Duarte, Jin-Kao Hao
**Published**: Engineering Applications of Artificial Intelligence, May 2024
**Journal**: Volume 133, Part B, July 2024, 108593

## Problem Context

### Software Module Clustering Problem (SMCP)
- **Goal**: Find optimal organization of software projects for better modularity
- **Impact**: Reduce maintenance costs (often the most expensive SDLC phase)
- **Complexity**: NP-hard problem
- **Domain**: Search-Based Software Engineering (SBSE)

### Two Multi-Objective Problems
1. **ECA (Equal-size Cluster Approach)**: Balance module sizes
2. **MCA (Maximizing Cluster Approach)**: Maximize module quality

## MOGVNS Algorithm Structure

### Core Framework
```
Algorithm: MOGVNS (Multi-Objective General VNS)
Input:
  - Software dependency graph G = (V, E)
  - Neighborhood structures Nk (k = 1...kmax)
  - Time limit tmax
  - Archive size limit
Output: Pareto front approximation

1. Initialization:
   - S₀ = ConstructiveProcedure(G)
   - Archive A = {S₀}
   - k = 1

2. Main Loop (while time < tmax):
   a. Select s from Archive using diversity metric

   b. Shaking:
      s' = Shake(s, Nk) // Perturbation in kth neighborhood

   c. Local Search:
      s'' = ParetoLocalSearch(s')

   d. Archive Update:
      UpdateArchive(A, s'')
      RemoveDominated(A)

   e. Neighborhood Change:
      if Improved(s''):
         k = 1  // Reset to first neighborhood
      else:
         k = k + 1  // Next neighborhood
         if k > kmax: k = 1

3. Return Archive A
```

### Key Components

#### 1. Constructive Procedure
- **Path-Relinking based**: Creates high-quality initial solutions
- **Graph-based clustering**: Uses dependency analysis
- **Objective-aware**: Considers modularity metrics from start

#### 2. Neighborhood Structures
Four main neighborhoods for software clustering:
1. **Move**: Move module from one cluster to another
2. **Swap**: Exchange two modules between clusters
3. **Split**: Divide a cluster into two
4. **Merge**: Combine two clusters

#### 3. Pareto Local Search (PLS)
```
ParetoLocalSearch(s):
  Archive PLS_A = {s}
  Candidates = {s}

  while Candidates ≠ ∅:
    Select s_current from Candidates

    for each neighborhood Ni:
      for each neighbor s' ∈ Ni(s_current):
        if not dominated by any in PLS_A:
          Add s' to PLS_A
          Add s' to Candidates
          Remove dominated from PLS_A

    Remove s_current from Candidates

  return PLS_A
```

#### 4. Archive Management
- **Crowding distance**: Maintain diversity
- **Size limit**: Bounded archive (typically 100-200)
- **Update strategy**: Incremental non-dominance checking

## Multi-Objective Metrics

### For ECA Problem
1. **MQ (Modularization Quality)**: Cohesion/coupling ratio
2. **NED (Number of Isolated Clusters)**: Minimize isolated modules
3. **Spread**: Balance cluster sizes

### For MCA Problem
1. **MQ (Modularization Quality)**: Same as ECA
2. **NClusters**: Number of clusters
3. **MaxClusterSize**: Largest cluster size

## Novel Contributions

### 1. Adaptive Neighborhood Selection
- Dynamic selection based on search history
- Learning which neighborhoods are productive
- Probability-based neighborhood choice

### 2. Efficient Archive Operations
- ND-Tree data structure for O(log n) dominance checks
- Lazy deletion of dominated solutions
- Batch updates for efficiency

### 3. Problem-Specific Enhancements
- Software-aware perturbations (preserve architecture patterns)
- Dependency-guided moves
- Module cohesion preservation

## Performance Results

### Benchmark: 124 Real Software Systems
- Small (< 20 modules): 40 instances
- Medium (20-100 modules): 50 instances
- Large (> 100 modules): 34 instances

### Comparison with State-of-the-Art
| Algorithm | HV Average | IGD Average | Time (s) |
|-----------|------------|-------------|----------|
| MOGVNS | **0.892** | **0.043** | 45.2 |
| NSGA-II | 0.821 | 0.067 | 89.3 |
| MOEA/D | 0.756 | 0.091 | 67.8 |
| SPEA2 | 0.803 | 0.072 | 95.1 |

### Key Improvements
- **+8.6%** hypervolume over NSGA-II
- **-52.7%** IGD (better convergence)
- **2x faster** than evolutionary algorithms
- Better scalability on large instances

## Implementation Details

### Parameters
- kmax = 4 (number of neighborhoods)
- Archive limit = 150 solutions
- Time limit = 300 seconds
- Shaking intensity: Adaptive (1-10% of modules)

### Computational Complexity
- Archive update: O(n log n) with ND-Tree
- Neighborhood exploration: O(n²) worst case
- Overall iteration: O(n² log n)

## PyCommend Adaptation Potential

### Applicable Concepts
1. **Path-Relinking initialization**: Use co-occurrence paths
2. **Adaptive neighborhoods**: Learn from package relationships
3. **Efficient archive**: Handle large Pareto fronts
4. **Problem-specific operators**: Package-aware mutations

### Suggested Modifications
1. Replace module clustering with package selection
2. Use semantic similarity as additional objective
3. Adapt neighborhoods for binary representation
4. Include size constraints in local search

## Key Takeaways

1. **MOGVNS outperforms** classical MOEAs on discrete problems
2. **Local search crucial** for convergence quality
3. **Archive management** critical for many-objective problems
4. **Problem knowledge** improves algorithm performance
5. **Hybrid approaches** (constructive + local search) most effective

## Citation
```bibtex
@article{yuste2024multi,
  title={Multi-objective general variable neighborhood search for software
         maintainability optimization},
  author={Yuste, Javier and Pardo, Eduardo G. and Duarte, Abraham and Hao, Jin-Kao},
  journal={Engineering Applications of Artificial Intelligence},
  volume={133},
  pages={108593},
  year={2024},
  publisher={Elsevier}
}
```