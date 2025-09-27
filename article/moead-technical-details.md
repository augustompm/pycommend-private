# MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition

## Publication Information
**Title:** MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
**Authors:** Qingfu Zhang and Hui Li
**Publication:** IEEE Transactions on Evolutionary Computation, Vol. 11, No. 6, pp. 712-731, December 2007
**DOI:** 10.1109/TEVC.2007.892759

## Algorithm Overview

MOEA/D decomposes a multi-objective optimization problem (MOP) into N scalar optimization subproblems and optimizes them simultaneously. Each subproblem is optimized using information from its neighboring subproblems, making the algorithm computationally efficient.

## Key Concepts

### 1. Problem Formulation

Minimize F(x) = (f₁(x), f₂(x), ..., fₘ(x))ᵀ
Subject to: x ∈ Ω

Where:
- x is the decision vector
- Ω is the feasible region
- m is the number of objectives

### 2. Decomposition Methods

#### 2.1 Weighted Sum Approach
g^ws(x|λ) = Σᵢ₌₁ᵐ λᵢfᵢ(x)

Where λ = (λ₁, ..., λₘ)ᵀ is a weight vector with λᵢ ≥ 0 and Σλᵢ = 1

#### 2.2 Tchebycheff Approach
g^te(x|λ,z*) = max₁≤ᵢ≤ₘ {λᵢ|fᵢ(x) - zᵢ*|}

Where:
- λ is the weight vector
- z* = (z₁*, ..., zₘ*)ᵀ is the reference point
- zᵢ* = min{fᵢ(x) | x ∈ Ω} for minimization

#### 2.3 Penalty-based Boundary Intersection (PBI)
g^pbi(x|λ,z*) = d₁ + θd₂

Where:
- d₁ = ||((F(x) - z*)ᵀλ)/||λ|||| (distance along weight vector direction)
- d₂ = ||F(x) - (z* + d₁(λ/||λ||))|| (perpendicular distance)
- θ is a penalty parameter (typically θ = 5)

## MOEA/D Algorithm Framework

### Input Parameters
- N: Population size (number of subproblems)
- λ¹, ..., λᴺ: Weight vectors
- T: Neighborhood size
- δ: Probability of selecting parents from neighborhood
- nᵣ: Maximum number of solutions replaced by offspring
- Stopping criterion

### Data Structures
- EP: External population (archive of non-dominated solutions)
- z*: Reference point (best value for each objective)
- B(i): Neighborhood of weight vector λⁱ

### Algorithm Steps

```
Algorithm: MOEA/D Framework
1. Initialization
   a) Generate N weight vectors: λ¹, ..., λᴺ
   b) Calculate Euclidean distances between weight vectors
   c) For each i = 1,...,N:
      - Find T closest weight vectors to λⁱ
      - Set B(i) = {i₁, ..., iₜ} as indices of T closest vectors
   d) Generate initial population: x¹, ..., xᴺ
   e) Initialize reference point: zᵢ* = min{fᵢ(xʲ) | j = 1,...,N}
   f) Initialize EP = ∅

2. Main Loop (while stopping criterion not met)
   For i = 1 to N:

   a) Reproduction
      - Select mating pool:
        If rand() < δ: P = B(i)  // from neighborhood
        Else: P = {1,...,N}      // from entire population
      - Randomly select indices k, l from P
      - Generate offspring y using genetic operators on xᵏ and xˡ
      - Apply mutation to y
      - Repair y if needed to ensure feasibility

   b) Update Reference Point
      For j = 1 to m:
         If fⱼ(y) < zⱼ*: zⱼ* = fⱼ(y)

   c) Update Neighboring Solutions
      Set c = 0
      For each j ∈ B(i) in random order:
         If g(y|λʲ,z*) ≤ g(xʲ|λʲ,z*):
            xʲ = y
            c = c + 1
         If c = nᵣ: break

   d) Update External Population
      Remove from EP all vectors dominated by F(y)
      If F(y) is not dominated by any vector in EP:
         Add F(y) to EP

3. Output: EP
```

## Genetic Operators

### Differential Evolution (DE/rand/2/exp)
For generating offspring from parents xᵏ and xˡ:

1. Select three random indices r₁, r₂, r₃ from mating pool
2. Generate mutant vector: v = xʳ¹ + F·(xʳ² - xʳ³)
3. Crossover with CR probability to create offspring

### Polynomial Mutation
Apply polynomial mutation with distribution index ηₘ and mutation probability pₘ

## Key Parameters and Recommendations

### Essential Parameters
- **N (Population size):** 100-300 for 2-3 objectives
- **T (Neighborhood size):** N/10 (typically 20 for N=100)
- **δ (Neighborhood selection probability):** 0.9
- **nᵣ (Max replacements):** 2
- **CR (Crossover rate):** 1.0 for DE
- **F (DE scaling factor):** 0.5
- **ηₘ (Mutation distribution index):** 20
- **pₘ (Mutation probability):** 1/n (n = number of decision variables)

### Weight Vector Generation
1. **Simplex-lattice design:** For uniform distribution
   - For 2 objectives: λᵢ = (i/H, (H-i)/H) where H determines granularity
   - For m objectives: use Das and Dennis's method

2. **Low-discrepancy sequences:** For better coverage

## Advantages

1. **Computational Efficiency:** O(mNT) per generation
2. **Scalability:** Works well for many-objective problems
3. **Flexibility:** Various decomposition methods available
4. **Parallelization:** Subproblems can be solved in parallel
5. **Local Search Integration:** Easy to incorporate problem-specific knowledge

## Implementation Considerations

1. **Normalization:** Normalize objectives when scales differ significantly
2. **Constraint Handling:** Use penalty functions or repair mechanisms
3. **Adaptive Parameters:** Consider adaptive neighborhood size and operator rates
4. **Archive Management:** Limit EP size for computational efficiency
5. **Diversity Preservation:** Use diversity mechanisms for better spread

## Performance Characteristics

- Effective for problems with:
  - Regular Pareto fronts
  - Moderate number of objectives (2-5)
  - Separable or partially separable fitness landscapes

- May struggle with:
  - Highly irregular or disconnected Pareto fronts
  - Very high-dimensional objective spaces (>10 objectives)
  - Problems with complex constraints

## Extensions and Variants

1. **MOEA/D-DE:** Uses Differential Evolution operators
2. **MOEA/D-DRA:** Dynamic Resource Allocation
3. **MOEA/D-AWA:** Adaptive Weight Adjustment
4. **MOEA/D-M2M:** Many-to-Many decomposition
5. **MOEA/D-STM:** Stable Matching model