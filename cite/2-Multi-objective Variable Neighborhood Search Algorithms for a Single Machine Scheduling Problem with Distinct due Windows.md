# Multi-objective Variable Neighborhood Search Algorithms for a Single Machine Scheduling Problem with Distinct due Windows

**Authors:** José Elias Claudio Arroyo, Rafael dos Santos Ottoni, Alcione de Paiva Oliveira

**Institution:** Departamento de Informática, Universidade Federal de Viçosa, Viçosa - MG - Brazil

**Published:** Electronic Notes in Theoretical Computer Science 281 (2011) 5–19

**DOI:** 10.1016/j.entcs.2011.11.022

## Abstract

This paper compares three multi-objective algorithms based on Variable Neighborhood Search (VNS) heuristic. The algorithms are applied to solve the single machine scheduling problem with sequence dependent setup times and distinct due windows. The problem considers minimizing the total weighted earliness/tardiness and the total flowtime criteria. Two intensification procedures are introduced to improve a multi-objective VNS (MOVNS) algorithm proposed in the literature. The performance of the algorithms is tested on a set of medium and larger instances of the problem. Computational results show that the proposed algorithms outperform the original MOVNS algorithm in terms of solution quality. A statistical analysis is conducted to analyze the performance of the proposed methods.

## Problem Statement

### Single Machine Scheduling Problem (SMSP)

**Problem Definition:**
- Set of n jobs to be processed on a continuously available single machine
- Machine can process only one job at a time
- Each job j has:
  - Known processing time pj
  - Due window [dej, dtj] where dej is earliest due date and dtj is latest due date
  - Earliness penalty (αj) and tardiness penalty (βj)
  - Sequence dependent setup time sij between jobs i and j

**Completion Time Rules:**
- If Cj ∈ [dej, dtj]: No penalties (Tj = 0, Ej = 0)
- Otherwise: Incurs earliness (αjEj) or tardiness (βjTj) penalties
- Earliness: Ej = max{0, dej − Cj}
- Tardiness: Tj = max{0, Cj − dtj}

**Machine Idle Time:**
- Allowed in this problem
- May be required to complete a job within its due window
- Avoids earliness penalties

### Bi-Objective Formulation

**Objective Functions:**

For a sequence s = (i1, ..., ij, ..., in):

1. **f1(s): Total weighted earliness/tardiness penalties** (primary objective)
```
f1(s) = Σ(αjEj + βjTj)  for j=1 to n
```

2. **f2(s): Total flow time** (secondary objective)
```
f2(s) = ΣCj  for j=1 to n
```

**Objective Priority:**
- f1 is considered more important than f2
- f2 is computed after computing optimal completion times
- Objectives are generally conflicting

### Optimal Timing Algorithm

**From Wang and Yen (2002):**
- Given a job sequence, determines optimal completion time for each job
- Decides optimal completion time according to corresponding due window
- Goal: Minimize earliness/tardiness penalties for each sequence
- Complexity: Polynomial time

### Example Problem

**5-Jobs Instance:**

| Parameter | Job 1 | Job 2 | Job 3 | Job 4 | Job 5 |
|-----------|-------|-------|-------|-------|-------|
| Processing time (pj) | 9 | 15 | 8 | 12 | 5 |
| Earliest due date (dej) | 15 | 150 | 22 | 140 | 21 |
| Latest due date (dtj) | 25 | 170 | 30 | 180 | 22 |
| Earliness penalty (αj) | 3 | 2 | 4 | 1 | 5 |
| Tardiness penalty (βj) | 7 | 6 | 8 | 4 | 10 |

**Example Sequences:**
- Sequence (1, 5, 3, 4, 2): f1 = 0, f2 = 360 (with idle times)
- Sequence (5, 4, 1, 2, 3): f1 = 580, f2 = 138 (no idle times)

## Multi-objective Optimization Concepts

### Pareto Optimality

**Definition 1 (Dominance):**
A solution s dominates s' if:
- fi(s) ≤ fi(s') for all objectives i
- fi(s) < fi(s') for at least one objective i

**Definition 2 (Pareto-optimal):**
A solution s is Pareto-optimal (efficient) if there is no s' such that f(s') dominates f(s).

**Goal:**
Identify all elements belonging to the Pareto or efficient set containing all non-dominated alternatives.

## Multi-objective VNS Algorithms

### Background

**Variable Neighborhood Search (VNS):**
- Metaheuristic based on systematic change of neighborhood during search
- Originally proposed by Mladenovic and Hansen (1997)
- Effective for single-objective optimization problems

**First Multi-objective VNS:**
- Developed by Geiger (2004)
- Random selection of neighborhoods
- Arbitrary selection of base solution from unvisited non-dominated solutions

### MOVNS1 Algorithm (Baseline)

**Algorithm Structure:**

```
Input: None
Output: Set D of non-dominated solutions

1. Generate initial solutions {s1, s2, s3} using dispatching rules
2. Initialize D with non-dominated solutions from {s1, s2, s3}
3. While StoppingCriterion = false:
   a. Select randomly an unvisited solution s from D (base solution)
   b. Mark s as visited
   c. Select at random a neighborhood structure Ni
   d. Determine randomly a solution s' from Ni(s) (shaking)
   e. For each neighbor s'' ∈ Ni(s'):
      - Evaluate solution s''
      - Update D with non-dominated solutions from D ∪ {s''}
   f. If all solutions in D are marked as visited:
      - Remove all marks
4. Return D
```

**Key Features:**
- Uses greedy heuristics for initialization
- Two neighborhood structures: Insertion (N1) and Exchange (N2)
- Random neighborhood selection
- Updates Pareto front continuously

### Initial Solutions

**Dispatching Rules Used:**

1. **Earliest Due Date (EDD):**
   - s1: Jobs arranged by increasing dej
   - s2: Jobs arranged by increasing dtj

2. **Shortest Processing Time (SPT):**
   - s3: Jobs arranged by increasing total processing time

**Initialization:**
- Set D contains non-dominated solutions from {s1, s2, s3}
- Guarantees at least one solution in D

### Neighborhood Structures

**N1: Insertion Neighborhood**
- Insert job iq (1 ≤ q ≤ n) in another position k
- If k < q, then k ≠ q − 1
- Size: (n − 1)²

**N2: Exchange Neighborhood**
- Interchange jobs iq and ip (1 ≤ q ≤ n, 1 ≤ p ≤ n, q ≠ p)
- Size: n(n − 1)/2

**Local Search Strategy:**
1. Select base solution s randomly from D
2. Mark s as visited
3. Select neighborhood Ni randomly
4. Perturb s by choosing random s' from Ni(s) (shaking)
5. Explore all neighbors in Ni(s')

### MOVNS2 Algorithm (With Intensification1)

**Algorithm Structure:**

```
Input: Parameter d (number of jobs to remove)
Output: Set D of non-dominated solutions

1. Generate initial solutions and initialize D
2. While StoppingCriterion = false:
   a. Select randomly an unvisited solution s from D
   b. Mark s as visited
   c. Select at random a neighborhood structure Ni
   d. Determine randomly a solution s' from Ni(s)
   e. Initialize Da = Φ
   f. For each neighbor s'' ∈ Ni(s'):
      - Evaluate solution s''
      - Da ← non-dominated solutions from Da ∪ {s''}
   g. Select randomly a solution s from Da
   h. Db ← Intensification1(s, d)
   i. D ← non-dominated solutions from D ∪ Da ∪ Db
   j. If all solutions marked as visited: Remove all marks
3. Return D
```

**Key Improvement:**
Adds intensification procedure based on scalarizing functions after local search.

### Intensification1 Procedure

**Based on Scalarizing Functions:**

```
Input: Solution s, parameter d
Output: Set Db of non-dominated solutions

1. Initialize Db = Φ
2. Define randomly weights w1 and w2 such that w1 + w2 = 1
3. Remove d random jobs from s to create partial sequence sp
4. Store removed jobs in sR
5. For i = 1 to d:
   a. Insert job sR(i) in all positions of sp generating set S
   b. Evaluate each sequence s' ∈ S
   c. If d = n:
      - Db ← non-dominated solutions from Db ∪ S
   d. Else:
      - sp ← best solution from S w.r.t. fw = w1f1 + w2f2
6. Return Db
```

**Weighted Utility Function:**
- fw = w1f1 + w2f2
- Weights randomly generated each execution
- Explores different search directions on Pareto frontier

**Two Stages:**
1. **Destruction:** Remove d jobs randomly from solution
2. **Construction:** Rebuild solution incrementally, selecting best at each step

### MOVNS3 Algorithm (With Intensification2)

**Intensification2 Procedure:**

**Based on Pareto Dominance:**

```
Input: Solution s, parameter d
Output: Set Db of non-dominated solutions

1. Remove d random jobs from s to create partial sequence sp
2. Store removed jobs in sR
3. Initialize Db = {sp}
4. For i = 1 to d:
   a. Initialize Da = Φ
   b. For each partial sequence sp ∈ Db:
      - Insert job sR(i) in all positions of sp generating set S
      - Evaluate each sequence s' ∈ S
      - Da ← non-dominated solutions from Da ∪ {s'}
   c. Db ← Da
5. Return Db
```

**Key Difference from Intensification1:**
- Maintains multiple partial solutions at each step
- Selects ALL non-dominated partial solutions (not just best)
- Explores more solution space
- Final step returns complete non-dominated solutions

**Visual Example (n=5, d=2):**
- Step 1: (n−d+1) = 4 partial solutions → keep all non-dominated
- Step 2: Each non-dominated partial solution generates 5 complete solutions
- Result: Set of complete non-dominated solutions

## Computational Experiments

### Experimental Setup

**Hardware:**
- CPU: Intel Core Quad 2.4GHz
- RAM: 2.0 GB
- Language: C++

**Stopping Criterion:**
- CPU time = 3n seconds (n = number of jobs)
- Time-based criteria widely used in literature
- Assigns more time to larger instances

**Parameter Tuning:**
- Parameter d tested with values {2, 4, 6, 8}
- Best results with d = 6 for both MOVNS2 and MOVNS3

### Problem Instances

**Instance Generation (72 total instances):**

**Instance Sizes:**
- n ∈ {20, 30, 40, 50, 75, 100} (6 sizes)

**Job Parameters:**
- Processing time pj: Uniform [1, 100]
- Tardiness penalty βj: Uniform [20, 100]
- Earliness penalty αj: k × βj where k ∈ Uniform [0, 1]
- Setup times sij: Uniform [0, 50] for all pairs (i, j), i ≠ j

**Due Window Parameters:**
- Tardiness factor T ∈ {0.1, 0.2, 0.3, 0.4} (4 values)
- Relative range RDD ∈ {0.8, 1.0, 1.2} (3 values)
- Total: 4 × 3 = 12 settings

**Due Window Generation:**
- Center: Uniform [(1−T−RDD/2)TP, (1−T+RDD/2)TP]
- Size: Uniform [1, TP/n]
- Where TP = total processing time of all jobs

**Total Instances:**
- 12 settings × 6 sizes = 72 instances
- One instance per (T, RDD) pair for each size n

### Performance Measures

**Reference Set (Ref):**
- Constituted by gathering all non-dominated solutions from three algorithms
- Best known Pareto front for each instance
- Used since optimal Pareto front is unknown

#### 1. Cardinal Measure

**Definition:**
Number of obtained non-dominated solutions that belong to reference set:
- |Ref ∩ D1| for MOVNS1
- |Ref ∩ D2| for MOVNS2
- |Ref ∩ D3| for MOVNS3

#### 2. Distance Metrics

**Average Distance (dav):**
```
dav(Di) = (1/|Ref|) × Σ min{d(x,y) | x ∈ Di}  for y ∈ Ref
```

**Maximum Distance (dmax):**
```
dmax(Di) = max{min{d(x,y) | x ∈ Di}}  for y ∈ Ref
```

**Distance Function:**
```
d(x,y) = √[(f₁*(y) − f₁*(x))² + (f₂*(y) − f₂*(x))²]
```

**Normalization:**
```
fi*(x) = 100 × (fi(x) − fimin) / (fimax − fimin)
```
where fimax and fimin are max/min values in reference set Ref.

**Interpretation:**
- Smaller dav and dmax indicate better quality
- Widely employed measure for multi-objective problems

#### 3. Hypervolume Indicator

**Definition (Zitzler and Thiele, 2003):**
```
H(Di) = H*(Ref) − H*(Di)
```

where H*(X) is the hypervolume (area for 2 objectives) of solution space dominated by set X.

**Interpretation:**
- Smaller H(Di) corresponds to higher quality
- Indicates both better convergence and good coverage of reference set

### Experimental Results

**Run Configuration:**
- 5 independent runs (replicates) for all 72 instances
- Sets D1, D2, D3 contain non-dominated solutions from all runs

#### Cardinal Measure Results

| n | |Ref| | MOVNS1 | | MOVNS2 | | MOVNS3 | |
|---|------|--------|---------|--------|---------|--------|---------|
| | | |D1| | |Ref∩D1| | |D2| | |Ref∩D2| | |D3| | |Ref∩D3| |
| 20 | 1071 | 996 | 446 | 987 | 761 | 1050 | 628 |
| 30 | 1316 | 836 | 75 | 917 | 272 | 1321 | 981 |
| 40 | 1249 | 676 | 28 | 718 | 161 | 1280 | 1060 |
| 50 | 1235 | 558 | 12 | 635 | 148 | 1282 | 1075 |
| 75 | 1364 | 460 | 9 | 406 | 63 | 1435 | 1292 |
| 100 | 843 | 152 | 6 | 192 | 67 | 889 | 770 |
| **Total** | **7078** | **3678** | **576** | **3855** | **1472** | **7257** | **5806** |

**Key Findings:**
- MOVNS3 provides 82.0% (5806/7078) of reference solutions
- MOVNS2 provides 20.8% (1472/7078) of reference solutions
- MOVNS1 provides only 8.1% (576/7078) of reference solutions
- MOVNS3 superior for all instance groups except n=20
- MOVNS2 superior to MOVNS1 for all instance groups

#### Distance and Hypervolume Results

| n | MOVNS1 | | | MOVNS2 | | | MOVNS3 | | |
|---|--------|-------|---------|--------|-------|---------|--------|-------|---------|
| | dav | dmax | H×10⁻⁵ | dav | dmax | H×10⁻⁵ | dav | dmax | H×10⁻⁵ |
| 20 | 8.2 | 23.0 | 1836.8 | 0.6 | 9.5 | 137.4 | 8.0 | 19.6 | 1799.2 |
| 30 | 4.7 | 15.3 | 5751.9 | 3.3 | 18.1 | 3960.8 | 0.8 | 8.8 | 672.7 |
| 40 | 11.2 | 24.9 | 30058.4 | 8.2 | 29.1 | 22582.4 | 0.4 | 4.4 | 806.6 |
| 50 | 16.4 | 38.0 | 89502.2 | 9.7 | 37.3 | 48277.5 | 0.6 | 5.1 | 2088.0 |
| 75 | 37.7 | 65.0 | 642678.5 | 16.2 | 45.0 | 273728.3 | 0.2 | 5.4 | 8302.4 |
| 100 | 113.2 | 154.8 | 1430734.9 | 37.1 | 82.5 | 546607.6 | 1.0 | 6.0 | 36927.1 |
| **Avg** | **31.9** | **53.5** | **366760.5** | **12.5** | **36.9** | **149215.7** | **1.8** | **8.2** | **8432.7** |

**Key Findings:**
- MOVNS3 performs best on all three measures (except n=20)
- MOVNS2 performs better than MOVNS1 and MOVNS3 for n=20
- MOVNS2 notoriously better than MOVNS1 for all instance groups
- Average hypervolume: MOVNS3 best (8432.7), MOVNS2 second (149215.7), MOVNS1 worst (366760.5)

### Statistical Analysis

**Analysis of Variance (ANOVA):**
- Confidence level: 95% (α = 0.05)
- Response variables: dav and H
- Result: Measures statistically different for three algorithms (p-values = 0.00)

**Tukey Multiple Comparison Test:**
- 95% confidence intervals for each algorithm do not overlap
- Confirms measures are statistically significantly different
- MOVNS3 shows best performance with statistical significance

**Confidence Interval Analysis:**
- Individual 95% confidence intervals show no overlap
- MOVNS3 < MOVNS2 < MOVNS1 for both dav and H
- Clear statistical separation between algorithms

## Key Contributions

1. **Two Intensification Procedures:**
   - Intensification1: Based on weighted scalarizing functions
   - Intensification2: Based on Pareto dominance approach
   - Both improve original MOVNS significantly

2. **Comprehensive Experimental Validation:**
   - 72 problem instances across 6 sizes (20-100 jobs)
   - Multiple performance measures (cardinal, distance, hypervolume)
   - Statistical significance testing (ANOVA, Tukey)

3. **Significant Performance Improvements:**
   - MOVNS3 obtains 82% of reference solutions vs 8.1% for MOVNS1
   - Average hypervolume: 43× better for MOVNS3 vs MOVNS1
   - Improvements increase with problem size

4. **Partial Enumeration Heuristic:**
   - Destruction and construction stages
   - Balances exploration and exploitation
   - Works well with both scalarizing and Pareto dominance approaches

## Algorithm Comparison Summary

| Algorithm | Intensification | Reference Solutions | Avg Hypervolume | Best For |
|-----------|----------------|---------------------|-----------------|----------|
| MOVNS1 | None | 8.1% (576/7078) | 366760.5 | Small instances (n=20) |
| MOVNS2 | Scalarizing functions | 20.8% (1472/7078) | 149215.7 | Small instances |
| MOVNS3 | Pareto dominance | 82.0% (5806/7078) | 8432.7 | All instance sizes |

**Best Overall:** MOVNS3 with Intensification2 (Pareto dominance)

## Conclusions

### Main Findings

1. **Intensification procedures significantly improve solution quality:**
   - MOVNS2 and MOVNS3 vastly outperform original MOVNS1
   - Improvement increases with instance size
   - Statistical significance confirmed by ANOVA and Tukey tests

2. **Pareto dominance approach superior to scalarizing functions:**
   - MOVNS3 (Pareto) > MOVNS2 (scalarizing) > MOVNS1 (none)
   - Pareto approach maintains multiple candidates at each step
   - Better exploration of solution space

3. **VNS effective for multi-objective scheduling:**
   - Systematic neighborhood change beneficial
   - Random neighborhood selection provides diversification
   - Combination with intensification provides intensification

### Practical Implications

**For JIT Manufacturing:**
- Provides decision makers with diverse Pareto-optimal schedules
- Balances earliness/tardiness penalties with total flow time
- Handles sequence dependent setup times (realistic constraint)
- Allows machine idle time when beneficial

**Algorithm Selection:**
- Use MOVNS3 for best solution quality (recommended)
- Use MOVNS2 if computational resources limited
- MOVNS1 only suitable for very small instances

### Future Work

**Proposed Extensions:**
1. Apply MOVNS with intensification to other scheduling problems
2. Explore additional neighborhood structures
3. Investigate adaptive parameter tuning for d
4. Extend to multi-machine environments
5. Consider additional objectives (energy, tardiness count, etc.)

## References

**Key Papers Cited:**

- **Geiger (2004):** First multi-objective VNS algorithm
- **Wang and Yen (2002):** Optimal timing algorithm for distinct due windows
- **Mladenovic and Hansen (1997):** Original VNS metaheuristic
- **Zitzler and Thiele (2003):** Hypervolume indicator for multi-objective optimization
- **Lee and Choi (1995):** Genetic Algorithm for distinct due dates scheduling

**Related Work:**
- Tabu Search for scheduling (multiple authors)
- Genetic Algorithms for earliness/tardiness (Ribeiro et al., 2009-2010)
- Variable Neighborhood Search (Hansen and Mladenovic, 2003)
- Multi-objective metaheuristics review (Jones et al., 2002)

**Total References:** 30 papers cited covering scheduling, metaheuristics, and multi-objective optimization.
