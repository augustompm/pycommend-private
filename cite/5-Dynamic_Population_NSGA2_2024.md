# Speeding Up the NSGA-II via Dynamic Population Sizes

**Authors:** Benjamin Doerr¹, Martin S. Krejca¹, Simon Wietheger²

**Institutions:**
1. Laboratoire d'Informatique (LIX), CNRS, École Polytechnique, Institut Polytechnique de Paris
2. Algorithms and Complexity Group, TU Wien

**Published:** arXiv:2509.01739v1 [cs.NE] 1 Sep 2025

**Contact:** {first-name.last-name}@polytechnique.edu, swietheger@ac.tuwien.ac.at

## Abstract

Multi-objective evolutionary algorithms (MOEAs) are among the most widely and successfully applied optimizers for multi-objective problems. However, to store many optimal trade-offs (the Pareto optima) at once, MOEAs are typically run with a large, static population of solution candidates, which can slow down the algorithm.

This paper proposes the **dynamic NSGA-II (dNSGA-II)**, which is based on the popular NSGA-II and features a non-static population size. The dNSGA-II starts with a small initial population size of four and doubles it after a user-specified number τ of function evaluations, up to a maximum size of µ.

### Main Results

Via mathematical runtime analysis, we prove that the dNSGA-II with parameters µ ≥ 4(n + 1) and τ ≥ (256/5)en computes the full Pareto front of the ONEMINMAX benchmark of size n in **O(log(µ)τ + µ log(n))** function evaluations, both in expectation and with high probability.

For an optimal choice of µ and τ, the resulting **O(n log(n))** runtime improves the optimal expected runtime of the classic NSGA-II by a factor of **Θ(n)**.

In addition, we show that the parameter τ can be removed when utilizing concurrent runs of the dNSGA-II. This approach leads to a mild slow-down by a factor of O(log(n)) compared to an optimal choice of τ for the dNSGA-II, which is still a speed-up of Θ(n/log(n)) over the classic NSGA-II.

## Introduction

### Multi-Objective Optimization Context

Real-world problems often require the optimization of conflicting objectives, resulting in several incomparable optimal trade-offs, known as **Pareto optima**. Due to their incomparable nature, it is desirable to quickly find as many Pareto optima as possible—ideally all of them, called the **Pareto front**.

Multi-objective evolutionary algorithms (MOEAs) lend themselves well to this task, as they maintain a population of multiple solution candidates at once. Hence, MOEAs are among the most widely and effectively applied approaches in multi-objective optimization.

### State-of-the-Art MOEAs

Popular state-of-the-art MOEAs include:
- **NSGA-II** (Deb et al. 2002)
- **NSGA-III** (Deb and Jain 2014)
- **SMS-EMOA** (Beume, Naujoks, and Emmerich 2007)
- **MOEA/D** (Zhang and Li 2007)
- **SPEA2** (Zitzler, Laumanns, and Thiele 2001)

Recently, the empirical success of these algorithms has been complemented by theoretical analyses, which provide rigorous performance guarantees and insights into their merits and shortcomings.

### The Performance Bottleneck

Despite the large variety of MOEAs, the mathematical performance guarantees for simple benchmark problems are often asymptotically the same. For the ONEMINMAX (OMM) benchmark of size n—the most popular MOEA theory benchmark—the common runtime for an MOEA with a static population of size µ is:

**O(µn log(n))**, with the constraint µ ≥ C(n + 1) for some constant C ≥ 1

This constraint is a consequence of requiring that the population can represent the entire Pareto front of OMM, which has size n + 1. For optimal µ, this results in a bound of **O(n² log(n))**.

### The Gap with Single-Objective EAs

The closely related single-objective ONEMAX problem can be solved by evolutionary algorithms (EAs) in an expected time of **Θ(n log(n))**. The slow-down of MOEAs in comparison to single-objective EAs lies in:

1. **Large population size**: MOEAs need an at least linear population size on OMM
2. **Population clustering**: Large populations are likely to be clustered
3. **Wasted evaluations**: Only few solutions exist that easily lead to new, improving solutions

This results in many wasted function evaluations in each iteration.

## Problem Statement

### The ONEMINMAX Benchmark

The **ONEMINMAX (OMM)** benchmark function is the most commonly studied benchmark in the theory of MOEAs.

**Definition:** OMM: {0, 1}ⁿ → [n]₀²

For a bit string x ∈ {0, 1}ⁿ:
```
OMM(x) = (n - |x|₁, |x|₁)
```

where |x|₁ denotes the number of ones in x.

### Conflicting Objectives

OMM features two conflicting objectives:
1. **First objective**: Returns the number of zeros in a given individual
2. **Second objective**: Returns the number of ones

Due to the conflicting objectives, **all individuals are Pareto-optimal**, and the Pareto front is:
```
{(i, n-i) | i ∈ [n]₀}
```

The Pareto front has size **n + 1**.

### Runtime Definition

The **runtime** of an MOEA is the (random) number of function evaluations until the current population covers the Pareto front for the first time, that is, until the objective values of the population contain the Pareto front.

### Empty Intervals

Given a population P of individuals containing at least 0ⁿ and 1ⁿ, for all i ∈ [n], the **iᵗʰ empty interval** Iᵢ (of P for OMM) is defined as:

```
Iᵢ = [max{f₁(x) | x ∈ P ∧ OMM₁(x) ≤ i - 0.5}..
      min{f₁(x) | x ∈ P ∧ OMM₁(x) ≥ i - 0.5}]
```

The **maximum empty interval (MEI)** is:
```
MEI(P) = max{|Iᵢ| | i ∈ [n]}
```

If P covers the Pareto front, then MEI(P) = 1.

## The Classic NSGA-II

### Algorithm Overview

The NSGA-II (Algorithm 1) maintains a multi-set of promising individuals (the parent population). The initial population P₀ of size µ is generated uniformly at random and updated iteratively while maintaining its size µ.

### Key Components

#### 1. Mutation (Offspring Creation)

In each iteration, the NSGA-II creates an offspring population of size µ by performing **standard bit mutation** on each individual in the parent population.

**Standard bit mutation:** Given a parent x ∈ {0, 1}ⁿ, creates offspring y ∈ {0, 1}ⁿ by:
1. Copying x to y
2. Flipping each bit of y independently with probability 1/n

#### 2. Non-Dominated Fronts

Given a combined population R, the **non-dominated fronts** are a partition (Fᵢ)ᵢ∈[k] of R defined recursively:

- **First front F₁:** All non-dominated individuals in R
  ```
  F₁ = {x ∈ R | ∄y ∈ R: y ≻ x}
  ```

- **Remaining fronts Fᵢ:** Non-dominated individuals in R \ ⋃ⱼ∈[i-1] Fⱼ

The NSGA-II attempts to add entire fronts in increasing order to its next parent population.

#### 3. Crowding Distance

When a front Fᵢ* leads to excess population size, the NSGA-II selects individuals based on **crowding distance** as a tie-breaker (higher values preferable).

**Crowding distance** of individual x ∈ Fᵢ* is the sum of crowding distances per objective:

For each objective j ∈ {1, 2}:
- Sort Fᵢ* by objective j: (yₖ)ₖ∈[|Fᵢ*|]
- If x is in first or last position: crowding distance = +∞
- Otherwise, at position k*: crowding distance =
  ```
  (fⱼ(yₖ*₊₁) - fⱼ(yₖ*₋₁)) / (fⱼ(y|Fᵢ*|) - fⱼ(y₁))
  ```

#### 4. Current Crowding Distance (Modification)

The **current crowding distance** (Zheng and Doerr 2024a) is a modification of the classic crowding distance with the useful property of decreasing the maximum distances of the resulting population.

It achieves this by **iteratively**:
1. Removing an individual with the smallest (classic) crowding distance
2. Recomputing all crowding distances
3. Repeating until desired population size is achieved

This leads to a roughly **equidistant distribution** of individuals on the Pareto front, which is useful when the population size is smaller than the Pareto front size.

## The Dynamic NSGA-II (dNSGA-II)

### Core Innovation

The dNSGA-II (Algorithm 1) runs the NSGA-II with current crowding distance, starting with an **initial population of size 4**, which it **periodically attempts to double** in size.

An attempt is successful if and only if the current population size is less than the user-defined maximum population size µ ∈ ℕ.

### Phases

The consecutive function evaluations between attempts are called **phases**:
- **Phase 0:** Starts at the beginning of the algorithm run
- **Phase i:** Ends at the iteration in which population size is attempted to be doubled for the iᵗʰ time
- Number of phases is unbounded, but number of doublings is bounded by log(µ/4)

### Two Variants

#### Variant 1: (τ, µ)-dNSGA-II

**Phase length:** User-specified uniform phase length τ ∈ ℕ

Each phase lasts the same number τ of function evaluations.

**Population growth:**
```
Phase 0: 4 individuals, τ evaluations
Phase 1: 8 individuals, τ evaluations
Phase 2: 16 individuals, τ evaluations
...
Phase d: µ individuals (where 4·2^d ≥ µ)
```

#### Variant 2: (τ, µ)⁺-dNSGA-II

**Phase 0 length:** Extended to match total evaluations after Phase 0 until reaching µ

Let d = ⌈log(µ/4)⌉ be the number of doublings. Phase 0 length is set to **dτ**.

**Rationale:** The number of individuals created in Phase 0 is roughly the same as the number created afterward until population size reaches µ.

**Initialization:** In Algorithm 1, w is initialized to -(d-1)τ for the (τ, µ)⁺-dNSGA-II.

### Algorithm Structure

```python
# Initialization
if (τ, µ)-dNSGA-II:
    w = 0
if (τ, µ)⁺-dNSGA-II:
    w = -(⌈log(µ/4)⌉ - 1)τ

t = 0
P₀ = 4 individuals uniformly at random

# Main loop
while termination criterion not met:
    N_t = |P_t|
    Generate offspring Q_t with size N_t
    w = w + N_t

    # Check if doubling condition met
    if w ≥ τ and N_t < µ:
        P_{t+1} = P_t ∪ Q_t  # Double population
        w = 0
    else:
        # Standard NSGA-II selection
        Partition R_t = P_t ∪ Q_t into non-dominated fronts
        Select P_{t+1} based on fronts and current crowding distance

    t = t + 1
```

## Main Runtime Results

### Theorem 2: Performance Guarantees

Consider the dNSGA-II optimizing OMM with µ ≥ 4(n + 1). Let T denote the number of iterations until the population covers the Pareto front, and let F denote the number of function evaluations in the first T iterations.

#### For (τ, µ)-dNSGA-II

If τ ≥ 8en ln(n), then:
- **E[T] = O(τ)**
- **E[F] = O((τ + µ) log(n))**

These bounds also hold with probability at least 1 - 4/n.

#### For (τ, µ)⁺-dNSGA-II

If τ ≥ (256/5)en, then:
- **E[T] = O(log(µ)τ)**
- **E[F] = O(log(µ)τ + µ log(n))**

These bounds also hold with probability at least 1 - 4/n.

#### Independence from µ

If τ ≥ 520e(n + 1) ln(n), then:
- **F = O(log(n)τ)** for the (τ, µ)-dNSGA-II
- **F = O(log(µ)τ)** for the (τ, µ)⁺-dNSGA-II

Each with probability at least 1 - 4/n.

### Optimal Parameter Choices

#### (τ, µ)-dNSGA-II
With optimal parameters:
- Runtime: **O(n log²(n))**
- Speed-up over classic NSGA-II: **Ω(n/log(n))**

#### (τ, µ)⁺-dNSGA-II
With optimal parameters:
- Runtime: **O(n log(n))** ✓ **OPTIMAL**
- Speed-up over classic NSGA-II: **Θ(n)**

**Note:** O(n log(n)) is optimal, as finding even one of 0ⁿ and 1ⁿ requires Ω(n log(n)) function evaluations with standard bit mutation and four individuals.

### Comparison with Classic NSGA-II

| Algorithm | Runtime | Population | Speed-up |
|-----------|---------|------------|----------|
| Classic NSGA-II | O(n² log(n)) | Static: µ ≥ 4(n+1) | Baseline |
| (τ, µ)-dNSGA-II | O(n log²(n)) | Dynamic: 4 → µ | Ω(n/log(n)) |
| (τ, µ)⁺-dNSGA-II | **O(n log(n))** | Dynamic: 4 → µ | **Θ(n)** |

### Trade-offs Between Variants

**Comparison:**

| Feature | (τ, µ)-dNSGA-II | (τ, µ)⁺-dNSGA-II |
|---------|-----------------|-------------------|
| Runtime | O(n log²(n)) | O(n log(n)) |
| Dependency on µ | Weak (removable) | Stronger |
| τ requirement | Larger (8en ln(n)) | Smaller ((256/5)en) |
| High-probability bound | Can eliminate µ | Cannot eliminate µ |

**Remarkable feature of (τ, µ)-dNSGA-II:** The partial independence on µ implies that the algorithm does not suffer from a too large µ. Since µ only acts as an upper bound, it is possible to set **µ = ∞**, eliminating the choice of µ entirely.

## Removing Parameter τ: Concurrent Runs

### Algorithm 2: Automatic Parameter Selection

To relieve users from manually selecting τ, we propose a framework that **concurrently runs multiple instances** of the dNSGA-II with distinct values for τ, namely all powers of two.

For each τ = 2ⁱ, we consider an instance Aᵢ of the dNSGA-II.

### Selection Strategy

The framework repeatedly picks an instance Aᵢ and runs its next phase. Each time, Aᵢ is picked such that the **total number of function evaluations** spent on Aᵢ after the potential next phase is **minimal among all currently started instances**.

```python
# Initialization
for i ∈ ℕ≥2:
    rᵢ = 0  # evaluations received
    φᵢ = 0  # phase number

# Main loop
while no instance meets termination criterion:
    # Pick instance with minimal projected evaluations
    i = arg min{rⱼ + PhaseLength(µ, j, φⱼ)}

    if φᵢ = 0:
        initialize dNSGA-II instance Aᵢ with µ and τ = 2ⁱ

    Run the next phase of Aᵢ
    φᵢ = φᵢ + 1
    rᵢ = rᵢ + PhaseLength(µ, i, φᵢ)
```

### PhaseLength Function

For (τ, µ)-dNSGA-II:
```
PhaseLength(µ, i, φ) = max{2ⁱ, 4·2^min{φ,⌈log(µ/4)⌉}}
```

For (τ, µ)⁺-dNSGA-II:
```
PhaseLength(µ, i, 0) = ⌈log(µ/4)⌉ · 2ⁱ
PhaseLength(µ, i, φ) = max{2ⁱ, 4·2^min{φ,⌈log(µ/4)⌉}} for φ > 0
```

### Theorem 5: Performance of Concurrent Runs

Consider Algorithm 2 optimizing OMM with µ ≥ 4(n + 1). Let F denote the total number of function evaluations until one of the instances covers the Pareto front.

#### For (τ, µ)-dNSGA-II instances:
**E[F] = O(n log³(n))**

#### For (τ, µ)⁺-dNSGA-II instances:
**E[F] = O(min{µ, n log(n log(µ))} log(µ) log(n))**

Furthermore, all bounds hold with probability at least 1 - 4/n.

### Parameterless MOEA

**Key insight:** Algorithm 2 with (τ, µ)-dNSGA-II effectively eliminates both τ and µ as parameters, as its runtime result is independent of µ (once sufficiently large).

**Result:** A **parameterless MOEA** with runtime O(n log³(n)), which is still a **speed-up of Θ(n/log²(n))** over the classic NSGA-II.

### Slow-down Analysis

The concurrent scheme leads to a slow-down of only **O(log(n))** compared to an optimal choice of τ for the dNSGA-II, but this is still a speed-up of **Θ(n/log(n))** over the classic NSGA-II.

## Dynamic Population Sizes Versus Archives

### Archives in MOEAs

An **archive** is an external memory that an MOEA may use to maintain best-so-far solutions. This allows MOEAs to keep a small population size while storing good solutions that do not affect the run further.

### NSGA-II with Archive

**Theoretical result** (Bian et al. 2024): The NSGA-II combined with an archive optimizes OMM in **O(n log(n))** function evaluations with optimal parameters (constant population size).

**Key requirement:** The analysis relies heavily on **crossover**, which combines different solutions into a single new one.

**Limitation:** For the OMM benchmark, it is possible to efficiently create the entire Pareto front by relying solely on crossover applied to the two extreme points of the Pareto front (0ⁿ and 1ⁿ).

**Problem:** Without crossover, it is unlikely that the NSGA-II with an archive and constant population size can witness the entire Pareto front of OMM, as the few solutions in the population make it very unlikely to create all other solutions.

### Comparison: dNSGA-II vs Archive Approach

| Feature | NSGA-II + Archive | dNSGA-II |
|---------|-------------------|----------|
| Population size | Constant (small) | Dynamic (4 → µ) |
| Crossover | **Required** for O(n log(n)) | **Not required** |
| Dependency | Problem structure | Population spreading |
| Generality | Problem-specific | General approach |

**Key distinction:** The dNSGA-II results do not rely on crossover (but allow for it). The main motivation for gradually increasing population size is that the **population by itself is capable of exploring the search space quickly** if it has enough time to spread.

**Conclusion:** Dynamic population sizes are generally very powerful when applied with the correct phase length, and they are **less reliant on problem-specific operators** than archive-based approaches.

## Mathematical Tools and Proofs

### Theorem 1: Classic NSGA-II Baseline

**Based on Zheng and Doerr (2024a, Theorem 16):**

Let µ ∈ ℕ≥4 and τ ∈ ℕ. Consider the classic NSGA-II with current crowding distance, or the dNSGA-II optimizing OMM. If there is t such that |Pₜ| ≥ 4(n + 1), then after **O(n log n)** expected iterations, the algorithm covers the Pareto front.

### Lemma 3: Finding Extreme Points

Consider the NSGA-II or dNSGA-II optimizing OMM starting from an arbitrary population of size µ ≥ 4. Let T denote the number of iterations until the population contains 1ⁿ and 0ⁿ. Then:

**T ≤ 2en ln(n)** with probability at least 1 - 2/n

**Proof sketch:** Uses concentration bound for sums of independent geometrically distributed random variables.

### Lemma 4: Reducing Empty Intervals

Consider the NSGA-II or dNSGA-II optimizing OMM at iteration t ∈ ℕ such that {0ⁿ, 1ⁿ} ⊆ Pₜ. Let m, m' ∈ ℕ such that max{2n/(|Pₜ|-3), 1} ≤ m' ≤ MEI(Pₜ) ≤ m, and let T ≥ 0 be minimal such that MEI(Pₜ₊ₜ) ≤ m'. Then:

**T ≤ 4e(m - m')** with probability at least 1 - n exp(-(m-m')/4)

**Key property:** Once 0ⁿ and 1ⁿ are found, the maximum empty interval quickly reduces.

### Proof Outline of Theorem 2

The proof analyzes distinct phases:

1. **Phase 0:** Bound probability that population contains 0ⁿ and 1ⁿ
   - Uses Lemma 3: at least 2en ln(n) iterations needed
   - After finding extremes: MEI(Pₜ₀) ≤ n ≤ 2n/(N₀-3) as N₀ = 4

2. **Subsequent phases:** Estimate probability that MEI is halved in each phase
   - Uses Lemma 4: empty intervals reduce quickly
   - Population size doubles periodically: 4 → 8 → 16 → ... → µ
   - Number of doublings: d = ⌈log(µ/4)⌉

3. **Final phase:** Once MEI is small enough (≤ 16 ln(n) + 8 ln(log(n))), remaining Pareto front is quickly sampled
   - Uses Theorem 1: O(n log n) iterations suffice
   - Population size at least 4(n + 1)

4. **Function evaluations:** Count evaluations across all phases
   - Each phase i with population Nᵢ requires at most τ + Nᵢ evaluations
   - Total: O(τ log(n) + µ log(n))

### High-Probability vs Expectation

**High probability bound:** With probability at least 1 - 4/n, the algorithm covers Pareto front in O(τ log(n)) or O(log(µ)τ + µ log(n)) evaluations.

**Expected value:** Uses the fact that high-probability bounds allow bounding the expected runtime by considering the low-probability case separately:

E[F] ≤ (High prob. bound) · 1 + (Worst case) · Pr[failure]
     = O(bound) + O(µn log(n)) · O(1/n)
     = O(bound)

## Experimental Implications

### Parameter Recommendations

#### For (τ, µ)-dNSGA-II:
- **µ:** Set to ≥ 4(n + 1), or µ = ∞ to eliminate parameter
- **τ:** Set to ≥ 8en ln(n) for good performance
  - Optimal: τ = Θ(n ln(n)) → runtime O(n log²(n))
  - Larger τ up to o(n²) still beats O(n² log(n))

#### For (τ, µ)⁺-dNSGA-II:
- **µ:** Set to ≥ 4(n + 1)
- **τ:** Set to ≥ (256/5)en for good performance
  - Optimal: τ = Θ(n) → runtime O(n log(n))

#### For Algorithm 2 (Concurrent):
- **µ:** Set to ≥ 4(n + 1), or µ = ∞ for (τ, µ)-dNSGA-II variant
- **τ:** Automatically handled (no manual selection needed)
- **Runtime:** O(n log³(n)) for (τ, µ)-dNSGA-II variant

### Applicability Beyond Standard Operators

**Important note:** All results apply not only to standard bit mutation and fair selection, but also to:
- **Uniform parent selection**
- **One-bit mutation**
- **Crossover** with crossover rate at most a constant less than 1

**Requirement:** For all iterations, for each Hamming neighbor of the population, there is a constant probability to create it in this iteration.

## Related Work and State of the Art

### MOEA Runtime Analysis

**Common runtime for MOEAs on OMM:**
- NSGA-II: O(n² log(n)) (Doerr and Qu 2023)
- NSGA-III: Similar bounds (Wietheger and Doerr 2023)
- SMS-EMOA: Similar bounds (Zheng and Doerr 2024b)
- MOEA/D: Similar bounds (Li et al. 2016; Ren et al. 2024)
- SPEA2: Similar bounds (Ren et al. 2024)

**Meta-theorem:** Wietheger and Doerr (2024) state a meta theorem that applies to a plethora of MOEAs and results in the same performance guarantees: O(µn log(n)) with µ ≥ C(n+1).

### Improvements to Classic MOEAs

**Recent improvements:**
- Better approximation guarantees for NSGA-II using current crowding distance (Zheng and Doerr 2024a)
- Improved NSGA-II version (Doerr, Ivan, and Krejca 2025): Still O(n² log(n))
- Archive-based NSGA-II (Bian et al. 2024): O(n log(n)) with crossover

**Novel contributions of this work:**
- First theoretical speed-up via dynamic population sizes for MOEAs
- Removes dependency on crossover
- Applicable to general problem structures

### Dynamic Population Sizes in Single-Objective EAs

**Previous work:** For single-objective problems, dynamic updates to population size can result in provable speed-ups over static choices (Doerr and Doerr 2018).

**Gap:** No such result was previously known for the multi-objective domain until this work.

## Conclusion

### Main Contributions

1. **Introduced dNSGA-II:** Dynamic NSGA-II with periodically increasing population size
   - Two variants: (τ, µ)-dNSGA-II and (τ, µ)⁺-dNSGA-II
   - Starts with population size 4, doubles periodically up to µ

2. **Proven speed-ups:** For optimal parameters, achieves factors up to **Θ(n)** over classic NSGA-II (Theorem 2)
   - (τ, µ)-dNSGA-II: O(n log²(n)) runtime
   - (τ, µ)⁺-dNSGA-II: **O(n log(n)) runtime (optimal)**

3. **Concurrent scheme:** Algorithm 2 eliminates parameter τ (and µ for one variant)
   - Leads to parameterless MOEA
   - Slow-down of only O(log(n)) compared to optimal τ
   - Still Θ(n/log(n)) speed-up over classic NSGA-II

4. **Generality:** Results do not rely on crossover
   - Population spreading provides exploration power
   - Less problem-specific than archive approaches

### Comparison Summary

| Algorithm | Runtime | Parameters | Speed-up vs NSGA-II |
|-----------|---------|------------|---------------------|
| Classic NSGA-II | O(n² log(n)) | µ | Baseline |
| (τ, µ)-dNSGA-II | O(n log²(n)) | µ, τ | Ω(n/log(n)) |
| (τ, µ)⁺-dNSGA-II | **O(n log(n))** | µ, τ | **Θ(n)** |
| Algorithm 2 + (τ, µ)-dNSGA-II | O(n log³(n)) | **None** | Θ(n/log²(n)) |
| Algorithm 2 + (τ, µ)⁺-dNSGA-II | O(µ log(µ) log(n)) | µ | Depends on µ |

### Future Research Directions

**Open questions:**
1. Does dNSGA-II maintain its advantage on other popular benchmarks?
   - **LOTZ** (Laumanns, Thiele, and Zitzler 2004)
   - **OJZJ** (Doerr and Zheng 2021)

2. Can dynamic population sizes improve other MOEAs?
   - NSGA-III, SMS-EMOA, MOEA/D, SPEA2

3. What is the optimal phase length strategy?
   - Fixed vs adaptive phase lengths
   - Problem-dependent adjustments

4. Practical performance on real-world problems?
   - Engineering design optimization
   - Machine learning hyperparameter tuning

### Practical Implications

The dNSGA-II demonstrates that:
- **Dynamic strategies are powerful:** Adjusting population size during the run can dramatically improve performance
- **Small initial populations work:** Starting with just 4 individuals is sufficient
- **Spreading takes time:** Giving the population time to spread before increasing size is crucial
- **Parameterless MOEAs are possible:** Concurrent runs can eliminate manual parameter tuning

**Recommendation:** The dNSGA-II is drastically more efficient than many state-of-the-art MOEAs for the OMM benchmark, suggesting that dynamic population strategies should be explored for practical multi-objective optimization problems.

## Key References

**Classic MOEAs:**
- Deb et al. (2002): NSGA-II
- Zhang and Li (2007): MOEA/D
- Zitzler, Laumanns, and Thiele (2001): SPEA2

**Runtime Analysis:**
- Zheng and Doerr (2024a): Current crowding distance for NSGA-II
- Doerr and Qu (2023): Lower bounds for NSGA-II
- Wietheger and Doerr (2024): Meta-theorem for MOEAs

**Dynamic Population Sizes:**
- Doerr and Doerr (2018): Dynamic population sizes for single-objective EAs

**Archives:**
- Bian et al. (2024): Archive-based NSGA-II with crossover

---

*Full paper with proofs available on arXiv:2509.01739*

**Keywords:** Multi-objective optimization, evolutionary algorithms, NSGA-II, dynamic population size, runtime analysis, ONEMINMAX benchmark
