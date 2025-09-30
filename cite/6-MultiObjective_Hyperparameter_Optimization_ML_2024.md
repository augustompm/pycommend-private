# Multi-Objective Hyperparameter Optimization in Machine Learning – An Overview

**Authors:** Florian Karl, Tobias Pielok, Julia Moosbauer, Florian Pfisterer, Stefan Coors, Martin Binder, Lennart Schneider, Janek Thomas, Jakob Richter, Michel Lang, Eduardo C. Garrido-Merchán, Juergen Branke, Bernd Bischl

**Institution:** Ludwig-Maximilians-Universität München, Germany (and collaborators)

**Published:** arXiv:2206.07438v3 [cs.LG] 6 Jun 2024

## Abstract

Hyperparameter optimization constitutes a large part of typical modern machine learning workflows. But in many applications, we are not only interested in optimizing ML pipelines solely for predictive accuracy; additional metrics or constraints must be considered when determining an optimal configuration, resulting in a multi-objective optimization problem.

This work introduces the reader to the basics of multi-objective hyperparameter optimization and motivates its usefulness in applied ML. Furthermore, we provide an extensive survey of existing optimization strategies, both from the domain of evolutionary algorithms and Bayesian optimization. We illustrate the utility of MOO in several specific ML applications, considering objectives such as:
- Operating conditions
- Prediction time
- Sparseness
- Fairness
- Interpretability
- Robustness

## 1. Introduction

### Key Challenges

**Hyperparameter Optimization (HPO) is often classified as:**
- Black-box optimization problem
- Outputs can be observed but no analytic expression known
- Often noisy and expensive
- Rarely a clear-cut, obvious, single performance metric

**Modern ML Requirements:**
- Models must be reliable, robust, accountable
- Efficient for seamless deployment
- Meet legal/ethical requirements
- Handle multiple stakeholders with different priorities

**Examples of Multi-Objective Challenges:**
- Medical diagnostics: Different costs for false negatives vs false positives
- Edge devices: Power consumption, memory capacity, inference latency vs accuracy
- Internet-of-things: Resource constraints with performance requirements

### Why Multi-Objective Optimization?

**Advantages over scalarization:**
- Often unclear how to define trade-off between objectives a priori
- MOO seeks to approximate the set of Pareto-optimal solutions
- Domain experts can analyze trade-offs post-hoc
- Informed decision without requiring a priori specification

**Pareto Optimal Solutions:**
- Solutions where you cannot improve any objective without degrading at least one other
- Provides set of alternatives with different trade-offs

## 2. Hyperparameter Optimization

### 2.1 The Machine Learning Problem

**Formal Definition:**
- Dataset D with n input-output pairs (x⁽ⁱ⁾, y⁽ⁱ⁾) sampled i.i.d. from ℙₓᵧ
- ML algorithm I(·, λ) configured by hyperparameters λ ∈ Λ
- Maps dataset D to model f: X → ℝᵍ
- Goal: Optimize expected generalization performance GE w.r.t. loss L

**Taxonomy of Black-Box Problems (Figure 1):**

**Stochasticity:**
- Deterministic
- Stochastic homoscedastic
- Stochastic heteroscedastic (highlighted for MOHPO)

**Domain:**
- Dimensionality: low dim d, high dim d
- Type: numerical, categorical, mixed numerical and categorical, hierarchical/structured (highlighted)

**Codomain:**
- Single-objective ℝ
- Multi-objective ℝᵐ, m ∈ {2,3,4} (highlighted)
- Many objective ℝᵐ, m ≥ 5

**Evaluation cost:**
- Cheap
- Expensive (highlighted)

### 2.2 Multi-Objective Hyperparameter Optimization

**General MOHPO Problem:**
Given evaluation criteria c₁: Λ̃ → ℝ, ..., cₘ: Λ̃ → ℝ with m ∈ ℕ:

```
arg min c(λ) = arg min (c₁(λ), c₂(λ), ..., cₘ(λ))
  λ∈Λ̃           λ∈Λ̃
```

Where:
- c: Λ̃ → ℝᵐ assigns m-dimensional cost vector to HPC λ
- Λ̃ is bounded subspace of hyperparameter space Λ (search space)
- All criteria assumed to be minimized (w.l.o.g.)

**MOHPO Characteristics:**
- Expensive, multi-objective optimization
- Mixed and hierarchical search space
- Possibly heteroscedastic in objectives
- Domain can be low or high dimensional

**Key Challenge:** How to handle multiple objectives with:
- Different scales (e.g., LU: -10000 to 0, SS: -1 to 0, RSS: 2 to 15)
- Different evaluation costs
- Possible noise/stochasticity
- Unknown correlations

### 2.3 Multi-Objective Machine Learning

**Important Distinction:**
- **First level parameters:** Model weights, decision rules (learned during training)
- **Second level hyperparameters:** Architecture, optimizers (set before training)

**This work focuses on:** Tuning second level hyperparameters (MOHPO)

**Multi-objective ML:** Methods that focus on learning first level parameters (sometimes together with second level)

**Feature Selection:** Borders both MOHPO and multi-objective ML

## 3. Foundations of Multi-Objective Optimization

### 3.1 Objectives and Constraints

**Constrained MOHPO Problem:**
```
arg min c(λ)
  λ∈Λ̃

subject to:
  k₁(λ) = 0, ..., kₙ(λ) = 0  (equality constraints)
  k̂₁(λ) ≥ 0, ..., k̂ₙ̂(λ) ≥ 0  (inequality constraints)
```

**Common Inequality Constraints:**
- Predictive performance threshold
- Energy consumption limit
- Memory requirements
- Fairness criteria satisfaction

**Objective vs Constraint:**
- Depends on use case
- Example: Is memory efficiency a goal to optimize or hard constraint?

### 3.2 Pareto Optimality

**Pareto Dominance:**

A vector c ∈ ℝᵐ (Pareto-)dominates c', written as c ≺ c', if and only if:
```
∀i ∈ {1,...,m}: cᵢ ≤ c'ᵢ  ∧
∃j ∈ {1,...,m}: cⱼ < c'ⱼ
```

**In other words:**
- λ dominates λ' if there is no criterion where λ' is superior to λ
- AND at least one criterion where λ is strictly better

**Weak Dominance:**
c weakly dominates c' (c ⪯ c') if c ≺ c' or all components equal

**Non-dominated/Pareto Optimal:**
A configuration λ* is non-dominated if no other λ ∈ Λ̃ dominates λ*

**Pareto Set P:**
```
P := {λ ∈ Λ̃ | ∄ λ' ∈ Λ̃ s.t. λ' ≺ λ}
```

**Pareto Front:** Image of P under c, written as c(P)

**Partial Order:**
- Two vectors can be incomparable
- Occurs when cᵢ < c'ᵢ for some i but c'ⱼ < cⱼ for some j
- No unique single best solution, but set of Pareto optimal solutions

### 3.3 Evaluation

#### 3.3.1 Comparing Solution Sets

**Weak Dominance of Sets:**
P̂ₛ₁ weakly dominates P̂ₛ₂ (P̂ₛ₁ ⪯ P̂ₛ₂) if:
- For every solution λ₂ ∈ S₂ there is at least one solution λ₁ ∈ S₁ which weakly dominates λ₂

**Better Relation:**
P̂ₛ₁ is better than P̂ₛ₂ (P̂ₛ₁ ⊳ P̂ₛ₂) if:
- P̂ₛ₁ ⪯ P̂ₛ₂, but NOT P̂ₛ₂ ⪯ P̂ₛ₁

**Four Qualities of Solution Set:**
1. **Convergence:** Proximity to true Pareto front
2. **Spread:** Coverage of the Pareto front
3. **Uniformity:** Evenness of distribution of solutions
4. **Cardinality:** Number of solutions

**Diversity = Spread + Uniformity**

#### 3.3.2 Quality Indicators

**Purpose:** Objective measurement of quantitative difference between solution sets

**Main Categories:**

**1. Distance-Based Indicators:**
- Require knowledge of true Pareto front (or approximation)
- Examples: Inverted Generational Distance (IGD), Dist2, ε-indicator

**2. Volume-Based Indicators:**
- Measure volume between approximated Pareto Front and reference point
- Examples: Hypervolume, R-class indicators, Integrated Preference Functional

**Hypervolume (HV) - Most Popular:**

```
HV_r(S) := μ(⋃_{λ∈S} domHC_r(λ))
```

Where:
- μ is Lebesgue measure
- domHC_r(λ) = {u ∈ ℝᵐ | cᵢ(λ) ≤ uᵢ ≤ rᵢ ∀i ∈ {1,...,m}}
- r is reference point

**Properties of Hypervolume:**
- Strictly Pareto compliant
- S₂ ⊲ S₁ ⇒ HV_r(S₂) < HV_r(S₁)
- True Pareto front maximizes HV
- Does not require knowledge of true Pareto front

**Reference Point Selection:**
- Often nadir point (worst objective values)
- Or point slightly beyond nadir

#### 3.3.3 Normalizing Objectives

**Problem:** Different objectives live on different scales
- May bias quality indicators like hypervolume
- Example: LU (-10000 to 0), SS (-1 to 0), RSS (2 to 15)

**Solution:** Normalize objectives to [0,1]

**When to Normalize:**
- At least for analysis and comparison of algorithms
- During optimization for some methods (essential)
- During evaluation (recommended)

**How to Normalize:**
- Use ideal and nadir points if known a priori
- Estimate from previously evaluated points
- Update estimates iteratively

**Ideal Point:** Minimum value for each objective
**Nadir Point:** Maximum value on Pareto front (harder to estimate)

**Method-Specific Impact:**
- Some MOO methods more robust to non-normalized objectives
- For others (e.g., decomposition-based), normalization is essential
- Addressed in Section 4 for specific methods

#### 3.3.4 Multimodal Multi-Objective Optimization

**Relevance:** When multiple solutions have similar objective values but very different decision space locations

**Applications:**
- If parts of decision space become infeasible, alternatives available
- Identifying all global optima (not getting trapped in local minima)

**Open Challenge:** No commonly accepted quality indicators exist for this

## 4. Multi-Objective Optimization Methods

**Focus:** A posteriori methods that return set of configurations P̂ approximating true Pareto set P

**Coverage:**
- Scalarization approaches (Section 4.1)
- Random and grid search (Section 4.2)
- Evolutionary algorithms (Section 4.3)
- Model-based optimization / Bayesian Optimization (Section 4.4)
- Multi-fidelity optimization (Section 4.5)

### 4.1 Scalarization Approaches

**Definition:** Function s: ℝᵐ × T → ℝ that maps m criteria to single criterion
- Configured by scalarization hyperparameters α ∈ T
- Transforms MOO to single-objective optimization

**Drawbacks:**
1. Scalarization hyperparameters α must be chosen sensibly (not trivial)
2. One single solution cannot adequately represent multi-objective problem with conflicting objectives

**Three Popular Scalarization Techniques:**

**1. Weighted Sum Approach:**
```
min_{λ∈Λ̃} ∑ᵢ₌₁ᵐ αᵢcᵢ(λ)
```

Properties:
- If αᵢ ≥ 0, i = 1,...,m ⇒ λ* is non-dominated in Λ̃
- For convex Λ̃ and convex cᵢ: every non-dominated solution can be found
- **Limitation:** Cannot find solutions on non-convex parts of Pareto frontier

**2. Tchebycheff Approach:**
```
min_{λ∈Λ̃} max_{i=1,...,m} [αᵢ|cᵢ(λ) - z*ᵢ|]
```

Where z*ᵢ = min_{λ∈Λ̃} cᵢ(λ) defines ideal reference point

Properties:
- For every optimal solution λ* exists combination of weights α̂
- Can find solutions on non-convex parts

**3. ε-Constraint Approach:**
Given constants (ε₂,...,εₘ) ∈ ℝᵐ⁻¹:
```
min_{λ∈Λ̃} c₁(λ)
subject to: c₂(λ) ≤ ε₂, ..., cₘ(λ) ≤ εₘ
```

Properties:
- Converts all but one objective into constraints
- Constraints must be sensibly chosen (challenging)

**Usage in MOHPO:**
- Used in MOEA/D (Section 4.3.3)
- Used in ParEGO (Section 4.4.2)
- Multiple scalar problems created instead of multi-objective one

### 4.2 Random and Grid Search

**Random Search:**
- Competitive baseline for single-objective HPO
- Generally preferred over grid search
- Anytime algorithm
- Scales better with low effective dimensionality
- Often surprisingly competitive

**Grid Search:**
- Less preferred
- Does not scale well
- Fails to provide good Pareto front approximation

**MOO Extension:** Trivial
- All points independently spawned and evaluated
- Simply return all non-dominated solutions from archive

**Usage:** Reasonable baselines when introducing more sophisticated methods

### 4.3 Evolutionary Algorithms (EAs)

#### 4.3.1 Fundamentals

**Evolutionary Algorithms:**
- Population-based, randomized meta-heuristics
- Inspired by principles of natural evolution
- Include: genetic algorithms, evolution strategies, evolutionary programming

**Basic Loop (Figure 4):**
1. Initialize population (often randomly)
2. Evaluate population
3. **Selection:** Select better solutions as parents
4. **Crossover & Mutation:** Generate offspring
5. **Survival Selection:** Keep population size constant
6. Repeat until stopping criterion

**Main Operators:**
- **Crossover:** Recombines information from two parents
- **Mutation:** Randomly perturbs a solution

**Advantages:**
1. No specific domain knowledge necessary
2. Ease of implementation
3. Low likelihood of local minima trapping
4. General robustness and flexibility
5. Straightforward parallelization
6. Well-suited for multi-objective optimization (population-based)

**For Multi-Objective:** Only selection step needs changing

#### 4.3.2 Multi-Objective Evolutionary Algorithms (MOEAs)

**Three Main Categories:**

**1. Pareto Dominance-Based:**
- Two-level ranking:
  - First level: Pareto dominance (coarse ranking)
  - Second level: Diversity measure (refinement)
- Example: NSGA-II

**2. Decomposition-Based:**
- Decompose original problem into single-objective subproblems
- Use scalarization with different parametrizations
- Solve simultaneously
- Example: MOEA/D

**3. Indicator-Based:**
- Use single metric (e.g., hypervolume)
- Selection governed by marginal contribution to indicator
- Example: SMS-EMOA

#### 4.3.3 Prominent MOEAs

**NSGA-II (Pareto Dominance-Based):**
- Non-dominated Sorting Genetic Algorithm II
- Still one of most popular MOEAs
- Popular baseline in benchmarks

**Algorithm:**
1. **Non-dominated Sorting:** Iteratively determine non-dominated solutions
2. **Crowding Distance:** Among solutions in each class:
   - Extreme solutions (best in each objective) ranked highest
   - Others ranked by crowding distance (sum of differences to left/right neighbor)

**Limitations:**
- Works well for 2 objectives
- Breaks down with more objectives (non-dominated sorting less discriminative)

**NSGA-III:** Alternative for higher number of objectives

**MOEA/D (Decomposition-Based):**
- Multi-objective EA based on Decomposition
- Decomposes into N scalar optimization problems
- Usually uses Tchebycheff scalarization

**Key Ideas:**
- Population = best solution for each sub-problem
- Each generation: create offspring for each sub-problem
- Parents selected from sub-problem's neighborhood
- Offspring replaces individuals if better for corresponding sub-problem
- Diversity maintained implicitly by sub-problem definitions

**Distribution of Solutions:**
- Governed by set of scalarizations chosen
- Challenging without good knowledge of Pareto frontier
- Solves sub-problems simultaneously with mutual influence

**SMS-EMOA (Indicator-Based):**
- S Metric Selection Evolutionary Multi-objective Optimization Algorithm
- Uses non-dominated sorting (like NSGA-II)
- Secondary criterion: marginal hypervolume contribution

**Marginal Hypervolume:**
```
ΔHV(i, R) := HV(R) - HV(R \ i)
```

**Algorithm:**
- Produces one offspring per generation
- Adds to population
- Discards worst solution based on ranking

#### 4.3.4 Relevant Software

**PlatEMO (MATLAB):**
- Very established package
- Wide range of EMOAs
- Generally recommended

**pymoo (Python):**
- Very established package
- Wide range of EMOAs
- Generally recommended

**mle-hyperopt (Python):**
- Offers MOHPO via NSGA-II directly
- Uses nevergrad package internally

### 4.4 Model-Based Optimization / Bayesian Optimization

#### 4.4.1 Bayesian Optimization Basics

**Why BO for HPO:**
- Sample efficient compared to other techniques
- Good choice for expensive black-box problems
- Well-suited for machine learning model optimization

**Key Strategy:**
- Model mapping λ ↦ c(λ) based on observed performance
- Use (non-linear) regression
- Approximating model = **surrogate model**

**Common Surrogate Models:**
- Gaussian Processes (GPs) - most widely used
- Random Forests (better for mixed/hierarchical spaces)

**BO Algorithm:**
1. Start with archive A of evaluated configurations (e.g., Latin Hypercube Sampling)
2. Fit surrogate model to archive
3. Predict performance ĉ(λ) and uncertainty σ̂(λ)
4. Compute **acquisition function** u(λ):
   - Encodes exploitation vs exploration trade-off
   - Exploitation: high predicted performance
   - Exploration: high uncertainty (under-explored areas)
5. Optimize acquisition function to find candidate λ⁺
6. Evaluate true objective c(λ⁺)
7. Add to archive and iterate

**Common Acquisition Functions:**

**Expected Improvement (EI):**
- Improvement over best solution found so far

**Lower Confidence Bound (LCB):**
- Treats uncertainty as additive bonus
- Control parameter κ balances exploration

#### 4.4.2 Multi-Objective Bayesian Optimization

**Two Main Approaches:**

**1. Scalarization-Based:**
- Transform MO problem to single objective
- Fit surrogates to approximate Pareto front
- Example: ParEGO

**2. Non-Scalarization:**
- Train independent surrogates for each output dimension
- Either:
  - Get acquisition function per dimension, use MOO to find candidates
  - Aggregate predictions into single-objective acquisition function

**ParEGO (Scalarization-Based):**

**Algorithm:**
1. Determine set of scalarization weights ensuring even Pareto front exploration:
   ```
   α = (α₁, α₂,..., αₘ) | ∑ αⱼ = 1 ∧ αⱼ = l/s, l ∈ {0,1,...,s}
   ```
   Generates (s+m-1 choose m-1) different weight vectors

2. Normalize output space to [0,1]
   - Easy for bounded metrics (accuracy, AUC)
   - Estimate bounds for unbounded metrics

3. Each iteration: create scalarized objective using **augmented Tchebycheff:**
   ```
   c_α(λ) = max_{j∈{1,...,m}} [αⱼcⱼ(λ)] + ρ[α · c(λ)]
   ```
   Where:
   - ρ is small positive constant
   - α is weight vector drawn uniformly from set
   - Second term breaks ties

4. Fit surrogate model on scalarized outcomes
5. Optimize EI on this model to propose new HPC

**ParEGO Extensions:**
- Parallel batch proposals: sample q different weight vectors per iteration
- Can be adapted to focus search on one objective by limiting max weights of others

**Concerns:**
- Uniformly sampled weights don't necessarily give best distribution of non-dominated points

**EHI / EHVI (Expected Hypervolume Improvement):**

**Algorithm:**
1. Fit surrogate model for each objective individually
2. Calculate EHI as expectation of hypervolume improvement over predictive distribution

**Formula:**
- Expectation of HV improvement when adding candidate
- Non-trivial multidimensional integral

**Computational Complexity:**
- Monte-Carlo approximations available
- KMAC method: O(n log n) for 3D, O(n^⌊m/2⌋) for m dimensions
- More efficient methods for m=2: O(n log n)

**Batch Proposals:**
- Divide objective space into sub-spaces
- Search for optimal solutions in each sub-space using truncated EHI

**SMS-EGO (S-Metric Selection-based EGO):**

**Algorithm:**
1. Each BO iteration: approximate each objective with separate surrogate
2. For each objective compute LCB → m-dimensional outcome u_LCB
3. Acquisition function value:
   ```
   u_SMS(λ) = HV_r(P̂ ∪ u_LCB(λ)) - HV_r(P̂) - p
   ```
   Where:
   - r reference point = max(P̂) + 1_m
   - p penalty for dominated solutions

**Penalty Purpose:**
- Without penalty: dominated solutions would have u_SMS = 0
- Penalty guides search towards non-dominated solutions even in dominated areas

**Multi-EGO:**

**Algorithm:**
1. Each iteration: approximate each objective with separate surrogate
2. Get single-objective acquisition function from each surrogate (e.g., EI)
3. Optimize m acquisition functions jointly as MOO problem itself
4. Use MOEA to get set of candidates with non-dominated acquisition values
5. Select multiple points to be evaluated

**Advantages:**
- Naturally leads to parallelization
- Always generates multiple proposals per iteration

**MESMO and PESMO (Information-Theoretic):**

**Key Ideas:**
- Model each objective with independent surrogate
- Pareto set modeled as random variable
- Compute entropy of Pareto set location
- Acquisition = expected reduction of entropy if point evaluated

**Advantages:**
- Global measure of uncertainty (not local/heuristic)
- Should explore search space more efficiently
- Linear combination of expected entropy of all predictive distributions

**MESMO:** Multi-objective Maximum Entropy Search
**PESMO:** Predictive Entropy Search for Multi-objective

#### 4.4.3 Relevant Software and Implementations

**Comprehensive Comparison (Table 1):**

| Framework | Scal. | HV | Info | Constraints | Parallel |
|-----------|-------|-----|------|-------------|----------|
| **Dragonfly** (Python) | ✓ | × | × | × | ✓ |
| **HyperMapper** (Python/C++) | ✓ | × | × | ✓ | ✓ |
| **OpenBox** (Python) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Ax** (Python) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **trieste** (Python) | ✓ | ✓ | × | ✓ | ✓ |
| **BoTorch** (Python) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **GPflowOpt** (Python) | × | ✓ | × | × | × |
| **mlr3mbo** (R) | ✓ | ✓ | × | × | ✓ |
| **Optuna** (Python) | ✓ | × | × | ✓ | ✓ |

**Legend:**
- **Scal.:** Scalarization-based approaches (e.g., ParEGO)
- **HV:** Hypervolume-based (e.g., EHVI, QNEHVI, SMS-EGO)
- **Info:** Information-theoretic approaches
- **Constraints:** Support for constrained optimization
- **Parallel:** Can utilize computational resources in parallel

**Most Comprehensive:**
- **Ax, trieste, BoTorch:** Support all major approaches
- **OpenBox:** Good all-around support

### 4.5 Multi-Fidelity Optimization

**Motivation:**
- Assumes existence of cheaper approximate functions
- Popular for expensive-to-evaluate black-box optimization
- Especially useful for deep learning (costly to fully train)

**Key Idea:**
- Optimize configuration **evaluation** (not just selection)
- Allocate resources efficiently:
  - More resources to configurations performing well at lower fidelity
  - Discard configurations showing worse performance
- Illustrated in Figure 5

**Advantages:**
1. Deals with expensive optimization problems
2. Handles mixed/hierarchical search spaces (when drawing configurations randomly)
3. Can be combined with model-based optimization (BO instead of random sampling)

**Multi-Objective Extensions:**

**Hyperband for MOO:**
- Define suitable performance indicator to decide resource allocation
- Can be achieved via:
  - Scalarization using random weights
  - Non-dominated sorting

**BO + Multi-Fidelity:**
- Configurations drawn via BO (not randomly)
- Enhances BO instead of enhancing random search
- Recently applied to MOO

**Examples:**
- **Multi-fidelity multi-objective BO:** Output space entropy search
- **BOHB for MOO:** Bayesian Optimization Hyperband
- Applications in MOHPO and neural network optimization

### 4.6 Further Issues

#### 4.6.1 Focusing Optimization Through User Preferences

**Growing Body of Literature:** Integrating decision maker (DM) preferences into MOO

**Timing Categories:**
- **A priori:** Before optimization
- **Progressive:** During optimization
- **A posteriori:** After optimization

**Majority of Literature:** A posteriori (find good approximation of entire Pareto front)

**Three Reasons for Earlier Preference Integration:**

1. **More Relevant Sample:**
   - Smaller set of most relevant alternatives
   - OR more fine-grained resolution of relevant Pareto frontier parts

2. **Faster Search:**
   - Focusing search on relevant part of space
   - Find solutions more quickly
   - Particularly important for expensive HPO

3. **Higher Dimensions:**
   - Increasing number of objectives makes complete Pareto approximation harder
   - More Pareto optimal solutions
   - Almost all solutions in random sample become non-dominated
   - DM preferences re-introduce necessary order relation

**Ways to Specify Preferences:**

**Explicit Specification:**
- **Reference points:** "Ideal" solution
- **Constraints:** Minimum acceptable qualities
- **Trade-offs:** Max willing to sacrifice in one criterion for unit improvement in another
- **Desirability functions:** Non-linear scaling of each objective to [0,1]

**Learned Preferences:**
- Ask DM to rank pairs or small sets of solutions
- Ask DM to pick most preferred solution from set
- **Advantage:** DM only compares solutions (more comfortable)
- Example: Interactive preference learning with Shapley values for explainability

**Knee Solutions:**
- Solutions that "stick out" on Pareto frontier
- Improving slightly in either objective leads to significant deterioration in other
- Can search for these without asking DM anything

#### 4.6.2 Noisy Environments

**Definition of Noise:**
If MOO problem is noisy, we only have measurements c̃ᵢ with:
```
c̃ᵢ = cᵢ + εᵢ  ∀i = 1,...,m
```
Where εᵢ is observational noise modeled as random variable

**Noise in MOHPO:**
- Generalization error estimates based on finite dataset
- Random sequence of data during training
- Stochastic optimizer used during training

**Challenges:**
- False performance comparisons
- Incorrectly inferred dominance relationships
- Dominated solutions classified as non-dominated
- Pareto-optimal solutions incorrectly discarded
- Over-optimistic estimate of Pareto frontier ("lucky" evaluations)

**Solutions:**

**1. Multiple Evaluations:**
- Evaluate each solution multiple times
- Optimize based on mean values
- Reduces standard error but computationally expensive

**2. From Evolutionary Computation Community:**
- Statistical tests
- Surrogate models
- Probabilistic dominance
- Ranking and selection techniques
- **Rolling Tide EA:** Alternates between sampling new candidates and refining archive (re-evaluating promising HPCs)

**3. For Bayesian Optimization:**
- Re-interpolation
- GP regression (rather than interpolation)
- Appropriate acquisition functions designed for noise

**Recent Surveys:**
- Methods under noise with provable convergence to local non-dominated set
- Broader scope surveys on multi-objective optimization under noise

#### 4.6.3 Realistic Evaluation of MOO Methods

**Challenge:** Understanding how solution will behave on unseen data after deployment

**Standard HPO:** 3-way split (train/validation/test) or nested resampling

**Quality Indicators (Section 3.3.2):** Useful for evaluating performance over whole objective space

**Decision Maker Interest:** Ultimately single solution for deployment

**Simple Approach (single split):**
1. DM looks at Pareto front computed on validation set
2. Choose configuration to use
3. Evaluate performance on test set

**Extension to Nested Resampling:**
- Multiple Pareto fronts generated (one per outer fold)
- **Challenge:** DM needs to make choices for each outer loop (impractical)
- Can become difficult for larger benchmark studies

**Automatic Approach:**
1. Evaluate each Pareto front on associated outer test set
2. Train each Pareto set candidate on joint training+validation set
3. Evaluate on test set
4. Results in unbiased Pareto front for each outer iteration
5. Calculate measures like hypervolume from outer results

**Open Challenge:** Proper MOHPO evaluation is understudied area for further research

**Drill-Down to Single Solution:** Would require DM choices per fold (not practical)

### 4.7 Relevant Benchmarks and Results

**Current State:**
- Most benchmarks conducted when proposing new optimizers
- **No extensive dedicated benchmark** covering all relevant MOHPO scenarios
- Some standardized HPO benchmark suites include multi-objective use cases

**Benchmark Suites:**

**HPOBench:**
- Supports selected multi-objective use cases
- Only multiple prediction performance objectives

**YAHPO Gym:**
- Includes multiple prediction performance objectives
- Also includes: computational efficiency, interpretability objectives
- Available for some scenarios

**EvoXBench:**
- Recently introduced
- Various EAs applied to NAS problems (image classification)
- Focus on evolutionary algorithms

**Table 2: Selection of Relevant Papers with MOHPO Benchmarks**

| Reference | Algorithms | # Scenarios | Note |
|-----------|-----------|-------------|------|
| [214] | Random, ParEGO, SMS-EGO, EHVI, Multi-EGO, Mixed integer ES | 25 | Surrogate benchmark, focus on mixed hierarchical spaces |
| [117] | ParEGO, SMS-EGO, NSGA-II, Latin Hypercube | 9 | Only binary classification |
| [118] | SMS-EGO, Rolling tide EA, Random | 9 | Only SVMs, emphasis on noisy scenarios |
| [113] | PESMO, EHVI, SMS-EGO, ParEGO, Sequential uncertainty reduction | 1 | - |
| [135] | EMOA with successive halving, Multi-objective BOHB, EHVI, Multi-objective BANANAS, BULK & CUT | 2 | - |
| [183] | NSGA-II, NSGA-III, MOEA/D, IBEA, HypE, RVEA | 18 | Focus on EAs for NAS, benchmark supports more instances |

**General Lessons and Best Practices:**

**1. Random Search is Competitive Baseline:**
- Preferred over grid search
- Grid search fails to provide good Pareto front approximation for MOHPO
- Random search shows good results
- Sometimes outperforms model-based methods (ParEGO, SMS-EGO, PESMO)
- On par with or outperformed by: Multi-EGO, ParEGO, mixed integer evolution strategy

**2. Different Horses for Different Courses:**
- Optimal algorithm choice heavily depends on MOHPO problem
- Examples from experiments:
  - ParEGO and SMS-EGO outperform NSGA-II variant and Latin Hypercube in some cases
  - Rolling tide EA competitive against SMS-EGO and outperforms random search in other cases
- Strong performance differences across benchmark problems:
  - Mixed integer evolution strategy: exceptional on some, average on others
  - Similar patterns observed for various EAs on NAS instances

**3. Sample Efficiency of BO is Big Factor:**
- Many MOHPO problems (like NAS) expensive to evaluate repeatedly
- BO methods appealing due to sample efficiency
- Good performance of BO methods (especially PESMO) on fewer iterations
- Particularly relevant for expensive tasks (e.g., neural network tuning)

**Current State Summary:**
- Benchmarks in [214] are first steps towards exhaustive MOHPO benchmark
- Better understanding of optimizers' behavior on variety of tasks needed
- **Plenty of work still needed** to give general recommendations on when to use which optimizer

**Transparency Recommendation:**
- Researchers should be transparent and exhaustive in experiment set-up
- Several works omit interesting baselines
- Don't detail exact configuration of algorithms

## 5. Objectives and Applications

**Organization:** Three aspects of ML model evaluation:

1. **Prediction Performance** (Section 5.1)
2. **Computational Efficiency** (Section 5.2)
3. **FAT-ML Related:** Fairness, Interpretability, Robustness, Sparseness (Sections 5.3-5.6)

**Visual Overview:** Figure 6 shows classification of application scenarios

### 5.1 Prediction Performance

**Challenge:** Which performance metric aligns best with ML task goals and costs?
- Not always readily apparent
- Especially when misprediction costs hard to quantify or unknown

**Scope:** Restricted to supervised ML (not unsupervised)

#### 5.1.1 ROC Analysis

**Context:** Binary classification models predict scores/probabilities
- Convert to classes by applying decision threshold
- Different thresholds = different trade-offs

**ROC Curve:**
- True Positive Rate (TPR) vs False Positive Rate (FPR)
- For different threshold values
- Improving one metric typically deteriorates the other
- **Fundamentally a multi-objective problem**

**Common Single-Objective Approaches:**
- F-Measure (aggregates confusion matrix elements)
- AUC (aggregates ROC curve)
- **Limitation:** Not all information preserved

**Example (Figure 7):**
- Two ROC curves with similar AUC
- But quite distinct shapes
- Single metric doesn't capture full picture

**Multi-Objective ROC Approach:**
- Combine information from each iteration into one **ROC front**
- Displays all Pareto-optimal trade-offs between TPR and FPR
- Preserves all relevant information
- Allows decisions according to user/case-specific preferences

**ROC Front Construction:**
- Each iteration contributes Pareto-optimal points
- Final combined model shows all non-dominated trade-offs
- Dashed line in Figure 7 shows resulting Pareto front

**Similar Approach for Regression:**
- **REC Curve:** Regression Error Characteristic Curve
- Trade-offs between error tolerance and percentage of points within tolerance

**Multi-Class ROC:**

**Challenges:**
- Number of misclassification errors grows quadratically with classes g
- g(g-1) misclassification rates to minimize simultaneously
- Increasingly high-dimensional surfaces

**Two Solution Approaches:**

**1. Single-Model Approaches:**
- Identify one classifier
- Find suitable trade-off on ROC surface at prediction time (when costs known)

**2. Multi-Model Approaches:**
- Produce pool of suitable classifiers available at prediction time
- Select appropriate one based on costs

**Optimal ROC Surface as Pareto Front:**
- Search space = space of classifiers
- Each classifier corresponds to ROC curve
- Classifier non-dominated if any part of ROC curve non-dominated
- Length of non-dominated part used as crowding distance

**AUC Generalization Challenge:**
- AUC principle doesn't carry over to multi-class
- Hand and Till (2001) attempted generalization to meaningful multi-class metric

**Multi-Class Decision Rule:**

For g-class setting, classifier outputs probability vector:
```
h(x) = [h(c₁|x), h(c₂|x), ..., h(c_g|x)]
```

Decision rule applies weight vector w = (w₁, w₂,..., w_g):
- Highest value chosen

**Cost of ROC Surface Generation:**
- Binary classification: vary decision threshold (cheap, post-hoc)
- Multi-class: evaluate large number of weight vectors (expensive)
- No longer trivial to obtain post-hoc for each configuration

**Applications:**

**Example Use Case (Chatelain et al. 2010):**
- Task: Digit recognition from handwritten mail document images
- Algorithm: SVM (effective but extremely sensitive to hyperparameters)
- Objectives: Two misclassification classes (digit vs non-digit)

**Hyperparameters Tuned:**
- C⁻ and C⁺: penalties for misclassifying respective classes
- γ: kernel parameter for RBF

**Method:**
- Use NSGA-II
- Evolve pool of non-dominated hyperparameter configurations
- Approximate Pareto optimal set

**Another Application (Horn and Bischl 2016):**
- Minimize false-negative rate and false-positive rate
- SVM on variety of binary classification tasks
- Different domains
- Use ParEGO and SMS-EGO

#### 5.1.2 Natural Language Processing

**Challenge:** Language and human speech are complex
- NLP tasks notoriously hard to evaluate
- Example: Natural Language Generation (NLG)

**NLG Quality Aspects:**
- Semantics
- Syntax
- Lexical overlap
- Fluency
- Coherency

**Problem:** No single metric correlates with all desirable aspects

**Example Application (Schmucker et al. 2021):**
- Task: Transformer-based language models
- Objectives: Perplexity and word error rate
- Method: Successive halving for multi-objective optimization

#### 5.1.3 Object Detection

**Task:** Given image, determine:
- Whether instances of given type exist
- Where they are located

**Aspects:**
- Regression: How close is proposed bounding box to ground truth?
- Classification: Are all objects identified correctly?

**Common Metrics:** Precision and recall
- **Problem:** Only focuses on one aspect
- Need to define when prediction is "true" or "false"

**Liddle et al. (2010) Proposal:**

Two specific aspects:
1. **Detection Rate (DR):**
   ```
   DR = # correctly located objects / # objects in image
   ```

2. **False Alarm Rate (FAR):**
   ```
   FAR = # falsely reported objects / # objects in image
   ```

**Additional Complexity:**
- Detection speed often crucial
- Makes single metric evaluation even more challenging

### 5.2 Computational Efficiency

**Context:**
- Technical constraints have always limited ML research
- With deep learning rise, efficiency has become important topic
- Focus: efficiency as desirable quality of **fitted model** (not optimization process)

**Scope:**
- Not addressing efficiency of optimization process itself
- Focusing on model efficiency for training or prediction
- Common scenario: resource limitations (e.g., memory constraints for deployment)

**Constrained vs Multi-Objective:**
- When hard resource limitations exist: formulate as constrained optimization
- Otherwise: treat efficiency as objective in MOHPO

**Feature Efficiency:** Addressed separately in Section 5.6

**Three Broad Approaches to Quantifying Efficiency:**

#### 5.2.1 Energy Consumption and Computational Complexity

**Limiting Computational Complexity:**
- Reduces number of operations
- Generally leads to energy-efficient models

**Measures for Deep Learning Computational Complexity:**

**1. Number of Active Nodes**
**2. Number of Active Connections**
**3. Number of Parameters**
**4. FLOPs (Floating-Point Operations)** - Ideal metric

**FLOPs as Standard:**
- Long used in ML, especially deep learning over last decade
- Examples: ResNet, ShuffleNet introductions
- Used to describe complexity/size of model

**Alternative: MAC Operations**
- Multiply-Accumulate operations
- Roughly linear relationship to FLOPs

**Simulation Approach:**
- Use simulator like **Aladdin**
- Designed to simulate NN energy consumption
- Requires C code describing NN operations
- Used for architecture evaluation

**Example Application (Wang et al. 2019):**

**Problem:** Complex deep learning models for computer vision
- Challenges for deployment on edge devices

**Approach:** Multi-objective to identify models:
- Highly accurate
- Minimal FLOPs

**Architecture: DenseNet**
- Several blocks of dense layers
- Connected via convolutional and pooling layers

**Search Space:**
- Number of layers in dense blocks
- Growth for dense blocks
- Deep learning hyperparameters (max epochs, learning rate)

**Method:**
- Initialize population
- Use Particle Swarm Optimization for Pareto front

**Results on CIFAR-10:**
- Some identified models outperform DenseNet-121
- Also outperform other DenseNet configurations
- While being less complex (smaller FLOPs)

#### 5.2.2 Model Size and Memory Consumption

**Relationship in Deep Learning:**
- Model size and efficiency often go hand in hand
- More parameters generally = more FLOPs

**Parameters:**
- Mostly weights
- Number can be straightforwardly derived in deep learning

**Example: MobileNets (Howard et al. 2017):**
- Specifically designed for efficient deployment on edge devices
- Introduce separable convolutions
- Reduce number of parameters needed
- For top-performing CNN

**Example Application (Loni et al. 2020):**

**Framework: DeepMaker**
- Identify efficient and accurate models for embedded devices

**Objectives:**
1. Classification accuracy
2. Model size (number of trainable weights)

**Findings:**
- High correlation between model size and prediction time

**Method:**
- NSGA-II for discrete hyperparameter search

**Hyperparameters:**
- Activation function
- Number of condense blocks
- Number of convolution layers per block
- Learning rate
- Kernel size
- Optimizer

**Multi-Fidelity Aspect:**
- During optimization: train 16 epochs for selection
- Final performance: report after 300 epochs

**Datasets Tested:**
- MNIST
- CIFAR-10
- CIFAR-100

#### 5.2.3 Prediction and Training Time

**Prediction Time:**
- Usually trained before deployment OR training not as time-critical
- **Primary interest:** Minimize prediction time

**Exception:** Some applications/deployment strategies require frequent retraining
- Training time becomes crucial factor

**Measurement Challenges:**
- Very hard to measure reliably
- Various differences in computing environment

**Correlation:**
- Prediction time may correlate strongly with energy efficiency metrics
- Some use FLOPs as proxy for inference latency

**Example Application (Hernández-Lobato et al. 2016):**

**Method: PESMO**
- Applied to several MOO problems

**Specific Application:**
- Fast and accurate NNs for image classification on MNIST

**Hyperparameters Tuned:**
- Hidden units per layer (50-300)
- Number of layers (1-3)
- Learning rate
- Dropout amount
- ℓ₁ and ℓ₂ regularization levels

**Objectives:**
1. Prediction error
2. Prediction time (measured as ratio)
   - Time to predict 10,000 images
   - Compared to fastest network in search space

**Comparison:**
- PESMO vs SMS-EGO, ParEGO, EHI, others
- PESMO shows superior hypervolume performance

**Follow-up:** Similar use case in their later paper

**Other Applications:**

**Dong et al. (2018):**
- Bi-objective NAS for image classification
- Objectives: classification accuracy + (FLOPs OR # parameters OR prediction time)

**Three Efficiency Metrics:**
- Some works optimize with two efficiency metrics + performance
- Examples: Lu et al. (2020a), Elsken et al. (2019), Chu et al. (2020)

**Chu et al. (2020) Framework:**
- Three objectives: PSNR or SSIM, FLOPs, # parameters
- Allows constraints on objectives
- Task: super-resolution domain

**HW-NAS Context:**
- Hardware-aware Neural Architecture Search
- Single-objective and constrained approaches also widespread
- Comprehensive survey: Benmeziane et al. (2021)

**Full Stack Context (Lokhmotov et al. 2018):**
- Tune MobileNets hyperparameters
- Objectives: test accuracy and prediction time
- Image classification task

**Comprehensive Overview:** Appendix A contains list of MOHPO applications with ≥1 efficiency objective

### 5.3 Fairness

**Context:** When algorithmic decisions impact human lives
- Important to avoid bias adversely affecting sub-groups
- Ethical and legal requirements

**Example Use Case:** Bank loan application decisions
- Want model to be performant
- Adhere to ethical/legal fairness requirements

**Metric Selection Challenge:**
- Choice of appropriate fairness metric is complex
- Depends on context of decision

**References:**
- Comprehensive surveys: Mehrabi et al. (2021), Pessach and Shmueli (2023)
- Benchmark: Friedler et al. (2019)

**Fairness Perspectives:**
- Causal fairness
- Individual fairness
- **Statistical group fairness** (most widely used in practice)

**Statistical Group Fairness:**
- Easy to implement
- Don't require access to causal data generating mechanism

**Common Group Fairness Metrics:**

Given:
- Protected attribute A (e.g., race)
- Outcome Y
- Binary predictor Yb

**1. Equalized Odds (Independence):**
```
Pr(Yb = 1|A = 1, Y = y) = Pr(Yb = 1|A = 0, Y = y), y ∈ {0,1}
```
- Requires equal true positive AND false positive rates between subpopulations
- More generally: independence on conditional Y

**2. Equality of Opportunity (Sufficiency):**
```
Pr(Yb = 1|A = 1, Y = 1) = Pr(Yb = 1|A = 0, Y = 1)
```
- Relaxation of equalized odds
- Only requires independence on event Y = 1 (advantageous outcome)
- E.g., getting approved for loan

**3. Calibration:**
For classifier h(x) yielding predicted probability:
```
∀a∈{0,1} ∀p∈[0,1] Pr(Yb = 1|A = a, h(x) = p) = p
```
- Requires calibrated probabilities in all groups
- Particularly desirable for classifiers in fairness context

**Applications and Example Use Case:**

**Existing Approaches:**
- Decrease discrepancies in classifier performance between sub-groups/individuals
- Achieved via:
  1. **Preprocessing data** (e.g., reweighing)
  2. **Imposing fairness constraints during model fit**
  3. **Post-processing model predictions** (e.g., equalized odds adjustment)

**Hyperparameter Tuning:**
- These methods often have hyperparameters
- Can be tuned to emphasize fairness during training

**MOHPO Approaches:**

**Pfisterer et al. (2019):**
- Multi-objective BO
- Jointly optimize fairness criteria and prediction accuracy

**[211] Constrained BO:**
- Fairness metric constrained to small deviation from optimal
- While predictive accuracy optimized

**Pelegrina et al. (2020):**
- MOEA to simultaneously optimize fairness metrics and performance
- Find fair PCA

**Martinez et al. (2020):**
- Each sensitive group risk as separate objective
- Leads to MOHPO for choosing classifier

**Important Observation:**
- Fairness heavily influenced by:
  - Parameters of debiasing method
  - Choice of ML algorithm
  - Hyperparameters of algorithm

**Example Dataset: COMPAS**
- Correctional Offender Management Profiling for Alternative Sanctions
- Goal: Predict risk of criminal defendant re-offending
- Objectives:
  1. Accurate prediction
  2. Not biased towards any race

**Fairness Metric Used:**
```
τ_FPR = FPR_S0 / FPR_S1
```
- Optimal when τ_FPR = 1
- FPR_S is false positive rate on group S
- S0 = advantaged group, S1 = disadvantaged group

**Example Results (Figure 8):**

**Model:** Random Forest

**Debiasing Techniques:**
1. Reweighing
2. Equalized Odds
3. Nonlinear Programs (NLP)

**Interpolation:** Between no debiasing and full debiasing

**Results:**
- Different strategies and strengths lead to different trade-offs
- Forms multi-objective optimization problem
- Trade-off between accuracy and fairness (τ_FPR)

**Observed Trade-offs:**
- RF baseline: ~0.64 accuracy, ~0.7 τ_FPR
- RF-Equalized Odds: Can reach ~0.64 accuracy, ~0.8 τ_FPR
- RF-NLP: Various points, some reaching ~0.9 τ_FPR at ~0.60 accuracy
- RF-Reweighing: Similar patterns

### 5.4 Interpretability

**Goal:** Make ML model decisions more transparent
- Provide human-understandable explanations

**Range of Methods:**
- Fully interpretable model classes
- Post-hoc interpretation techniques

**Interpretable Model Classes:**
- Required for regulatory constraints (e.g., banking sector)
- Can be thought of as constraint on search space for model selection

**Post-hoc Interpretation:**
- Understand functional relationships between features and target
- Understand single decisions made by algorithm

**Challenge:**
- Can produce misleading results if:
  - Model too complex
  - Explainability technique unreliable (e.g., using additional features)

**Quantifying Interpretability:**
- Determining complexity of predictive decisions not straightforward
- Terms highly subjective: interpretability, explainability, complexity

**Popular Alternative: Sparseness**
- Number of features
- Easily quantifiable proxy for complexity/interpretability
- Explored further in Section 5.6

**Focus:** Current approaches on tabular data

**Two Metrics (Molnar et al. 2019):**

**1. Main Effect Complexity (MEC):**

Measures average shape complexity of ALE main effects:

```
MEC = 1/Σⱼ₌₁ᵖ Vⱼ Σⱼ₌₁ᵖ Vⱼ · MECⱼ
```

Where:
- Vⱼ = 1/n Σᵢ₌₁ⁿ (f_j,ALE(x⁽ⁱ⁾))²
- MECⱼ = K + Σₖ₌₁ᴷ I_β₁,ₖ>0 - 1
- K = number of linear segments for approximation
- β₁,ₖ = slope of k-th segment
- p = number of features
- n = number of samples

**Interpretation:**
- Number of parameters needed to approximate ALE curve with linear segments
- Higher value = more complex main effects

**2. Interaction Strength (IAS):**

Fraction of variance not explained by main effects:

```
IAS = E(L(f, f_ALE1st)) / E(L(f, f₀)) ≥ 0
```

Where:
- f = prediction function
- f_ALE1st = sum of first order ALE effects
- f₀ = mean of predictions
- L = loss function

**Interpretation:**
- Measures impact of interaction effects
- Most interpretability techniques use linear relationships
- Higher interaction = less interpretable via standard methods

**Applications and Example Use Case:**

**Molnar et al. (2019) Application:**

**Task:** Predict quality of white wines (scale 0-10)

**Search Space:**
- Several models: SVM, gradient boosted trees, random forest, others
- Number of tunable hyperparameters

**Four Objectives:**
1. Cross-validated mean absolute error
2. Number of features used
3. Main effect complexity
4. Interaction strength

**Method:**
- ParEGO for optimization
- 500 iterations

**Results:**
- Find good trade-offs between objectives
- Identify desirable model configurations

**Comparison with Single-Criterion Tuning:**
- Multi-objective approach reveals additional hyperparameter settings
- These additional configurations:
  - Comparable or better performance
  - Require fewer features
  - More stable feature selection

**Another Application (Carmichael et al. 2021):**
- MOHPO for deep learning architectures
- Trade-offs between accuracy and introspectability
- Image classification tasks: ImageNet-16-120, CIFAR-10, MNIST

**Comprehensive Reviews:**
- Additional examples in referenced surveys
- Specifically for Evolutionary Computation methods: Xue et al. (2015)

### 5.5 Robustness

**Context:**
- Important requirement for accountability in ML
- Only loosely defined
- Sometimes mixed with robustness of optimization procedure

**Focus:** Robustness of **fitted model**
- Susceptibility to data shifts in prediction step
- Mostly consider classification setting
- Indicate differences for regression where appropriate

**Lack of Standards:**
- No tried and proven metrics to assess robustness
- No proper taxonomy

**Taxonomy (Taori et al. 2020):** Overview of possible changes to input data

**Three Types of Changes:**

#### 5.5.1 Robustness Metrics and Approaches

**1. Distribution Shift**

**Definition:** Changes of:
- Marginal distribution of target, OR
- Distribution of features (conditional on target)
- On macro level

**Effective Robustness Metric:**
```
ρ(f) = acc₂(f) - β(acc₁(f))
```

Where:
- acc₁, acc₂ = accuracies pre and post distribution shift
- β(x) = baseline accuracy given pre-shift accuracy x
- Represents adjustment for expected performance drop without robustness intervention

**How to Compute β(x):**
- Information on expected drop for models without robustness intervention
- Details in Taori et al. (2020)

**2. Adversarial Examples**

**Context:**
- Substantial interest from visual deep learning community
- Model susceptible to adversarial attacks not trustworthy
- Well-known in image data
- Also shown in: text, sequence, tabular data

**Adversarial Accuracy (AAcc):**

For perturbations in ε-ball around each point:

```
AAcc = E[𝟙(f(x*) = cₓ)]

where x* = arg max_{d(x',x)≤ε} L(x', cₓ)
```

- cₓ = respective class label
- Measures percentage correctly classified after adversarial attack

**In Practice:**
- Conduct adversarial attack
- Calculate accuracy from new predictions

**Adversarial Frequency (AF):**

Accuracy on worst-case input in ℓₚ ε-ball:

```
AF = P(ρ(f, x*) ≤ ε)
```

Where:
- ρ(f, x*) = minimum distance ε̂ such that ∃x with d(x, x*) ≤ ε̂ : f(x) ≠ f(x*)

**Adversarial Severity (AS):**

Expected minimum distance to adversarial example:

```
AS = E[ρ(f, x*) | ρ(f, x*) ≤ ε]
```

**Comparison:**
- Bastani et al. (2016) deem frequency more important
- Significant work on minimum distance to adversarial example (especially for NNs)

**3. Perturbations**

**Relationship to Adversarial Examples:**
- Often strongly linked in deep learning
- Logical: adversarial examples often sought within ε-ball

**Independence:**
- Laugros et al. (2019) argue and show:
- Robustness to common perturbations
- Adversarial robustness
- Are **independent attributes** of model

**Common Perturbation: Gaussian Noise**

Add random Gaussian noise N(0, ε) to input data X:
- ε generally 0.001-0.01 times feature range

**Simple Robustness Measure (Pfisterer et al. 2019):**
```
|L(X, Y) - L(X + N(0, ε), Y)|
```

Where L is relevant loss measure

**Can apply to other perturbation types**

**Relationship (Gilmer et al. 2019):**
- For image data at least:
- Susceptibility to adversarial examples
- Susceptibility to perturbations
- Are two symptoms of same underlying problem
- Optimizing for robustness against adversarial attacks and perturbations should go hand-in-hand

**Machine Translation Perturbations (Niu et al. 2020):**
- Comprehensive overview of perturbations
- Mainly: synthetic misspellings, letter case changing
- Define variety of suitable robustness measures

#### 5.5.2 Robustness and Uncertainty Quantification

**Connection:**
- Uncertainty quantification heavily researched (especially for deep learning)
- Often mentioned with robustness

**Uncertainty Quantification in MOHPO:**
- Under-explored research direction
- Few exceptions (e.g., [263])
- Integrating respective metric into MOHPO even less explored

**Relationships:**

**Better Calibrated Models:**
- Tend to suffer less from adversarial examples

**Robustness to Domain Shift:**
- Closely linked to predictive performance on out-of-distribution samples
- Used as measure of predictive uncertainty

**Aleatoric Uncertainty:**
- Kendall and Gal (2017) show beneficial effect on robustness
- Particularly in context of noise (perturbations)

#### 5.5.3 Application and Example Use Case

**Guo et al. (2020) Application:**

**Goal:** Identify network architectures robust to adversarial attacks

**Method:** One-shot NAS
1. Initial training of supernet
2. Draw architectures through random search
3. Fine-tune and evaluate

**Analysis:**
- Examine different architectures
- Compare: robustness to adversarial examples AND model size
- Identify suitable models

**Figure 9 Illustration:**
- Several architectures plotted
- X-axis: Number of Parameters (M)
- Y-axis: Adversarial Accuracy (%)

**Architecture Families:**
- RobNet family (clustered, better performance)
- ResNet-18
- Wide-ResNet-28-10
- DenseNet-121
- Others

**Trade-off:**
- Better direction: Lower parameters, higher adversarial accuracy
- Shows clear Pareto frontier of non-dominated architectures

### 5.6 Sparseness via Feature Selection

**Motivation for Sparse Models:**

**Problem:** High-dimensional tasks with many features

**Why Prefer Sparseness:**
1. **Interpretability:** Too many features make relationship hard to interpret
2. **Cost:** Model fitting/inference increases (storage, computation, data acquisition)
3. **Performance:** May suffer from curse of dimensionality

**Feature Selection = Inherently MOO:**
- Model performance vs sparsity tend to conflict
- Lower features often = lower performance (reduced information)
- BUT: Specific desirable quantity exists (probably not maximum)

**Three Categories of Feature Selection:**

**1. Embedded Methods:**
- Perform feature selection as part of model fitting
- Examples:
  - Empirical risk minimization with L0 or L1 regularization
  - Trees/tree-based methods (greedy split selection)
- **Drawback:** Specific to model in use

**2. Filter Methods:**
- Use proxies to rank feature subsets
- Independent of learning algorithm
- Before model training
- Single measures:
  - Information theoretic measures
  - Correlation measures
  - Distance measures
  - Consistency measures

**3. Wrapper Methods:**
- Search over space of selected features
- Optimize model performance directly
- **Advantages:**
  - Take learning algorithm performance into account
  - Often yield better performance than filter methods
- **Disadvantages:**
  - Computationally expensive
  - Model evaluations noisy and expensive
- **Most amenable to MOO extensions:** Can use general optimization algorithms

#### 5.6.1 Sparseness Metrics

**1. (Weighted) Number of Features:**
```
Σⱼ₌₁ᵖ wⱼsⱼ
```

Where:
- sⱼ = indicator if feature j included
- wⱼ = weight (can incorporate different costs)

**Simple Approach:** Minimize number of features

**Weighted Version:** Different features have different:
- Acquisition costs
- Importance
- Application-specific weights

**2. Stability of Feature Selection:**

**Context:** Applications like omics data analysis in bioinformatics
- Main goal: identify important features
- Example: Important genes for later laboratory examination

**Measurement:**
- Compare sets of selected features from different resampling iterations
- Higher stability = more reliable feature identification

**Usefulness:** Helpful when identifying important features is primary goal

**Joint Search Space:**

Feature selection implies search over binary space {0,1}ᵖ:
- sᵢ = I[feature i is included]

**Joint Hyperparameter and Feature Optimization:**
```
{0,1}ᵖ × Λ̃
```

**Challenges:**
- Dimensionality grows exponentially with number of features
- Complex interactions between features and hyperparameters
- Expensive evaluations (especially wrapper methods)
- Need for efficient search methods

**Search Methods:**
- **Evolutionary algorithms:** Widely used due to ability to handle complex search spaces
- **Bayesian optimization:** More efficient alternative (recent works)

#### 5.6.2 Applications and Example Use Case

**Bommert et al. (2017) Application:**

**Problem:** Feature selection as MOHPO with three objectives:
1. Predictive performance
2. Number of features selected
3. Stability of feature selection

**Comprehensive Comparison:**
- Comprehensive comparison of stability measures
- Tune hyperparameters of ML pipelines:
  - Feature filter + classification learner

**Method:**
- Random search for tuning
- Identify desirable trade-offs

**All Pipelines Tuned w.r.t. Three Criteria**

**Post-Processing:**
1. Discard configurations not within 5% tolerance of best predictive performance

**Results vs Single-Criterion Tuning:**

Multi-objective approach additionally reveals hyperparameter settings that:
- Comparable or better performance
- Require **fewer features**
- More **stable feature selection**

**Comprehensive Reviews:**
- Al-Tashi et al. (2020): General comprehensive review
- Xue et al. (2015): Specifically for Evolutionary Computation methods
  - Includes works reducing feature selection to single-objective
  - Also several multi-objective approaches

### 5.7 MOHPO Applications in Industry

**Challenge:** Hard to gauge deployment in industry
- Companies don't necessarily advertise success

**Commercial AutoML Tools:**
- Several libraries support multi-objective optimization (e.g., Syne Tune, Optuna)
- Commercial AutoML tools have NOT widely adopted such methods

**Applicability Depends on Metrics:**
- ML community still struggling with:
  - Taxonomies for interpretability, robustness, fairness
  - Quantifying these notions
- **Without established metrics:** Productive application limited

**More Established: Performance + Efficiency**
- Joint optimization of pipelines/architectures for performance and efficiency
- More established within HW-NAS
- Several publications emphasize importance for industrial applications
- Intuitively makes sense:
  - Embedded devices relevant
  - Devices with compute/energy consumption constraints

**Company Involvement:**
- Some companies involved in MOHPO research
- Example: Amazon introducing new methods/algorithms
- BUT: Not on Amazon's/industrial datasets
- Mainly on typical benchmark datasets

**Diversity in Applications:**
- Especially for performance and efficiency (Section 5.2)
- Speaks to relevance for industry
- Some publications present industrial applications directly

**Example Industrial Applications:**

**Chandrashekaran and Lane (2016):**
- Optimize hyperparameters for large vocabulary continuous speech recognition
- Multiple relevant performance metrics
- Memory footprint
- Results on custom evaluation dataset from LGE Electronics

**Quadrana et al. (2022):**
- Jointly optimize relevant performance metrics
- Context: Behavioral song embeddings
- Two datasets examined:
  1. Standard dataset
  2. **Large-scale proprietary dataset:**
     - Anonymized streaming listening sequences
     - Playlists from Apple

**Summary:**
- Industrial applications exist
- Growing interest
- But widespread adoption still limited by:
  - Lack of standardized metrics (for FAT-ML objectives)
  - Need for more established commercial tools

## 6. Discussion and Open Challenges

**Summary:**
- Presented comprehensive overview of MOHPO concepts, methods, applications
- Evident merit in formulating ML problems in multi-objective manner
- Many application examples support this

**State of ML:**
- Single-objective ML tasks (pure prediction performance) no longer state-of-the-art
- Models must meet standards for secondary goals

**Advantages of MOO for HPO:**
- Provides suite of Pareto optimal trade-offs
- Enables practitioners to select suitable model
- Allows meaningful and precise metrics (no forced aggregation like AUC, F-score)

**Caution on Aggregate Measures:**
- Should be used with care in MOO context
- Objectives should correspond to real-world objectives as closely as possible

**Software Availability:**
- Topic not fully established
- Limited available software specifically for MOHPO
- BUT: Good implementations exist for several standard methods

**Open Challenges and Future Directions:**

### User Preferences

**Challenge:** Integrating preferences meaningfully
- Either a priori or during optimization
- Could help efficiency and transparency of MOHPO

**Future Research:**
- Obtaining (noisy) labels or user preferences during MOHPO process
- Utilizing them effectively
- Promising avenues:
  - Preference learning
  - Methods to integrate similar information

**Problem:**
- Underlying preferences may not be quantifiable
- Cannot be expressed through metrics in Section 5
- Need alternative approaches

### Beyond Supervised Learning

**Focus of This Work:** Supervised learning

**Unsupervised Methods:**
- Clustering, anomaly detection also depend on hyperparameters
- **Challenges (similar to single-objective HPO):**
  - Difficulty of performance evaluation
  - Lack of standardized metrics
  - Custom, use-case-specific measures
  - Sometimes visual inspection of results

**Still Applicable:**
- Other objectives like efficiency can be evaluated
- May be included in (MO)HPO of unsupervised methods

### Multiple Black-Box Functions

**Opportunity:** Multiple black-box functions with:
- Individual characteristics (cost, noise, etc.)
- Possible dependencies

**Different Evaluation Costs:**
- Can be exploited to some extent
- Hybrid methodologies: combine model-based and evolutionary optimization
- Hybrid algorithms: Open challenge for MOHPO

**Decoupled Evaluations:**
- Introduced for multi-objective BO
- Application to MOHPO: Open topic

**Dependencies Between Black-Boxes:**
- If correlation suggested by expert knowledge: could infer value of one black-box from another
- Properly exploited: could avoid expensive evaluation
- Especially if significant resource loss

### Constraints in MOHPO

**Motivated in Section 5:**
- General MOHPO problem can include constraints
- **Rarely combined in practice**

**Open Question:**
- Topic of treating model quality as objective vs constraint
- Hardly ever discussed

**Complexity:**
- Constraints similar to objectives:
  - May have different characteristics
  - May be black-box
- Presents directions for future work

**Hidden Constraints and Objectives:**
- Sometimes hidden in MOHPO
- Along with trustworthiness: reason for interactive methods being interesting

### Catching Up to Single-Objective HPO

**Perspective:** Many open challenges are about catching up
- MOHPO only recently of increased interest
- Lacking behind single-objective HPO in several respects
- Some future work = implementing for MOHPO what exists for single-objective

**Examples:**
- Certain algorithms only recently implemented for MOHPO (multi-fidelity methods)
- Used productively in single-objective HPO for years

**Unique HPO Challenges:**
- High dimensional, mixed, hierarchical search spaces
- Discussed for single-objective HPO:
  - Special kernel functions
  - Various surrogate models for mixed-hierarchical hyperparameters

**General Challenges:**
- Early stopping of optimization
- Noisy evaluations
- Need consideration for multi-objective case

### Benchmarks and Evaluation

**Critical Need:** Proper and extensive benchmarks for MOHPO field
- Could shed light on strengths/weaknesses of different methods
- On variety of MOHPO tasks

**Transparency Encouragement:**
- Researchers should be exhaustive in experiment set-up/presentation
- Several works omit interesting baselines
- Don't detail exact configuration of algorithms

**Evaluation Challenge:**
- Proper evaluation of MOHPO methods still open topic for research

**Discrepancy in Usage:**
- ROC analysis and multi-objective feature selection: well-established, well-researched
- MOHPO with efficiency objectives: grown rapidly only in past few years
  - With ascent of deep learning and HW-NAS

**FAT-ML Related Objectives:**
- With recent trends to integrate FAT-ML standards into ML process
- MOHPO with interpretability and fairness becoming relevant
- **BUT:** Few works published
- Community still struggles to establish metrics

**MOHPO Advantage for Transparency:**
- Already improves transparency vs single-objective HPO
- No reduction to single metric
- Result = collection of trade-offs (not only single hyperparameter configuration)

**Research Needed:**
- Exploration of appropriate metrics for interpretability/fairness
- Not necessarily open topic for **methodological** MOHPO research
- But for applied ML in general

**Omitted Topics:**
- Other desirable model attributes exist
- Currently no appropriate ways to translate to metrics
- Example: Amount of labeled data needed (important in semi-supervised learning)

### Constrained Optimization

**Complexity:** Comes with own set of challenges
- Orthogonal to multi-objective aspect
- Therefore excluded from this work
- Only mentioned in crucial parts
- Helpful references provided when appropriate

## Conclusions

**Main Contribution:**
- Comprehensive review on MOHPO for ML practitioners
- Explains most popular algorithms
- Discusses main challenges and opportunities
- Surveys existing applications

**Scope:** Restricted to supervised ML

**Key Takeaway:**
- Substantial merit in MOHPO approach
- Single-objective optimization of ML pipelines no longer sufficient for many applications
- Models must meet standards w.r.t. secondary goals

**Value Proposition:**
- Provides set of Pareto-optimal solutions
- Different trade-offs analyzed by domain experts post-hoc
- Informed decisions without requiring a priori specification
- More meaningful and precise metric selection

**Open Challenges:**
- Proper extensive benchmarks needed
- Better evaluation methods required
- Integration of user preferences
- Extension beyond supervised learning
- Standardization of FAT-ML metrics

## Software and Code Availability

Referenced throughout paper:
- **PlatEMO (MATLAB)**, **pymoo (Python)**: EMOAs
- **Dragonfly, HyperMapper, OpenBox, Ax, trieste, BoTorch, GPflowOpt, mlr3mbo, Optuna**: Multi-objective BO
- **mle-hyperopt**: MOHPO via NSGA-II
- **HPOBench, YAHPO Gym, EvoXBench**: Benchmark suites

---

**Page Count:** 48 pages
**References:** 272 citations
**Comprehensive appendix:** Table of MOHPO applications with efficiency objectives
