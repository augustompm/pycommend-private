# Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach

**Authors:** Wanyi Liu, Long Chen, Zhenzhou Tang (Senior Member, IEEE)

**Institutions:**
- Wenzhou Key Laboratory for Intelligent Networking, Wenzhou University, Wenzhou, China, 325035
- Key Laboratory of Intelligent Education Technology and Application of Zhejiang Province, Zhejiang Normal University, Jinhua, China, 321000

**Published:** arXiv:2410.02301v1 [cs.NE] 3 Oct 2024

**Funding:** Natural Science Foundation of Zhejiang Province, China, Grant LZ20F010008

---

## Abstract

Multi-objective optimization is a common problem in practical applications, and multi-objective evolutionary algorithm (MOEA) is considered as one of the effective methods to solve these problems. However, their randomness sometimes prevents algorithms from rapidly converging to global optimization, and the design of their genetic operators often requires complicated manual tuning.

To overcome this challenge, this study proposes a **new framework that combines a large language model (LLM) with traditional evolutionary algorithms** to enhance the algorithm's search capability and generalization performance.

### Framework Features

**Adaptive Mechanism:**
- Employs auxiliary evaluation function
- Automated prompt construction
- Flexibly adjusts LLM utilization
- Generates high-quality solutions refined through genetic operators

**Hybrid Mechanism:**
- Minimizes interaction costs with LLM
- Maximizes benefits of LLM's language understanding and generation capabilities
- Provides more intelligent and efficient search strategies

### Key Results

- LLM-assisted evolutionary search **significantly accelerates population convergence**
- **Outperforms state-of-the-art evolutionary algorithms** in competition
- Reveals potential advantages of utilizing pre-trained LLMs in MOEA design
- Offers a promising new approach to solving multi-objective optimization problems

---

## 1. Introduction

### Multi-objective Optimization Problems

The multi-objective optimization problem refers to optimization problems which need to consider several interrelated and mutually restricted optimization objectives in the decision-making process. Multi-objective optimization problems are very common in the real world. The key challenge is how to optimize multiple interdependent objectives to efficiently find the feasible Pareto optimal solution.

### Challenges with Traditional MOEAs

**Limitations:**
- Necessity for formulating sophisticated operators
- Refinement of solution convergence and distribution
- Labor-intensive challenge
- Dependency on expert knowledge

**Critical Needs:**
- Enhancing convergence rate
- Improving solution diversity
- Reducing dependency on expert knowledge

### LLM Integration with MOEAs

Recent years have witnessed increasing attention on integrating Large Language Models (LLMs) with the MOEA framework, aiming to enhance performance and adaptability through advanced natural language understanding and generation capabilities.

#### Two Principal Application Patterns

**1. LLMs as Black-Box Search Operators**
- Leverage representation and generation capacities
- Efficiently navigate solution space
- Iteratively generate successive solution iterations
- No need for explicit problem encoding
- Use natural language to articulate desired characteristics
- Simplifies process compared to programming or mathematical descriptions
- Minimizes necessity for expert intervention

**2. LLMs for Operator Design and Selection**
- Leverage extensive prior knowledge
- Design high-performance operators
- Select appropriate operators for algorithm evolution
- Enhance search capability and convergence speed
- Utilize code comprehension, representation, and generation capabilities
- Significant advancement at the algorithmic level

---

## 2. Related Works

### A. MOEA using the LLM

Over the past two years, LLMs have developed rapidly with their scale growing exponentially. Thanks to massive training datasets, these models have become increasingly powerful.

**LLM Capabilities:**
- Exceptional performance in natural language processing
- Strong data analysis abilities
- Progress in scientific research
- Personalized services
- Remarkable reasoning and prediction capabilities
- Superior generalization abilities

### Current Research Advancements

#### Approach 1: LLMs as Black-Box Operators

**Single-Objective Problems:**

1. **OPRO (Optimizer Prompting)** - Yang et al.
   - Leverages LLMs as optimizers for single-objective problems
   - Particularly suited for cases where gradients are unavailable
   - Natural language describes optimization problem and serves as meta-prompts
   - LLM uses previously generated solutions as prompts to continually generate new solutions

2. **LMEA (LLM-driven Evolutionary Algorithm)** - Liu et al.
   - LLMs perform crossover and mutation operations
   - Prompts constructed in each generation to assist LLM in selecting parent solutions

**Multi-Objective Problems:**

3. **Decomposition-based MOEA Framework** - Liu et al.
   - LLM acts as black-box optimization operator
   - Through prompt engineering and contextual learning
   - LLM generates new offspring solutions for each subproblem
   - Designed explicit white-box linear operator to approximate LLM results (mitigates high interaction costs)

4. **QDAIF (QD-based AI Feedback)** - Bradley et al.
   - Applied to Quality-Diversity (QD) search
   - Can be applied to complex qualitative domains
   - LLMs evaluate both quality and diversity of solutions
   - Evolutionary algorithm maintains solution archive
   - Higher-quality and more diverse solutions introduced into solution set

### Summary of LLM-Assisted MOEA Work

| Index | References | Type | Methodology |
|-------|-----------|------|-------------|
| 1 | [18] | N/A, Evaluation | AS-LLM |
| 2 | [13] | Single-objective | OPRO |
| 3 | [14] | Single-objective | LMEA |
| 4 | [15] | Multi-objective, MOEA/D | MOEA/D-DE, MOEA/D-LO |
| 5 | [17] | Multi-objective, Evaluation | Decomposition-based MOEA QDAIF |

### Current Limitations

**Challenges:**
- Application of LLMs to optimization algorithm design remains relatively new
- Still in early exploratory stage
- LLMs applied primarily as black-box solvers
- Each optimization step requires real-time, resource-intensive online interaction
- Significant consumption of computational resources and time
- Applying LLMs to complex real-world problems presents numerous challenges
- Relatively narrow scope of evaluation studies
- Not comprehensively validated overall capabilities in optimization domain

---

## 3. Problem Formulation

For the sake of generality, this paper considers the following form of multi-objective minimization problem:

```
min. F(x) = (f₁(x), ..., fₘ(x))ᵀ
s.t. x ∈ Ω
```

**Where:**
- `x = {x₁, ..., xₙ}ᵀ` is the decision variable
- `Ω` is the domain of the decision variable (usually n-dimensional Euclidean space)
- `F(x)` is an objective function vector containing m objective functions
- Objective functions represent different performance metrics or costs

**Goal:**
Find a set of solutions that cannot be improved simultaneously on all objective functions (Pareto optimal solution set). These solutions represent an optimal trade-off between different goals.

### Dominance Relation

A solution xᵢ is said to **dominate** another solution xⱼ if the following two conditions are simultaneously satisfied:

```
fₘ(xᵢ) ≤ fₘ(xⱼ), ∀m ∈ {1, 2, ..., M}
fₙ(xᵢ) < fₙ(xⱼ), ∃n ∈ {1, 2, ..., M}
```

**Notation:** xᵢ ≺ xⱼ

### Pareto Optimality

**Pareto-optimal solution x*:**
Not dominated by any other solution x ∈ Ω (the feasible solution set)

**Pareto set:**
The set of all Pareto-optimal solutions

**Pareto front (PF):**
The set of objective function values corresponding to all Pareto-optimal solutions. In graphical representation, typically a curve or polyhedron that illustrates trade-off relationships between different objectives.

**Optimization Goal:**
Approximate the entire Pareto frontier as uniformly as possible

---

## 4. LLM for MOEA

### A. Framework

This framework is a **general-purpose framework** that can be integrated with any other multi-objective evolutionary algorithm to enhance convergence and distribution. The following explanation uses **NSGA-II** as an example.

#### Initialization Stage

**Initial Population Generation:**
```
Pₜ = {x₁, x₂, ..., xₙ}
```

**Where:**
- N denotes population size
- xₖ represents the kth individual in the population
- Each individual xₖ = {xₖ₁, xₖ₂, ..., xₖd} is a d-dimensional decision vector

**Random Generation Method:**
```
xₖᵢ = lᵢ + (uᵢ - lᵢ) × rₖᵢ, ∀i = {1, 2, ..., d}
```

**Where:**
- xₖᵢ represents the i-th decision variable of individual xₖ
- lᵢ and uᵢ are lower and upper bounds of xₖᵢ
- rₖᵢ is a random number following uniform distribution in (0, 1)

#### Evaluation

Each individual in the population is evaluated:

```
F(xₖ) = {f₁(xₖ), f₂(xₖ), ..., fₘ(xₖ)}
```

**Where:**
- M denotes number of optimization objectives
- fₘ(x) is the evaluation function

#### Non-dominated Sorting

Partition population P into multiple layers F = {F₁, F₂, ..., Fₖ} based on dominance relationships:

```
F₁ = {xᵢ ∈ P | ∀xⱼ ∈ P, xⱼ ⊀ xᵢ}
Fₙ = {xᵢ ∈ Pₙ₋₁ | ∀xⱼ ∈ Pₙ₋₁, xⱼ ⊀ xᵢ}
Pₙ₋₁ = P \ ∪ₖ₌₁ⁿ⁻¹ Fₖ, n = {2, 3, ..., c}
```

### Auxiliary Evaluation Function

Determines whether to employ LLM to generate new individuals. For NSGA-II, the auxiliary function is:

```
Dᵥ = {dᵢ ∈ D | dᵢ is finite}

d̄ = {
  ∞,                    if Dᵥ = ∅
  1/|Dᵥ| Σ(dᵢ∈Dᵥ) dᵢ,   otherwise
}

S = -d̄ + 1/|B| Σ(fᵢ∈B) fᵢ
```

**Where:**
- D = {D₁, D₂, ..., Dₙ}
- Dᵢ denotes crowding distance of individual xᵢ
- B = {B₁, B₂, ..., Bₙ} where Bᵢ is the index of each individual within its rank Fᵢ

#### Crowding Distance Calculation

```
Dᵢ = Σ(m=1 to M) [fₘ(xᵢ₊₁) - fₘ(xᵢ₋₁)] / [fₘᵐᵃˣ - fₘᵐⁱⁿ]
```

**Where:**
- fₘ(xᵢ₊₁) and fₘ(xᵢ₋₁) represent objective function values of neighboring individuals
- fₘᵐᵃˣ and fₘᵐⁱⁿ are maximum and minimum values of m-th objective function

### Decision Mechanism

Compare auxiliary function scores to decide whether to utilize LLM:

```
Method = {
  NSGA-II,  if Sₜ - Sₜ₋₁ < δ
  LLM,      if Sₜ - Sₜ₋₁ ≥ δ
}
```

**Where:**
- δ is a pre-defined decision threshold

#### Generating Offspring Using LLM

1. **Tournament Selection:** Select N individuals from current population forming set Mₜ
2. **Elite Identification:** Identify high-frequency individuals in Mₜ (elite solutions)
3. **Ranking:** Rank by occurrence frequency (high to low)
4. **Prompt Construction:** Select top l individuals to form set pₜ
5. **LLM Generation:** Input prompts to LLM, generate offspring oₜ
6. **Replacement:** Replace recurring individuals in Mₜ with oₜ
7. **Breeding:** Use NSGA-II breeding strategies to generate offspring Qₜ
8. **Environmental Selection:** Combine Pₜ with Qₜ, use NSGA-II environmental selection to get Pₜ₊₁

#### Generating Offspring without LLM

If LLM is not employed, offspring Qₜ = {x₁, ..., xₙ} are directly generated through NSGA-II reproductive strategy. Remaining steps are the same.

---

### Algorithm 1: Algorithm Framework

**Input:**
- Maximum number of evaluations: Nₘₐₓ
- Population size: N
- Number of parents for LLM: l
- Number of new individuals generated by LLM: s
- Decision threshold: δ

**Output:** Next generation population Pₜ₊₁

```
Generate initial population Pₜ ← {x₁, ..., xₙ} randomly
Pₜ₊₁ ← ∅, Sₜ₋₁ ← 0, t ← 0

while t < Nₘₐₓ do:
    # Assessment
    Get score Sₜ by auxiliary function

    if Sₜ - Sₜ₋₁ > δ then:
        # Selection
        Get elitist solutions pₜ ← {x₁, ..., xₗ}
        through MOEA's selection algorithm

        # Prompt engineering
        Generate textual Prompt for LLM
        given subset pₜ

        # Direction
        LLM generates new individuals
        oₜ ← {x₁, ..., xₛ} with Prompt

        # Reproduction
        Replace solutions in pₜ with oₜ
        Use MOEA's propagation algorithm
        to generate offspring Qₜ ← {x₁, ..., xₙ}
    else:
        # Evolution
        Use MOEA's propagation algorithm in Pₜ
        to generate offspring Qₜ ← {x₁, ..., xₙ}

    # Update
    Merged population Uₜ ← {Pₜ ∪ Qₜ}
    Next generation Pₜ₊₁ ← {x₁, ..., xₙ}
    selected in Uₜ by MOEA child selection algorithm

    Sₜ₋₁ ← Sₜ, Pₜ ← Pₜ₊₁, t ← t + 1
```

### Token Consumption Comparison

| Index | Type | LLM Token Count |
|-------|------|-----------------|
| 1 | Adaptive use of LLM | 56,620 |
| 2 | LLM used for each iteration | 283,100 |

**Benefits:**
- Adaptive mechanism reduces interaction costs by **~80%**
- Makes full use of LLM's rich knowledge
- Helps EA effectively search solution space
- Maximizes benefit while minimizing cost

**Hybrid Mechanism Advantages:**
- Leverages inherent characteristics of EAs
- Provides knowledge for LLM to guide search process
- Improves speed and quality of convergence
- General applicability
- Easy integration of various algorithms and auxiliary evaluation functions

---

### B. LLM as a Black-box Optimizer

#### Key Strategy

Guiding LLM to generate high-quality solutions using a **small number of instances** can effectively reduce interaction costs with LLM. Due to limited number of instances, we need precise prompts to guide LLM.

#### 1. Prompts the Desired Individual for Selection

**Requirement:**
Provide high-quality instances to help LLM learn their characteristics

**Flexibility:**
Number of selected individuals can be flexibly adjusted due to interactive flexibility of LLM

#### 2. Prompt Engineering

Divided into **4 parts** to make it easy for LLM to understand and respond quickly:

**Part 1: Identity Localization**
- Positions LLM as expert in multi-objective optimization
- Enhances logical reasoning efficiency and rigor
- Ensures accurate comprehension

**Part 2: Task Description**
- Brief description of the task
- Brief overview of input information nature
- Defines minimization problem involving multiple variables
- Provides relevant definitions and contextual examples

**Part 3: Context Information**
- Format of input information
- Sample information
- Number of parameters per sample
- Formatting conventions (e.g., <start> and <end> tags)

**Part 4: Expected Output**
- Output requirements and format
- Specify format of LLM output
- Each output sample begins with <start>, ends with <end>
- Formatted prompt improves LLM understanding
- Facilitates user extraction of results

#### Example Prompt

```
You are an expert in multi-objective optimization algorithms.
Your task is to generate improved solutions with better
objective values through given solutions.

I have several solutions, all of which are in the form of
10 dimensional decision vectors. The following is the
initial solution in the mating pool.

solution: <start>0.322,0.947,0.378,0.583<end>
obj value: 5.483
solution: <start>0.937,0.264,0.472,0.473<end>
obj value: 3.483
...
solution: <start>0.573,0.483,0.937,0.937<end>
obj value: 1.374
solution: <start>0.837,0.374,0.374,0.196<end>
obj value: 1.037

You can use these multi-objective optimization algorithms
to generate new solutions (one or more algorithms can be used).
Simply output three new solutions with better objective values.
Each solution must start with <start> and end with <end>.
```

#### 3. Error Handling

**Challenge:**
Due to randomness and uncertainty of LLM responses, behavior significantly different from traditionally manually designed evolutionary search operators. LLM may generate responses that deviate from expected format.

**Solution:**
- Rigorously validate generated text responses
- Initiate new context learning process if necessary
- Adopt **single-round polling strategy** to eliminate interference of historical information
- Ensure model always focuses on current prompt
- Prevents search capability from being influenced by historical information

---

## 5. Experiment

### A. Test Instances

Experiments conducted on widely used test instances:

**UF Test Suite:**
- Multi-objective problems from various aspects of real life
- Reflects generalization performance of framework
- Does not require excessive involvement of experts

**ZDT Test Suite:**
- Classic multi-objective test problems
- Various shapes of PF (Pareto Front) and PS (Pareto Set)
- Different problem characteristics

### B. Baseline Algorithms

Compared algorithms:
1. **NSGA-II** - Standard version
2. **NSGA-II-ARSBX** - Adaptive simulated binary crossover
3. **MOEA/D** - Decomposition-based
4. **NSGA-III** - Reference-point-based
5. **MOEA/D-DRA** - Dynamic resource allocation
6. **MOEA/D-DQN** - Deep Q-Network integration
7. **Our Framework** - LLM-assisted NSGA-II

**Note:**
NSGA-II and NSGA-II-ARSBX use same algorithmic framework as our framework, with main difference lying in reproductive strategy.

**Implementation:**
All algorithm implementations based on **PlatEMO**

### C. Experimental Settings

**Our Algorithm Settings:**
- Population number N: 100
- Maximum evaluation quantity Nₘₐₓ: 10,000
- Prompts to enter number of individuals l: 5
- Number of output items s: 3
- Decision threshold δ: 0.1
- Binary cross distribution index σ: 20
- Mutation score Index τ: 1
- Polynomial mutation distribution index χ: 20

**Note:**
Unmentioned settings same as in original papers

---

### D. Results

#### Convergence Analysis

Figure 2 shows convergence curves of **hypervolume (HV) values** over number of evaluations for four test instances.

**ZDT Test Instances:**
- Convergence speed of our framework is **fastest**
- Outperforms other algorithms in performance

**UF Test Instances (Real-world Problems):**
- Our framework converges relatively quickly
- Achieves better solutions
- Strong competitiveness and robustness
- Effectively accelerates convergence speed
- Optimizes solution set

**Example: UF1**
- Initial stage: Convergence speed slightly slower than other algorithms
- As iterations increase: Convergence speed gradually accelerates
- Final result: Solution set superior to those obtained by other algorithms

#### HV Comparison Results (Table III)

**ZDT Test Suite:**

| Problem | MOEA/D | MOEA/D-DQN | NSGA-II | NSGA-II-ARSBX | NSGA-III | NSGA-II-LLM |
|---------|--------|------------|---------|---------------|----------|-------------|
| ZDT1 | 5.5441e-1 (6.78e-2) - | 6.8687e-1 (2.79e-2) - | 6.4219e-1 (2.03e-1) - | 7.1420e-1 (8.46e-4) - | 6.9668e-1 (4.84e-3) - | **7.1777e-1 (5.50e-4)** |
| ZDT2 | 1.0568e-1 (4.04e-2) - | 4.2583e-1 (1.40e-2) - | 4.1981e-1 (1.15e-2) - | 4.3830e-1 (1.40e-3) - | 4.0955e-1 (1.15e-2) - | **4.4244e-1 (6.09e-4)** |
| ZDT3 | 5.5275e-1 (5.49e-2) - | 5.8643e-1 (1.66e-2) - | 6.0199e-1 (2.79e-2) + | 5.9654e-1 (6.57e-4) - | 5.8757e-1 (3.20e-3) - | **5.9851e-1 (3.56e-4)** |

**UF Test Suite:**

| Problem | MOEA/D | MOEA/D-DQN | NSGA-II | NSGA-II-ARSBX | NSGA-III | NSGA-II-LLM |
|---------|--------|------------|---------|---------------|----------|-------------|
| UF1 | 4.2971e-1 (4.23e-2) - | 4.8881e-1 (7.92e-2) - | 5.4658e-1 (4.74e-2) - | 5.8548e-1 (9.99e-3) = | 5.6095e-1 (3.79e-2) = | **5.8760e-1 (5.37e-3)** |
| UF2 | 5.5986e-1 (4.67e-2) - | 6.5250e-1 (1.22e-2) = | 6.4400e-1 (9.42e-3) - | 6.5703e-1 (5.95e-3) = | 6.4166e-1 (6.40e-3) - | **6.5787e-1 (6.13e-3)** |
| UF3 | 3.2073e-1 (2.32e-2) - | 3.7056e-1 (1.58e-2) - | 2.8297e-1 (4.69e-2) - | 4.5375e-1 (5.49e-2) = | 2.5261e-1 (4.53e-2) - | **4.6415e-1 (2.44e-2)** |
| UF4 | 2.7356e-1 (8.03e-3) - | 3.4066e-1 (1.23e-2) - | 3.4289e-1 (6.12e-3) - | 3.4839e-1 (2.86e-3) - | 3.3845e-1 (3.89e-3) - | **3.5873e-1 (3.54e-3)** |
| UF7 | 2.1599e-1 (6.18e-2) - | 2.9221e-1 (1.41e-1) - | 3.7465e-1 (9.21e-2) - | 5.0681e-1 (9.75e-3) = | 4.0941e-1 (9.08e-2) - | **5.1215e-1 (3.58e-3)** |

**Statistical Summary:**
- Format: mean (standard deviation)
- Best values in **bold**
- Each test instance run **10 times independently**
- All data are average values
- (+/-/=): Better/Worse/Equal compared to NSGA-II-LLM

**Overall Performance:**
Our framework shows **significant auxiliary role** of LLM. LLM only needs a few examples to derive excellent solutions, **without any artificial design and domain knowledge**. Due to rich prior knowledge of LLM, framework shows **strong performance on UF examples** of real problems.

#### IGD Comparison Results (Table IV)

Similar patterns observed with **Inverted Generational Distance (IGD)** metric:
- Our framework consistently achieves best or competitive IGD values
- Particularly strong on UF test suite
- Demonstrates both convergence and diversity advantages

#### Pareto Front Visualization

**ZDT Instances (Figures 3 and 4):**
- Our framework has **higher convergence rate**
- **Better fitting effect** on ZDT instances
- More accurate approximation of true Pareto front

**UF2 and UF3 Instances (Figures 5 and 6):**
- Our framework achieves **superior distribution of diversity**
- Solutions present in all directions
- More accurately fits actual PF surface

**Conclusion:**
Introduction of LLM not only helps to **accelerate convergence** but also contributes to **maintaining diversity** of solutions.

---

### E. Ablation Study

#### Purpose

Verify that decision threshold δ can effectively:
- Help framework reduce number of LLM interactions
- Reduce costs

#### Methodology

Tested score range (0, 1] of auxiliary evaluation function with decision threshold sizes:
- δ = 0.01
- δ = 0.05
- δ = 0.1
- δ = 0.5
- δ = 1.0

All other experimental settings kept unchanged.

#### Results (Figure 7)

**Token Usage Analysis:**
- Tested number of tokens used in single run for each input
- Measured IGD values on UF instance

**Key Finding:**
Experimental results confirm that when δ = **0.1**:
- IGD performs **best**
- Token cost is **small**
- No significant difference in performance

**Optimal Configuration:**
Decision threshold δ = 0.1 provides best **trade-off between cost and performance**

---

## 6. Future Works

In future work, we will focus on applying large language models to multi-objective evolutionary algorithms in the following directions:

### 1. Automatic Feature Extraction and Problem Modeling

**Objectives:**
- Use LLM to understand real-world problems described in natural language
- Automatically extract key features and constraints
- Based on problem, generate specific solutions
- Construct fitness functions

### 2. Algorithm Parameter Optimization

**Objectives:**
- Apply LLM to predict and optimize MOEA parameters (e.g., population size, crossover and mutation probability)
- Develop adaptive parameter adjustment mechanism based on LLM
- Deal with dynamic characteristics of different problems

### 3. Algorithm Structure Innovation

**Objectives:**
- Use LLM to generate new algorithm structures or operators

### 4. Decision Support System

**Objectives:**
- Develop LLM-based decision support tool
- Help decision-makers understand obtained solutions
- Make decision-making process more transparent and interpretable

---

## 7. Conclusion

This paper discusses the application of LLM in multi-objective evolutionary optimization. By leveraging the power of a pre-trained LLM, we propose a novel way to use LLM as a **black-box search operator** in the MOEA framework.

### Key Contributions

**1. Cost Reduction:**
Since LLM interactions consume a lot of time and resources, we employ:
- **Hybrid strategies** to minimize LLM calls
- **Flexible adjustment mechanisms** to optimize LLM usage

**2. Framework Advantages:**
- Adaptive mechanism ensures LLM assistance is maximized
- Hybrid mechanism minimizes interaction costs
- General applicability to various MOEA algorithms

**3. Experimental Validation:**
- Demonstrates effectiveness of proposed method
- Performance is **competitive** with widely used MOEAs
- **Ranks first** in some test instances

**4. Future Potential:**
As LLMs continue to evolve and improve, they will play an increasingly significant role in:
- Multi-objective optimization
- Other complex decision-making challenges

---

## References

### LLM and Optimization

[1] Meyerson, E., Nelson, M. J., Bradley, H., Moradi, A., Hoover, A. K., and Lehman, J. "Language model crossover: Variation through few-shot prompting." arXiv:2302.12170, 2023.

[2] Liu, S., Chen, C., Qu, X., Tang, K., and Ong, Y.-S. "Large language models as evolutionary optimizers." arXiv:2310.19046, 2023.

[3] Liu, F., Lin, X., Wang, Z., Yao, S., Tong, X., Yuan, M., and Zhang, Q. "Large language model for multi-objective evolutionary optimization." arXiv:2310.12541, 2023.

[4] Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. "Large language models as optimizers." arXiv:2309.03409, 2023.

[5] Bradley, H., Dai, A., Teufel, H. B., Zhang, J., Oostermeijer, K., Bellagente, M., Clune, J., Stanley, K., Schott, G., and Lehman, J. "Quality-diversity through ai feedback." Proceedings of the 2nd Agent Learning in Open-Endedness Workshop and in 37th Annual Conference on Neural Information Processing Systems, 2023.

[6] Sanderson, K. "Gpt-4 is here: what scientists think." Nature, vol. 6, no. 2, pp. 182–197, 2023.

[7] Chen, J., Liu, Z., Huang, X., Wu, C., Liu, Q., Jiang, G., Pu, Y., Lei, Y., and W. et al. "When large language models meet personalization: Perspectives of challenges and opportunities." arXiv:2307.16376, 2023.

[8] Guo, P.-F., Chen, Y.-H., Tsai, Y.-D., and Lin, S.-D. "Towards optimizing with large language models." arXiv:2310.05204, 2023.

[9] Wu, X., Zhong, Y., Wu, J., and Tan, K. C. "As-llm: When algorithm selection meets large language model." arXiv:2311.13184, 2023.

[10] Chen, H., Constante-Flores, G. E., and Li, C. "Diagnosing infeasible optimization problems using large language models." arXiv:2308.12923, 2023.

[11] Pluhacek, M., Kazikova, A., Kadavy, T., Viktorin, A., and Senkerik, R. "Leveraging large language models for the generation of novel metaheuristic optimization algorithms." Proceedings of the Companion Conference on Genetic and Evolutionary Computation, pp. 1812–1820, 2023.

[12] Bradley, H., Fan, H., Galanos, T., Zhou, R., Scott, D., and Lehman, J. "The openelm library: Leveraging progress in language models for novel evolutionary algorithms."

### Multi-objective Optimization

[19] Miettinen, Kaisa. Nonlinear multiobjective optimization. Springer Science & Business Media, 2012, vol. 12.

[20] Srinivas, N., and Deb, Kalyanmoy. "Muiltiobjective optimization using nondominated sorting in genetic algorithms." Evolutionary Computation, vol. 2, no. 3, pp. 221–248, 1994.

[21] Deb, Kalyanmoy, and Srinivasan, Aravind. "Multi-objective genetic algorithms: Problem difficulties and construction of test problems." Proceedings of the International Conference on Genetic Algorithms, pp. 109–115, 1995.

[22] Xiong, Miao, Hu, Zhiyuan, Lu, Xinyang, Li, Yifei, Fu, Jie, He, Junxian, and Hooi, Bryan. "Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms." arXiv:2306.13063, 2023.

[23] Li, Hui, and Zhang, Qingfu. "Multiobjective optimization problems with complicated pareto sets and MOEA/D and NSGA-II." IEEE transactions on evolutionary computation, vol. 13, no. 2, pp. 284–302, 2008.

[24] Zitzler, Eckart, Deb, Kalyanmoy, and Thiele, Lothar. "Comparison of multiobjective evolutionary algorithms: Empirical results." Evolutionary computation, vol. 8, no. 2, pp. 173–195, 2000.

### MOEA Algorithms

[25] Pan, Linqiang, Xu, Wenting, Li, Lianghao, He, Cheng, and Cheng, Ran. "Adaptive simulated binary crossover for rotated multi-objective optimization." Swarm and Evolutionary Computation, vol. 60, p. 100759, 08 2020.

[26] Zhang, Qingfu, and Li, Hui. "Moea/d: A multiobjective evolutionary algorithm based on decomposition." IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712–731, 2007.

[27] Deb, Kalyanmoy, and Jain, Himanshu. "An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach and part i: solving problems with box constraints." IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577–601, 2014.

[28] Zhang, Qingfu, Zhou, Aimin, Zhao, Shengxiang, Suganthan, P. Nagaratnam, Liu, Wenyin, and Zhang, Weiyin. "Multiobjective optimization test instances for the cec 2009 special session and competition." IEEE Congress on Evolutionary Computation. IEEE, 2009, pp. 1–30.

[29] Zhang, Hui, Zhou, Aimin, and Zhang, Qingfu. "A deep reinforcement learning based multiobjective evolutionary algorithm using decomposition." IEEE Transactions on Evolutionary Computation, vol. 24, no. 3, pp. 494–507, 2020.

[30] Tian, Ye, Cheng, Ran, Zhang, Xingyi, and Jin, Yaochu. "Platemo: A matlab platform for evolutionary multi-objective optimization [educational forum]." IEEE Computational Intelligence Magazine, vol. 12, no. 4, pp. 73–87, 2017.

---

*Document converted from PDF to Markdown - 2025-01-XX*
*Original: 4-LLM_Aided_MOEA_2024.pdf (746.6KB)*
