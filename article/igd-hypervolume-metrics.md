# Performance Metrics for Multi-Objective Optimization

## 1. Inverted Generational Distance (IGD)

### Overview
The Inverted Generational Distance (IGD) is a comprehensive performance indicator that simultaneously measures:
- **Convergence**: How close the obtained solutions are to the true Pareto front
- **Diversity**: How well-distributed the solutions are across the objective space

IGD calculates the average distance from each reference point on the true Pareto front to its nearest solution in the approximation set.

### Mathematical Formulation

#### Standard IGD
```
IGD(A,Z) = (1/|Z|) * Σᵢ₌₁^|Z| d(zᵢ,A)
```

Where:
- **A**: Approximation set (obtained solutions)
- **Z**: Reference set (points on true Pareto front)
- **d(zᵢ,A)**: Minimum Euclidean distance from reference point zᵢ to set A
- **|Z|**: Number of reference points

#### Distance Calculation
```
d(zᵢ,A) = min{||zᵢ - a|| : a ∈ A}
```

Where ||·|| denotes the Euclidean norm.

### Modified IGD+ Indicator

IGD+ addresses the weakness of standard IGD regarding Pareto compliance.

#### IGD+ Formula
```
IGD+(A,Z) = (1/|Z|) * Σᵢ₌₁^|Z| d⁺(zᵢ,A)
```

#### Modified Distance for IGD+
For minimization problems:
```
d⁺(z,a) = √(Σₖ₌₁^M (max{zₖ - aₖ, 0})²)
```

Where:
- M is the number of objectives
- The max operator ensures only dominated distances contribute

### Interpretation
- **Lower IGD values** indicate better performance
- **IGD = 0** means the approximation set perfectly covers all reference points
- Small IGD values suggest:
  - Good convergence to the Pareto front
  - Good distribution across the objective space

### Implementation Considerations

1. **Reference Set Generation**:
   - Must be uniformly distributed along true Pareto front
   - Density affects metric sensitivity
   - More points provide better resolution

2. **Normalization**:
   - Normalize objectives to [0,1] when scales differ
   - Prevents bias toward objectives with larger ranges

3. **Computational Complexity**:
   - O(|A| × |Z|) for distance calculations
   - Can be expensive for large reference sets

---

## 2. Hypervolume Indicator (HV)

### Overview
The Hypervolume indicator measures the volume of the objective space dominated by a solution set and bounded by a reference point. It is the only unary indicator known to be strictly Pareto-compliant.

### Mathematical Formulation

#### Basic Definition
```
HV(A,r) = volume(⋃ₐ∈A [a,r])
```

Where:
- **A**: Approximation set
- **r**: Reference point (must dominate all solutions)
- **[a,r]**: Hyperbox between solution a and reference point r

#### For 2D Problems
```
HV = Σᵢ₌₁^n (xᵢ - xᵢ₊₁) × (yᵢ - r_y)
```
After sorting solutions by first objective.

#### For Higher Dimensions
Recursive formulation or specialized algorithms required.

### Reference Point Selection

1. **Nadir Point Method**:
   ```
   r = nadir + offset
   ```
   Where offset is typically 10-20% of the range

2. **Fixed Reference**:
   ```
   r = (r₁, r₂, ..., rₘ)
   ```
   Predetermined based on problem knowledge

3. **Dynamic Reference**:
   ```
   rᵢ = max{fᵢ(a) : a ∈ A} + δᵢ
   ```
   Adapts to current approximation set

### Properties

#### Advantages
- **Pareto Compliance**: Strictly respects Pareto dominance
- **No True Front Required**: Only needs reference point
- **Single Value**: Easy to compare algorithms
- **Theoretical Foundation**: Well-established properties

#### Disadvantages
- **Computational Cost**: NP-hard for many objectives
- **Reference Point Sensitivity**: Results depend on reference choice
- **Bias**: May favor certain regions of Pareto front

### Computational Algorithms

#### 2D Case (O(n log n))
```python
def hypervolume_2d(points, ref):
    points = sorted(points, key=lambda p: p[0])
    hv = 0
    prev_x = ref[0]
    for point in reversed(points):
        if point[1] < ref[1]:
            hv += (prev_x - point[0]) * (ref[1] - point[1])
            prev_x = point[0]
    return hv
```

#### General Case Complexity
- **Exact**: O(n^(d-2) log n) worst-case
- **Monte Carlo**: O(1/ε²) for ε-approximation
- **Practical limit**: ~10 objectives for exact computation

### Hypervolume Contribution

Individual solution contribution:
```
HC(a,A) = HV(A) - HV(A \ {a})
```

Used for:
- Solution ranking
- Archive maintenance
- Selection operators

---

## 3. Comparison: IGD vs Hypervolume

### When to Use IGD
- Known true Pareto front
- Many objectives (>5)
- Fast evaluation needed
- Uniform coverage important

### When to Use Hypervolume
- Unknown true Pareto front
- Few objectives (≤5)
- Theoretical guarantees needed
- Pareto compliance critical

### Combined Usage
Many studies use both metrics:
- **IGD**: Measures convergence and diversity
- **HV**: Confirms Pareto optimality
- Together provide comprehensive assessment

---

## 4. Practical Implementation Guidelines

### IGD Implementation Steps
1. Generate reference set Z from true Pareto front
2. Normalize objectives if needed
3. For each reference point, find nearest solution
4. Calculate average distance

### Hypervolume Implementation Steps
1. Select appropriate reference point
2. Remove dominated solutions
3. Apply dimension-sweep algorithm
4. Return calculated volume

### Python Example (using pymoo)
```python
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV

# IGD calculation
ind = IGD(ref_points)
igd_value = ind(solutions)

# Hypervolume calculation
ind = HV(ref_point=nadir_point * 1.1)
hv_value = ind(solutions)
```

### Quality Thresholds
- **Good IGD**: < 0.01 (normalized space)
- **Good HV**: > 0.9 × ideal_hypervolume
- Values are problem-dependent

---

## 5. Advanced Considerations

### Normalized Metrics
- **IGD**: Normalize to [0,1] per objective
- **HV**: Normalize using ideal and nadir points

### Statistical Analysis
- Run algorithms 30+ times
- Report mean, median, standard deviation
- Use statistical tests (Wilcoxon, Friedman)

### Metric Limitations
- **IGD**: Requires true Pareto front knowledge
- **HV**: Exponential complexity in objectives
- Both: May not capture user preferences

### Alternative Metrics
- **Generational Distance (GD)**: Convergence only
- **Spacing**: Distribution uniformity
- **Spread**: Coverage extent
- **Epsilon Indicator**: Minimum translation distance