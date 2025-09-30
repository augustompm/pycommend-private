# Multi-Objective Optimization Metrics Knowledge Base

## Most Common Metrics (2024 Literature)

### 1. Hypervolume (HV) - Most Popular
- **Usage**: Most widely adopted metric
- **Measures**: Volume dominated in objective space
- **Pros**: Pareto compliant, single value metric
- **Cons**: Computationally expensive for many objectives
- **Formula**: Volume of union of hypercubes dominated by solutions

### 2. IGD+ (Inverted Generational Distance Plus) - Recommended
- **Usage**: Second most common, replacing IGD
- **Measures**: Average distance from reference set to solution set
- **Pros**: Weakly Pareto compliant, fast computation
- **Cons**: Requires reference Pareto front
- **Formula**: IGD+ = (1/|R|) × Σ min d+(r,s) for r in R, s in S
- **Key**: Uses modified distance when solution dominates reference point

### 3. IGD (Original) - Still Used but Deprecated
- **Issue**: Not Pareto compliant (can give misleading results)
- **Recommendation**: Use IGD+ instead

### 4. Other Common Metrics
- **GD (Generational Distance)**: Distance from solutions to reference
- **Spacing**: Uniformity of solution distribution
- **Maximum Spread**: Coverage of objective space
- **Epsilon Indicator**: Multiplicative distance to reference

## IGD+ Implementation Details

### Distance Calculation
```
If solution s dominates reference point r:
    d+(r,s) = modified distance considering dominance
Else:
    d+(r,s) = Euclidean distance
```

### Requirements
1. Reference Pareto front (true or approximated)
2. Normalized objectives for fair distance calculation

## Best Practices (2024)

1. **Primary Metrics**: HV and IGD+
2. **Complementary**: Spacing for distribution quality
3. **Statistical**: Mean ± std over multiple runs
4. **Normalization**: Essential for distance-based metrics

## References

- Ishibuchi et al. (2015): Modified Distance Calculation in GD and IGD
- Lopez-Ibanez & Paquete: IGD+ implementation in eaf package
- Taylor & Francis (2024): Diagnostic benchmarking of MOEAs
- IEEE (2024): IGD Indicator-Based Evolutionary Algorithms