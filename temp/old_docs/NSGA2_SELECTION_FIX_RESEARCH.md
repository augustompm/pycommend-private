# NSGA-II Survivor Selection: Research Review and Fix

## Literature Review (2023-2024)

### The Bug We Found
Our debug revealed that population shrinks to 0 during survivor selection. This aligns with known issues in NSGA-II implementation.

### Key Research Findings

#### 1. Traditional NSGA-II Problem (2024 arxiv:2407.17687)
> "The NSGA-II computes the crowding distance once and then repeatedly removes individuals with smallest crowding distance **without updating** the crowding distance after each removal."

This is exactly our bug! The selection stops too early without properly filling the population.

#### 2. Crowding Distance Instability (2023 ACM)
> "Instability stems from cases where two or more individuals share identical fitnesses. Their crowding distance either becomes null, or depends on position within the Pareto front."

With 4 objectives, this is more likely to happen.

#### 3. Current Crowding Distance Solution (2024)
> "An efficient way to implement NSGA-II using the **current crowding distance** - recalculating after each removal - approximates the Pareto front much better."

## The Correct Implementation

### Original NSGA-II (Deb et al. 2002)
```python
# CORRECT survivor selection:
1. Combine parent and offspring populations (size 2N)
2. Fast non-dominated sort into fronts F1, F2, ...
3. Fill new population with complete fronts until size > N
4. For the last front that doesn't fit completely:
   a. Calculate crowding distance for ALL individuals in that front
   b. Sort by crowding distance (descending)
   c. Select top individuals to fill exactly N
```

### Our Bug
```python
# WRONG (current implementation):
for front in fronts:
    if len(new_population) + len(front) <= pop_size:
        new_population.extend([population[i] for i in front])
    else:
        break  # ← STOPS WITHOUT FILLING!
```

### The Fix
```python
# CORRECT implementation:
new_population = []
for front_idx, front in enumerate(fronts):
    if len(new_population) + len(front) <= pop_size:
        # Add entire front
        new_population.extend([population[i] for i in front])
    else:
        # Last front - use crowding distance to select
        remaining = pop_size - len(new_population)
        front_individuals = [population[i] for i in front]

        # Calculate crowding distance
        crowding_distance_assignment(front_individuals)

        # Sort by crowding distance (descending)
        front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)

        # Add top individuals to fill population
        new_population.extend(front_individuals[:remaining])
        break

# Ensure we have exactly pop_size
assert len(new_population) == pop_size
```

## Recent Improvements (2023-2024)

### 1. Crowding Distance Elimination (CDE)
Instead of calculating once, recalculate after each removal:
```python
while len(front) > needed:
    calculate_crowding_distance(front)
    remove_individual_with_min_distance(front)
```

### 2. Truthful Crowding Distance
For many objectives (like our 4), use normalized Manhattan distance that better represents true diversity.

### 3. Unique Fitness Calculation
Calculate crowding distance on unique objective vectors, not individuals, to avoid instability.

## Implementation for PyCommend v6

### Current Issue
- Population: 100 → Fronts fill to ~50 → Selection stops → Population = 0

### Required Fix
1. Always fill to exactly pop_size
2. Use crowding distance for partial front
3. Handle edge cases (empty fronts, all dominated)

### Code to Add
```python
def environmental_selection(self, combined_population, pop_size):
    """
    Correct NSGA-II environmental selection
    Returns exactly pop_size individuals
    """
    fronts = self.fast_non_dominated_sort(combined_population)

    new_population = []

    for front_idx, front in enumerate(fronts):
        front_individuals = [combined_population[i] for i in front]

        if len(new_population) + len(front_individuals) <= pop_size:
            # Entire front fits
            new_population.extend(front_individuals)
        else:
            # Partial front - use crowding distance
            remaining = pop_size - len(new_population)

            if remaining > 0:
                # Calculate crowding distance
                self.crowding_distance_assignment(front_individuals)

                # Sort by crowding distance (preserve diversity)
                front_individuals.sort(
                    key=lambda x: x['crowding_distance'],
                    reverse=True
                )

                # Take top individuals
                new_population.extend(front_individuals[:remaining])

            break

    # Failsafe: if still not enough, add random
    while len(new_population) < pop_size:
        # This should never happen with correct implementation
        print(f"WARNING: Adding random individual to fill population")
        new_individual = self.smart_initialization('hybrid')
        objectives = self.evaluate_objectives(new_individual)
        new_population.append({
            'chromosome': new_individual,
            'objectives': objectives,
            'rank': None,
            'crowding_distance': 0
        })

    return new_population[:pop_size]  # Ensure exactly pop_size
```

## References

1. **2024 - ArXiv:2407.17687**: "A Crowding Distance That Provably Solves the Difficulties of the NSGA-II"
2. **2023 - ACM GECCO**: "Better approximation guarantees for the NSGA-II by using the current crowding distance"
3. **2023 - Semantic Scholar**: "Improved Crowding Distance for NSGA-II"
4. **2024 - GeeksforGeeks**: Current implementation guides

## Summary

The bug is a classic NSGA-II implementation error: not properly handling the last partial front during survivor selection. The fix is well-documented in literature and involves using crowding distance to select from the last front that doesn't completely fit.