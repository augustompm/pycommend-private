# Final Algorithms Summary - PyCommend v12

## Two Core Algorithms

### 1. MOVNS Advanced
**File**: `pycommend-code/src/optimizer/movns_advanced.py`
**Test**: `test_movns_advanced.py`, `test_movns_20.py`
**Report**: `MOVNS_ADVANCED_FINAL_REPORT.md`

#### Algorithm Components
1. **Pareto Local Search (PLS)**
   - Queue-based, max 20 neighbors
   - Non-dominated archive management

2. **Simulated Annealing**
   - Temperature: 1.0, cooling: 0.995
   - Multi-objective acceptance criterion

3. **Tabu Search**
   - Memory: deque(maxlen=50)
   - Prevents cycling

4. **Iterated Local Search**
   - 5 iterations per cycle
   - Adaptive perturbation

5. **Aggressive Local Search**
   - 4 operators: cooccurrence, semantic, cluster, exchange
   - Intensity: 10-20 (adaptive)

6. **Adaptive Learning**
   - Success rates tracking
   - Learning rates: [0.1, 1.0]

#### Performance
- **HV (20 iter)**: 0.3022
- **HV (15 iter)**: 0.2761
- **Archive size**: 17-30 solutions
- **Time**: 8-15 seconds
- **Best LU**: 2346
- **Best SS**: 0.841
- **Best RSS**: 2.3

### 2. MOEA/D Normalized
**File**: `pycommend-code/src/optimizer/moead_normalized.py`
**Test**: Built into comparison tests
**Base**: Zhang & Li (2007) IEEE TEVC

#### Algorithm Components
1. **Decomposition**: Tchebycheff
2. **Population**: 100 individuals
3. **Weight vectors**: Uniform distribution
4. **Neighborhood**: 20 neighbors
5. **Normalization**: Dynamic [0,1]

#### Performance (Typical)
- **HV**: ~0.23-0.24
- **Archive size**: 100 solutions
- **Time**: 30-60 seconds
- **Best LU**: ~5000
- **Best SS**: ~0.85
- **Best RSS**: 2.3

## Comparison Results

| Metric | MOVNS Advanced | MOEA/D | Difference |
|--------|---------------|---------|------------|
| HV (final) | 0.3022 | ~0.24 | +26-31% |
| Time | 15.3s | 30-60s | 2-4x faster |
| Archive | 30 | 100 | 70% smaller |
| Convergence | Monotonic | Variable | More stable |

## Key References

### MOVNS
- Dahite et al. (2022). Mathematics 10(11), 1807
- MOBI/P strategy for multi-objective VNS

### MOEA/D
- Zhang & Li (2007). IEEE TEVC 11(6), 712-731
- Decomposition-based multi-objective evolution

## Problem Definition

**Objectives**:
1. Maximize Linked Usage (LU) - co-occurrence
2. Maximize Semantic Similarity (SS) - coherence
3. Minimize Recommended Set Size (RSS) - compactness

**Dataset**:
- 9,997 Python packages
- 8,794 requirements.txt files
- 384-dim SBERT embeddings
- 200 K-means clusters

## Usage

### MOVNS Advanced
```python
from optimizer.movns_advanced import MOVNS_Advanced

algo = MOVNS_Advanced(
    'fastapi',
    archive_size=100,
    max_iterations=20,
    track_metrics=True
)
solutions = algo.run()
```

### MOEA/D Normalized
```python
from optimizer.moead_normalized import MOEAD_Normalized

algo = MOEAD_Normalized(
    'fastapi',
    pop_size=100,
    max_gen=30
)
solutions = algo.run()
```

## Key Findings

1. **MOVNS Advanced superiority**: Aggressive local search without simplifications beats decomposition
2. **Quality over quantity**: 30 high-quality solutions better than 100 mediocre ones
3. **Adaptive mechanisms crucial**: 15% performance gain from learning rates
4. **VNS for intensification**: Local search focus superior for this problem structure

## Files Structure

```
pycommend/
├── pycommend-code/src/optimizer/
│   ├── movns_advanced.py      # Final MOVNS
│   └── moead_normalized.py    # Final MOEA/D
├── test_movns_advanced.py     # Main test
├── test_movns_20.py          # 20 iteration test
├── test_movns_vs_moead.py    # Direct comparison
├── MOVNS_ADVANCED_FINAL_REPORT.md
├── article.md                # Scientific paper
└── articles.md               # References database
```