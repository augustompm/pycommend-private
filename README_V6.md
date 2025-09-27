# PyCommend v6 - Multi-objective Package Recommender

## Status
**Version**: v6 (2025-09-27)
**Implementation**: Complete with 4 objectives and full SBERT integration
**Known Issue**: Population selection bug identified in tests

## Key Features v6

### 1. Four Optimization Objectives
- **F1**: Co-occurrence strength (maximize)
- **F2**: Weighted semantic similarity (maximize)
- **F3**: Set coherence via embeddings (maximize) ← NEW!
- **F4**: Balanced size (minimize)

### 2. Full SBERT Integration (100%)
```python
# All 3 data sources now in use:
1. package_relationships_10k.pkl     # Co-occurrence matrix
2. package_similarity_matrix_10k.pkl # SBERT similarity
3. package_embeddings_10k.pkl       # Raw embeddings (384-dim) ← NOW USED!
```

### 3. Semantic Clustering
- 200 K-means clusters for domain segmentation
- Pre-computed candidate pools per package
- Cluster-based initialization strategy

### 4. Hybrid Initialization
- 40% co-occurrence-based
- 40% similarity-based
- 20% diversity from other clusters

## Architecture

```
pycommend-code/
├── src/optimizer/
│   ├── nsga2_v5.py           # Main implementation (520 lines)
│   ├── nsga2_integrated.py   # v4 with Weighted Probability
│   └── moead_integrated.py   # MOEA/D with Weighted Probability
├── data/
│   ├── package_relationships_10k.pkl
│   ├── package_similarity_matrix_10k.pkl
│   └── package_embeddings_10k.pkl
└── tests/
    ├── test_nsga2_real.py     # Real unit tests (no fallbacks)
    ├── test_nsga2_focused.py  # Focused performance tests
    └── debug_nsga2_v5.py      # Debug utilities
```

## Test Results

### Working Components ✓
```python
# From debug_nsga2_v5.py:
- Objectives calculation: OK
  F1=-4002.60, F2=-0.338, F3=-0.625, F4=11.40

- Clustering: OK
  numpy in cluster 35 (131 members)
  flask in cluster 184 (38 members)

- Initialization: OK
  All 4 strategies produce valid solutions

- Pareto sorting: OK (initially)
  Front 0: 8 individuals, Front 1: 2 individuals
```

### Known Issue ❌
```python
# Population shrinking bug:
Generation 0: Population 20 → Pareto 12 → Selected 0
Generation 1: ERROR - Cannot select from empty population
```

## Evolution Summary

| Version | Success Rate | Data Used | Objectives | Status |
|---------|-------------|-----------|------------|---------|
| v1 | 4.0% | 2/3 | 3 | Basic |
| v4 | 26.7% | 2/3 | 3 | Weighted Probability |
| v6 | TBD | 3/3 ✓ | 4 ✓ | Bug in selection |

## Installation & Usage

```bash
# Clone repository
git clone https://github.com/augustompm/pycommend-private.git
cd pycommend

# Test NSGA-II v5 (renamed from v5 internally)
cd pycommend-code
python -m src.optimizer.nsga2_v5 --package numpy

# Run tests
python test_nsga2_real.py      # Real tests
python debug_nsga2_v5.py       # Debug mode
```

## Key Code: Semantic Coherence (F3)

```python
def evaluate_objectives(self, chromosome):
    # ... F1, F2 calculations ...

    # F3: NEW - Semantic coherence using raw embeddings
    if len(indices) > 1:
        selected_embeddings = self.embeddings[indices]
        centroid = np.mean(selected_embeddings, axis=0)
        coherence_scores = cosine_similarity(selected_embeddings, [centroid])
        coherence = np.mean(coherence_scores)
    else:
        coherence = 0.5
    f3 = -coherence  # Maximize coherence

    # F4: size penalty...
```

## Next Steps

1. **Fix selection bug**: Ensure population maintains size
2. **Validate performance**: Measure actual success rate
3. **Optimize convergence**: Tune parameters for 4 objectives
4. **Production deployment**: After bug fixes

## Documentation

- `NSGA2_V5_FINAL_STATUS.md` - Detailed bug analysis
- `PROJECT_V5.md` - Full technical specification
- `CLAUDE.md` - Project memory and context

## Contributors

Project developed with Claude (Anthropic) following `rules.json` guidelines:
- No inline comments
- Clean Python code
- Real unit tests without fallbacks
- Comprehensive documentation

---
*PyCommend v6 - Full SBERT Integration with 4 Objectives*
*Bug identified, architecture complete*