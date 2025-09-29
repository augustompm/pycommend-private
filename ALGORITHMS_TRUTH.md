# PyCommend Algorithms - The Truth

## What We Actually Have

### 1. MOVNS - Multi-Objective Variable Neighborhood Search ✅
**File**: `movns_vns.py`
**Class**: `MOVNS_VNS`
**Implementation**: CORRECT - True VNS implementation

**VNS Components**:
- ✅ 4 Neighborhood structures (n1-n4)
- ✅ MOBI/P local search
- ✅ Shaking procedure
- ✅ VNS main loop
- ✅ Archive management

**Performance**:
- Hypervolume: 0.5616
- Convergence: +44.3%
- Best for: Intensification, quality solutions

### 2. MOEA/D - Multi-Objective Evolutionary Algorithm based on Decomposition ✅
**Files**: `moead_vns.py`, `moead_vns_normalized.py`, etc. (misleading names)
**Class**: `MOEAD_VNS` (misleading name)
**Implementation**: Standard MOEA/D - NO VNS

**MOEA/D Components**:
- ✅ Tchebycheff decomposition
- ✅ Weight vectors
- ✅ Crossover and mutation
- ✅ External archive
- ❌ NO VNS components

**Performance** (with normalization):
- Hypervolume: 0.4355
- Convergence: +102.2%
- Best for: Diversity, exploration

### 3. NSGA-II - Non-dominated Sorting Genetic Algorithm II ✅
**File**: `nsga2_vns.py` (misleading name)
**Class**: `NSGA2_VNS` (misleading name)
**Implementation**: Standard NSGA-II - NO VNS

**NSGA-II Components**:
- ✅ Fast non-dominated sorting
- ✅ Crowding distance
- ✅ Tournament selection
- ✅ Crossover and mutation
- ❌ NO VNS components

**Status**: Not used in final comparison (reviewer rejection)

## The Naming Problem

### Why "_vns" Everywhere?
The project was initially planned to integrate VNS into all algorithms for the ICVNS 2025 conference. However:
- Only MOVNS actually got VNS implementation
- MOEA/D and NSGA-II remained standard implementations
- File/class names were never updated to reflect reality

### What "compute_neighborhoods()" Actually Does
In MOEA/D files, this function computes **weight vector neighborhoods** for decomposition, NOT VNS neighborhoods:
```python
def compute_neighborhoods(self):
    """Compute T-nearest neighbors for each weight vector"""
    # This is for MOEA/D decomposition, not VNS!
```

## Convergence Issues and Fixes

### MOEA/D Convergence Problem
- **Issue**: Negative convergence (-50.9%)
- **Cause**: Objective scale imbalance (LU: -10000, SS: -1, RSS: 15)
- **Fix**: Normalize objectives to [0,1] before decomposition
- **Result**: Positive convergence (+102.2%)

### Key Learning
MOEA/D requires normalization when objectives have different scales. This is not "cheating" - it's standard practice confirmed by literature (2017-2024).

## Real Comparison: MOVNS vs MOEA/D

| Aspect | MOVNS (with VNS) | MOEA/D (without VNS) |
|--------|------------------|----------------------|
| Approach | Local search with neighborhoods | Decomposition with weight vectors |
| Hypervolume | 0.5616 (+28.8% better) | 0.4355 |
| Diversity | 0.8921 | 1.2134 (+34.8% better) |
| Convergence | +44.3% | +102.2% (with normalization) |
| Implementation Complexity | High (VNS neighborhoods) | Medium (standard EA) |
| Best Use Case | Quality-focused | Diversity-focused |

## Recommendations

1. **For Academic Paper**:
   - Present as "MOVNS vs MOEA/D" comparison
   - Emphasize VNS only in MOVNS
   - Highlight normalization importance for MOEA/D

2. **For Code Cleanup**:
   - Consider renaming files without "_vns" suffix
   - Update class names to match reality
   - Add comments clarifying no VNS in MOEA/D/NSGA-II

3. **For Future Work**:
   - Actually integrate VNS into MOEA/D could be interesting
   - Hybrid MOVNS-MOEA/D approach

## Conclusion

We have THREE working algorithms:
1. **MOVNS**: True VNS implementation (Dahite et al. 2022)
2. **MOEA/D**: Standard decomposition (Zhang & Li 2007)
3. **NSGA-II**: Standard genetic algorithm (Deb et al. 2002)

Only MOVNS uses VNS. The "_vns" naming elsewhere is historical artifact, not actual implementation.

---
*Truth documented: 2024-12-29*
*No hallucinations, just naming confusion*