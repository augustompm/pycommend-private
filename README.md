# PyCommend - Python Package Recommendation System

## Overview
Multi-objective optimization system for Python package recommendations using NSGA-II and MOEA/D algorithms with intelligent initialization.

## Current Solution: Weighted Probability Initialization
- **74.3% success rate** (vs 4% random)
- **62x better performance** in connection strength
- Based on research from Zhang et al. (2023) and Sharma & Trivedi (2020)

## Project Structure
```
pycommend/
├── pycommend-code/          # Optimization algorithms
│   ├── src/
│   │   └── optimizer/
│   │       ├── nsga2.py    # NSGA-II implementation
│   │       └── moead.py    # MOEA/D implementation
│   └── data/
│       └── package_relationships_10k.pkl
├── article/                 # Scientific documentation
│   ├── constructive.md     # Weighted Probability method
│   ├── audit-sources.md    # Source verification
│   └── test-results-summary.md
├── temp/                    # Testing and implementation
│   ├── test_initialization_methods.py
│   ├── simple_best_method.py
│   └── old/                # Archive
└── CLAUDE.md               # Project memory
```

## Quick Start

### Test Initialization Methods
```bash
cd temp
python test_initialization_methods.py
```

### Integrate Solution
```python
from temp.simple_best_method import weighted_probability_initialization

# Use in NSGA-II or MOEA/D
population = weighted_probability_initialization(rel_matrix, main_idx)
```

## Key Results
- **NumPy**: Found scipy, matplotlib, pandas (85.7% accuracy)
- **Flask**: Found werkzeug, jinja2, click (100% accuracy)
- **Pandas**: Found numpy, scipy, matplotlib (85.7% accuracy)

## Documentation
- `constructive.md` - Complete method documentation
- `CLAUDE.md` - Project memory and status

## Status
✅ Solution validated with real data
⏳ Ready for integration into NSGA-II and MOEA/D