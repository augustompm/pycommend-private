# PyCommend - Python Library Recommendation System

## Project Structure

```
pycommend-code/
├── src/
│   ├── optimizer/
│   │   └── nsga2.py          # NSGA-II implementation (baseline)
│   ├── preprocessor/
│   │   ├── create_distance_matrix.py
│   │   ├── create_sparse_distance_matrix.py
│   │   └── package_similarity.py
│   └── collector/           # (to be implemented)
├── data/
│   ├── PyPI/               # PyPI package data
│   ├── github/             # GitHub dependencies
│   ├── package_relationships_10k.pkl
│   ├── package_similarity_matrix_10k.pkl
│   ├── package_embeddings_10k.pkl
│   └── package_relationship_matrix.csv
├── results/                # Algorithm outputs
└── temp/                   # Temporary files and tests
    ├── tests/              # Test scripts
    ├── old-dependencies/   # Archive of old data
    └── readmes/            # README files from projects
```

## Objectives

1. **F1 - Maximize Linked Usage**: Co-occurrence in real projects
2. **F2 - Maximize Semantic Similarity**: Topical coherence
3. **F3 - Minimize Set Size**: Keep recommendations concise

## Current Status

- ✓ Data collection infrastructure
- ✓ NSGA-II baseline implementation
- ✓ Matrix creation for relationships and similarity
- ✗ VNS algorithm implementation (pending)
- ✗ Complete 24,000 GitHub projects dataset

## Usage

```bash
# Run NSGA-II recommender
python src/optimizer/nsga2.py <package_name> --pop_size 100 --generations 100

# Create relationship matrix
python src/preprocessor/create_distance_matrix.py

# Generate similarity matrix
python src/preprocessor/package_similarity.py
```

## Next Steps

1. Implement MOVNS (Multi-Objective VNS) algorithm
2. Complete data collection to 24,000 projects
3. Add performance metrics (Hypervolume, Spread, ε-indicator)
4. Comparative analysis NSGA-II vs MOVNS