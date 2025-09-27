# Data Files

This directory contains preprocessed matrices and datasets for PyCommend.

## File Structure

### Available in Repository (< 100MB)
- `package_relationships_10k.pkl` (19MB) - Co-usage matrix from 12,765 GitHub projects
- `package_embeddings_10k.pkl` (15MB) - Package embeddings for semantic similarity
- `package_relationship_matrix.csv` - Human-readable co-usage matrix
- `github/candidates_api.csv` - List of 23,001 GitHub projects
- `github/dependencies/` (30MB) - 12,765 parsed dependency files

### Large Files (Require Separate Download)
Due to GitHub's 100MB file limit, these files need to be downloaded separately:

- `package_similarity_matrix_10k.pkl` (763MB) - Semantic similarity matrix
- `PyPI/dependencies_10000.json` (81MB) - PyPI dependency graph
- `PyPI/top_10000_packages.json` (74MB) - Top 10k packages metadata

## Download Options

### Option 1: Manual Download
Download from [Google Drive link] or [Release Assets]

### Option 2: Use Git LFS
```bash
git lfs pull
```

### Option 3: Generate from Source
```python
# Generate similarity matrix
python src/preprocessor/package_similarity.py

# Generate relationship matrix
python src/preprocessor/create_distance_matrix.py
```

## Data Statistics

- **PyPI Packages**: 10,000 top packages by downloads
- **GitHub Projects**: 12,765 successfully parsed (55.5% of 23,001)
- **Matrix Dimensions**: 10,000 x 10,000
- **Total Dependencies Tracked**: ~500,000 relationships

## File Formats

### Pickle Files (.pkl)
Binary format for Python objects:
```python
import pickle
with open('package_relationships_10k.pkl', 'rb') as f:
    data = pickle.load(f)
    matrix = data['matrix']
    package_names = data['package_names']
```

### CSV Files
Human-readable format for analysis:
```python
import pandas as pd
df = pd.read_csv('package_relationship_matrix.csv', index_col=0)
```

### JSON Files
Metadata and dependency information:
```python
import json
with open('PyPI/top_10000_packages.json', 'r') as f:
    packages = json.load(f)
```

## Memory Requirements

- Minimum RAM: 4GB (without similarity matrix)
- Recommended RAM: 8GB (with all matrices loaded)
- Disk Space: ~1GB for all data files

## Updating Data

To update with new GitHub projects:
1. Update `github/candidates_api.csv`
2. Run collection script (see pycommend-collect/)
3. Regenerate matrices using preprocessor scripts