# Search-based Software Library Recommendation using Multi-objective Optimization

**Authors**: Ali Ouni, Raula Gaikovina Kula, Marouane Kessentini, Takashi Ishio, Daniel M. German, Katsuro Inoue

**Published**: Information and Software Technology, Volume 83, 2017, Pages 55-75

## Summary

This paper introduces **LibFinder**, a novel multi-objective search-based approach for software library recommendation to help developers find useful third-party libraries during software maintenance and evolution.

### Key Contributions

1. **Multi-Objective Formulation**: Uses NSGA-II to optimize three objectives:
   - **Maximize Linked Usage (LU)**: Co-usage between candidate library and existing system libraries
   - **Maximize Semantic Similarity (SS)**: Similarity between library and system source code identifiers
   - **Minimize Recommendation Set Size (RSS)**: Number of recommended libraries

2. **Dataset**:
   - 6,083 Maven libraries
   - 32,760 GitHub client systems
   - Extracted identifiers and usage history from both repositories

3. **Methodology**:
   - Combines Mining Software Repositories (MSR) with Search-Based Software Engineering (SBSE)
   - Uses library co-occurrence patterns ("wisdom of the crowd")
   - Employs semantic similarity based on code identifiers
   - Solution representation: chromosome where each gene represents a candidate library

### Experimental Results

- **Accuracy**: 92% at top-10 recommendations
- **Precision**: 51% at top-10
- **Recall**: 68% at top-10
- Significantly outperforms:
  - Random search
  - MOEA/D and IBEA algorithms
  - LibRec (state-of-the-art approach)
- Validated on two industrial Java systems from Ford Motor Company
  - Average developer rating: 3.25 out of 5

### Technical Details

**Solution Encoding**:
- Chromosome length = number of classes in system
- Each gene = candidate library or "NONE"
- Libraries ranked by frequency in solution

**Fitness Functions**:
1. LU: Measures co-usage frequency between libraries
2. SS: Cosine similarity between library and class identifiers
3. RSS: Count of unique libraries in solution

**Constraints**:
- Cannot recommend already-used libraries
- Avoids recommending similar libraries (Jaccard > 0.8)

### Industrial Validation

Evaluated with 8 Ford developers on two systems:
- **JDI-Ford**: 638 classes, 11 libraries
- **DROI-Ford**: 786 classes, 19 libraries

Top-rated recommendations:
- Quartz (4.25/5): Job scheduling
- Mahout-math (3.75/5): Scientific computing
- Guava (3.75/5): Core utilities

### Key Insights

- Library recommendation requires balancing usage history with content similarity
- Developers prefer libraries that:
  - Replace buggy code (29%)
  - Improve quality (23.5%)
  - Add new features (23.5%)
- MOEA/D is not elitist - can lose good solutions due to decomposition
- Small populations (30) lead to HV oscillation but enable fast recommendation

### Limitations

- Only tested on Java/Maven ecosystem
- Precision/recall assume non-dropped libraries are uninteresting
- Computational cost increases with system size
- Parameter sensitivity requires tuning

### Future Work

- Eclipse plugin for on-the-fly recommendations
- Consider change history for active classes
- Library version and quality assessment
- Parallel implementation for scalability