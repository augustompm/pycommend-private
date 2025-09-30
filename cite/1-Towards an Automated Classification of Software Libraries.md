# Towards an Automated Classification of Software Libraries

**Authors:** Maximilian Auch, Maximilian Balluf, Peter Mandl, Christian Wolf

**Institution:**
- University of Applied Sciences Munich, Lothstraße 34, 80335 Munich, Germany
- University of Regensburg, Universitätsstraße 31, 93053 Regensburg, Germany

**Published:** SN Computer Science (2024) 5:339

**DOI:** https://doi.org/10.1007/s42979-024-02654-2

## Abstract

The use of third-party libraries in software is common. At the same time, the number of published libraries continues to increase. An automated classification should help to maintain an overview and identify similar software libraries. This paper investigates if new approaches can be used to classify all software libraries crawled from Apache Maven repositories into defined classes using machine learning.

In addition to tags that are not always available or of poor quality, we examine one feature that is always available—the id. Consisting of group-id and artifact-id, the id of an Apache Maven software library contains valuable information that can help in classification. Through a developed preprocessing and an optimized recurrent neural network (RNN), the tokenised ids should allow a classification of most libraries.

Furthermore, we present an optimized approach through a hybrid use of id tokens and tags in combination. Based on the dataset including 28,600 labeled entries, a comparison of various approaches was carried out.

## Key Results

- **RNN on tokenised ids**: 71.36% balanced accuracy
- **Model trained on tags**: 92% balanced accuracy
- **Hybrid approach (tags + ids)**: **94.12% balanced accuracy** (best)

While a classification on tags achieves a better result than the more general id-based approach, the applicability is limited to software libraries that are tagged. The hybrid approach takes advantage of the classification results based on tags when these are available, but includes valuable information from the always available ids.

## Problem Statement

### Challenges in Library Organization

**Growth of Software Libraries:**
- Maven Central contains 300,000+ libraries
- 93.3% of software projects use third-party libraries (Thung et al., 2013)
- Average of 28 third-party libraries per project
- Need for automated classification to maintain overview

**Limitations of Manual Classification:**
- Not all libraries are tagged on MvnRepository.com
- Tags are often missing or only a single tag is available
- Quality of tags varies significantly
- Manual categorization not feasible for large repositories

## Dataset

### Data Collection (May-July 2020)

**Sources:**
- Maven Central: 313,122 libraries
- Sonatype: 681 libraries
- Spring IO: 35,434 libraries
- Atlassian: 1,180 libraries
- Hortonworks: 1,716 libraries
- Wso2: 1,448 libraries

**Total:** 325,000 unique libraries after removing duplicates

### Dataset Characteristics

**Tagged Libraries:** ~75% (250,000 libraries)
- 437 unique tags extracted
- Libraries usually have up to 7 tags
- Few outliers with up to 13 tags

**Categorized Libraries:** ~11% (about 28,600 libraries)
- 162 original categories from MvnRepository.com
- Mapped to 69 coarse-grained classes for classification
- Both tagged and categorized

**Tag Distribution:**
- Higher percentage of uncategorized libraries have single tag
- Few libraries have more than 3 tags
- Uncategorized data contains same tags as categorized data

### Classes (69 total after mapping)

**Major Categories:**
- Build Automation Tool Plugins (17%)
- Example (16%)
- Web Applications (15%)
- Testing (6%)
- File Handler (4%)
- UI (4%)
- Logging / Monitoring (4%)
- Database (3%)
- IDE Modules (3%)
- 60 other classes (28%)

**Data Imbalance:**
- Top 4 classes contain over 50% of labeled data
- Handled through macro-averaging metrics
- Balanced accuracy used instead of regular accuracy

## Approaches for Classification

### 1. Tag-Based Classification (Baseline)

**Method:**
- Uses manually assigned tags from MvnRepository.com
- Feedforward Neural Network (FNN)
- Best prior approach from previous study

**Limitations:**
- Only applicable to tagged libraries (~75% of dataset)
- Tag quality varies significantly
- Some tags are irrelevant across classes

**Excluded Tags:**
- "github", "codehaus", "apache" (hosting platforms)
- "experimental", "starter", "runner", "api", "bom" (generic status/structure)

**Results:** 92% balanced accuracy

### 2. Id-Based Classification (Novel Approach)

**Motivation:**
- **group-id + artifact-id** always available for all libraries
- Based on Java package naming conventions
- Contains semantic information about library function

**Example:**
```
org.mariadb.jdbc:mariadb-java-client
```

**Preprocessing Pipeline:**

1. **Replace special characters and remove numbers**
2. **Split group-id and artifact-id into tokens**
3. **Split camel case words** (e.g., "mariaDb" → "maria", "db")
4. **Convert to lowercase**
5. **Remove blacklisted tokens:**
   - Hosting platforms: "googlecode", "codehaus", "github"
   - Foundations: "apache", "eclipse"
   - Versioning tools: "git", "svn"
   - Top-level domains (with exceptions for domain abbreviations like "ai", "ml")

6. **Lemmatization** using spaCy
   - "embedding", "embedded" → "embed"

7. **Split concatenated tokens:**
   - "h2database" → "h2", "database"
   - "mariadb" → "maria", "db"
   - "dynamodb" → "dynamo", "db"

8. **Remove duplicates** in tokens for each library

**Token Analysis:**
- Common tokens: Present in multiple libraries
- Rare tokens: Below certain occurrence threshold
- Dictionary-based split using:
  - 125,000 words from Wikipedia
  - Domain-specific dictionary from common tokens

**Architecture: Bidirectional LSTM RNN**

```
Input: Tokenized ids (variable length)
↓
Encoding Layer
↓
Embedding Layer (word embeddings)
↓
Bidirectional LSTM (100 neurons)
↓
Dense Layer (100 neurons, ReLU activation)
↓
Dropout Layer (20% frequency)
↓
Output Layer (69 neurons, Softmax)
```

**Hyperparameter Tuning:**
- Tested: Unidirectional/Bidirectional LSTM and GRU
- Selected: Bidirectional LSTM
- Neuron counts and dropout optimized experimentally

**Results:**
- Base (no filtering): **71.36% balanced accuracy**
- With rare token split (≥3 occurrences): 75.11% balanced accuracy
- With rare token split (≥5 occurrences): **76.83% balanced accuracy**
- With rare token split (≥10 occurrences): 75.46% balanced accuracy

### 3. Hybrid Classification (Proposed - Best Results)

**Approach:**
- Combines tags AND tokenized ids in same RNN architecture
- Uses tags when available
- Supplements with id tokens for poor-quality or missing tags

**Rationale:**
- Tags provide high-quality classification when present
- Id tokens compensate for:
  - Missing tags
  - Poor tag quality
  - Generic tags used across classes

**Results:** **94.12% balanced accuracy** (best overall)

## Experimental Setup

### Evaluation

**Split:**
- Training/Validation: 80% (further split 80/20)
- Test: 20%
- Random stratified split

**Metrics:**
- **Balanced Accuracy**: Average of recall for each class (handles imbalanced data)
- **Macro-averaged Precision**: Precision calculated per class, then averaged
- **Macro-averaged Recall**: Recall calculated per class, then averaged
- **Macro-averaged F1-Score**

**Why Macro-averaging?**
- Prevents overrepresented classes from dominating metrics
- Each class weighted equally regardless of size
- More informative for imbalanced datasets

**Statistical Testing:**
- Friedman test (p-value: 0.0129 < 0.05)
- Nemenyi post-hoc test for pairwise comparisons
- 5 independent runs per configuration

### Computational Environment

- OS: Ubuntu 22.04.1 LTS
- GPU: NVIDIA GeForce GTX 1080
- CPU: AMD Ryzen 9 3900XT
- RAM: 64 GiB

**Training Times (mean ± std):**
- Tag-based (FNN): 15.4s ± 1.83s
- Tag-based (RNN): 32.2s ± 5.6s (2× FNN)
- Id-based (RNN): 26.8s ± 0.45s
- Hybrid (RNN): 45.9s ± 6.29s (largest input)

## Detailed Results

### Overall Performance Comparison

| Approach | Model | Acc | Bal_Acc | Prec_M | F1_M |
|----------|-------|-----|---------|--------|------|
| Tag-based | FNN | 97.48% | **92.00%** | 95.02% | 92.91% |
| Tag-based | RNN | 87.00% | 76.57% | 94.25% | 83.14% |
| Id-based | RNN | 90.41% | **71.36%** | 79.49% | 74.03% |
| Hybrid | RNN | **98.33%** | **94.12%** | **96.20%** | **94.87%** |

### Statistical Significance (Nemenyi Test p-values)

|  | Tag (FNN) | Tag (RNN) | Id (RNN) | Hybrid |
|--|-----------|-----------|----------|--------|
| Tag (FNN) | - | 0.3549 | 0.3549 | 0.6703 |
| Tag (RNN) | 0.3549 | - | 0.9000 | **0.0314** |
| Id (RNN) | 0.3549 | 0.9000 | - | **0.0314** |
| Hybrid | 0.6703 | **0.0314** | **0.0314** | - |

**Interpretation:**
- Hybrid significantly better than Tag-RNN and Id-RNN (p=0.0314 < 0.05)
- No significant difference between Tag-FNN and Hybrid (p=0.6703)
- Hybrid shows best overall performance with statistical support

### Confusion Matrix Analysis

**Observations:**
- Tag-based and Id-based models tend to over-predict "Web Applications" class
- "Web Applications" is one of the over-represented classes (15% of data)
- Hybrid approach reduces this bias
- Better distribution across all 69 classes in Hybrid model

### Optimization Analysis (Id-based with varying base_x)

| Base | Classes | Acc | Bal_Acc | Prec_M | F1_M |
|------|---------|-----|---------|--------|------|
| base₁ | 69 | 90.41% | 71.36% | 79.49% | 74.03% |
| base₃ | 69 | 91.72% | 75.11% | **80.57%** | 77.03% |
| base₅ | 69 | **92.12%** | **76.83%** | 79.50% | **77.19%** |
| base₁₀ | 68 | 91.64% | 75.46% | 80.46% | 76.88% |

**Where base_x = minimum token occurrences to be considered "common"**

**Insights:**
- Balanced accuracy improves from 71.36% → 76.83% with optimal filtering
- base₅ provides best trade-off
- base₁₀ starts losing classes (69 → 68) due to aggressive filtering
- Optimal filtering depends on use case requirements

## Key Findings

### Strengths of Each Approach

**Tag-Based:**
- ✅ Highest accuracy when tags are available (92%)
- ✅ Manually curated information
- ❌ Only applicable to ~75% of libraries
- ❌ Tag quality varies

**Id-Based:**
- ✅ Applicable to 100% of libraries (always available)
- ✅ No dependency on manual tagging
- ✅ Contains semantic information
- ❌ Lower accuracy (71.36%) due to naming inconsistencies
- ❌ Requires extensive preprocessing

**Hybrid (Best):**
- ✅ Highest overall accuracy (94.12%)
- ✅ Combines advantages of both approaches
- ✅ Compensates for missing/poor tags with id tokens
- ✅ Applicable to all libraries
- ❌ Slightly longer training time (45.9s vs 32.2s)

### Preprocessing Impact

**Critical Preprocessing Steps:**
1. **Camel case splitting**: Essential for extracting semantic tokens
2. **Lemmatization**: Reduces vocabulary, improves generalization
3. **Token filtering**: Removes noise from hosting platforms, foundations
4. **Concatenated token splitting**: Extracts domain keywords (e.g., "database" from "h2database")

**Without preprocessing:**
- Accuracy drops significantly (83.30% → 90.41%)
- Balanced accuracy: 66.04% → 71.36%

### Imbalanced Data Handling

**Strategy:**
- Use balanced accuracy instead of regular accuracy
- Macro-averaging for Precision, Recall, F1
- No oversampling/undersampling (avoids introducing biases)

**Validation:**
- ANOVA confirms statistical significance (p < 0.001)
- Tukey confidence intervals show non-overlapping ranges
- Models generalize across all 69 classes despite imbalance

## Comparison with Related Work

| Study | Target | Data Source | Method | Result |
|-------|--------|-------------|--------|--------|
| Escobar-Avila (2015) | Multi-label categorization | Bytecode (158 libs) | Clustering | ~40% precision |
| Yu et al. (2017) | App categorization | Descriptions + libs | Collaborative filtering | - |
| Velázquez-Rodríguez & De Roover (2020) | Multi-label tagging | Tags + word vectors | MEKA classifiers | - |
| **This study (2024)** | **Multi-class classification** | **Tags + ids (28,600 libs)** | **Hybrid RNN** | **94.12% bal_acc** |

**Key Differences:**
- Larger dataset (28,600 vs 158-3,000)
- Hybrid approach combines multiple features
- Focus on multi-class (single label) rather than multi-label
- Higher accuracy with statistical validation

## Limitations and Future Work

### Current Limitations

**1. Ecosystem-Specific:**
- Evaluated only on JVM/Maven libraries
- Id structure may differ in other ecosystems (PyPI, npm, CRAN, RubyGems)
- Generalization to other languages unknown

**2. Class Structure:**
- Based on MvnRepository.com categories
- 69 classes may not be optimal granularity
- Some libraries have cross-class functionality

**3. Multi-label vs Single-label:**
- Current approach: single label per library
- Reality: some libraries fit multiple categories
- Example: `javax.inject:javax.inject` could be both "Dependency Injection" and "Java Specifications"

**4. Token Quality:**
- 10% of uncategorized libraries have no common tokens
- May require manual review or extended training set

### Future Directions

**1. Cross-Language Study:**
- Apply to PyPI, npm, CRAN, RubyGems
- Compare id structures across ecosystems
- Develop language-agnostic approaches

**2. Multi-label Classification:**
- Allow libraries to belong to multiple categories
- More realistic representation of library functionality

**3. Dataset Enrichment:**
- Manual annotation of libraries with rare tokens
- Balance underrepresented classes
- Extend to more repositories

**4. Feature Engineering:**
- Include additional features (bytecode, README, dependencies)
- Explore transformer-based models (BERT for code)
- Investigate attention mechanisms for token importance

**5. Dynamic Classification:**
- Handle library evolution and version changes
- Continuous learning as new libraries are published

## Practical Applications

### Use Cases

**1. Repository Management:**
- Automatic categorization of new libraries
- Maintain organized library catalogs
- Improve search and discovery

**2. Software Architecture Analysis:**
- Identify technical stack from library dependencies
- Detect technology migrations (library A → library B)
- Calculate technical similarity between projects

**3. Recommendation Systems:**
- Suggest similar libraries based on classification
- Recommend complementary libraries in same category
- Support library selection decisions

**4. Security and Compliance:**
- Group libraries by security risk categories
- Track usage of deprecated library types
- Identify licensing patterns by category

## Conclusion

This study demonstrates that **automated classification of software libraries is feasible** using machine learning approaches. The key contributions are:

1. **Id-based classification** provides a general solution applicable to all libraries (71.36% balanced accuracy)

2. **Tag-based classification** achieves high accuracy but limited applicability (92% balanced accuracy, ~75% coverage)

3. **Hybrid approach** combines the best of both worlds:
   - **94.12% balanced accuracy** (best result)
   - Applicable to all libraries
   - Statistically significant improvement

4. **Comprehensive preprocessing pipeline** for id tokenization is crucial for extracting semantic information

5. **Large-scale validation** on 28,600 libraries across 69 classes demonstrates robustness

The hybrid approach proves most effective by leveraging high-quality tag information when available while falling back to always-available id tokens for untagged or poorly tagged libraries. This makes it a practical solution for real-world library classification systems.

**Code and Data Availability:**
- GitHub: https://github.com/CCWI/corpus-libsim
- Extended dataset: https://github.com/CCWI/corpus-libsim-extended
