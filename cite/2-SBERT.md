# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Authors:** Nils Reimers and Iryna Gurevych

**Institution:** Ubiquitous Knowledge Processing Lab (UKP-TUDA), Department of Computer Science, Technische Universität Darmstadt

**Published:** arXiv:1908.10084v1 [cs.CL] 27 Aug 2019

**Code:** https://github.com/UKPLab/sentence-transformers

## Abstract

BERT and RoBERTa have set new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS). However, they require both sentences to be fed into the network, causing massive computational overhead: finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT.

This paper presents **Sentence-BERT (SBERT)**, a modification of the pretrained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT/RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.

SBERT outperforms other state-of-the-art sentence embeddings methods on common STS tasks and transfer learning tasks.

## Problem Statement

### BERT Limitations for Sentence Similarity
- **Cross-encoder architecture:** Two sentences passed to transformer, target value predicted
- **Unsuitable for large-scale comparison:** Finding most similar pair in n=10,000 sentences requires n·(n-1)/2 = 49,995,000 inference computations
- **Time complexity:** ~65 hours on modern V100 GPU for 10,000 sentences
- **Poor sentence embeddings:** Averaging BERT output or using CLS token yields embeddings worse than GloVe

## SBERT Architecture

### Model Structure

SBERT adds a **pooling operation** to BERT/RoBERTa output to derive fixed-sized sentence embeddings.

**Pooling Strategies (tested):**
- **CLS-token:** Using output of [CLS] token
- **MEAN:** Computing mean of all output vectors (default)
- **MAX:** Computing max-over-time of output vectors

### Training Objectives

#### 1. Classification Objective Function
```
o = softmax(Wt(u, v, |u-v|))
```
- Concatenate sentence embeddings u and v with element-wise difference |u-v|
- Multiply with trainable weight Wt ∈ R^(3n×k)
- Optimize cross-entropy loss
- Used for training on NLI data

#### 2. Regression Objective Function
- Compute cosine-similarity between embeddings u and v
- Use mean-squared-error loss
- Used for training on STS benchmark data

#### 3. Triplet Objective Function
```
max(||sa - sp|| - ||sa - sn|| + ε, 0)
```
- Given anchor sentence a, positive p, and negative n
- Minimize loss to ensure p is closer to a than n
- Use Euclidean distance with margin ε = 1

### Training Details

**Dataset:** Combination of:
- SNLI (570,000 sentence pairs)
- Multi-Genre NLI (430,000 sentence pairs)

**Training Setup:**
- 3-way softmax classifier
- 1 epoch
- Batch size: 16
- Optimizer: Adam with learning rate 2e-5
- Linear learning rate warm-up over 10% of training data
- Default pooling: MEAN

## Evaluation Results

### Unsupervised STS Tasks

**Datasets:** STS 2012-2016, STS benchmark, SICK-Relatedness

**Metric:** Spearman's rank correlation (ρ)

| Model | Avg. Performance |
|-------|-----------------|
| Avg. GloVe embeddings | 61.32 |
| Avg. BERT embeddings | 54.81 |
| BERT CLS-vector | 29.19 |
| InferSent - GloVe | 65.01 |
| Universal Sentence Encoder | 71.22 |
| **SBERT-NLI-base** | **74.89** |
| **SBERT-NLI-large** | **76.55** |
| SRoBERTa-NLI-large | 76.68 |

**Key Finding:** Direct BERT output yields poor sentence embeddings (worse than GloVe). SBERT significantly improves performance.

### Supervised STS (STS Benchmark)

**Training Approaches:**
1. Train only on STSb
2. Train on NLI, then fine-tune on STSb (better)

**Results (Spearman correlation):**

| Model | Score |
|-------|-------|
| BERT-NLI-STSb-base | 88.33 ± 0.19 |
| SBERT-NLI-STSb-base | 85.35 ± 0.17 |
| BERT-NLI-STSb-large | 88.77 ± 0.46 |
| **SBERT-NLI-STSb-large** | **86.10 ± 0.13** |

Note: BERT slightly outperforms SBERT when fine-tuned on task-specific data, but SBERT provides faster inference.

### Argument Facet Similarity (AFS)

**Dataset:** 6,000 sentential argument pairs from social media on controversial topics

**Challenge:** Arguments must make similar claims AND provide similar reasoning

**Results (10-fold cross-validation):**

| Model | Spearman ρ |
|-------|-----------|
| SVR (Misra et al., 2016) | - |
| BERT-AFS-base | 74.84 |
| **SBERT-AFS-base** | **74.13** |
| BERT-AFS-large | 76.38 |
| **SBERT-AFS-large** | **75.93** |

**Cross-topic evaluation:** SBERT performance drops ~7 points compared to BERT, suggesting SBERT requires more training data for complex similarity notions.

### Wikipedia Sections Distinction

**Dataset:** ~1.8M training triplets from Wikipedia (Dor et al., 2018)
- Anchor and positive from same section
- Negative from different section of same article

**Metric:** Accuracy (is positive closer than negative?)

| Model | Accuracy |
|-------|----------|
| mean-vectors | 0.65 |
| skip-thoughts-CS | 0.62 |
| Dor et al. (BiLSTM) | 0.74 |
| **SBERT-WikiSec-large** | **0.8078** |
| **SRoBERTa-WikiSec-large** | **0.7973** |

### SentEval Transfer Learning Tasks

**Tasks:** MR, CR, SUBJ, MPQA, SST, TREC, MRPC

**Average Performance:**

| Model | Avg. Score |
|-------|-----------|
| Avg. GloVe embeddings | 81.52 |
| Avg. BERT embeddings | 84.94 |
| BERT CLS-vector | 84.66 |
| InferSent - GloVe | 85.59 |
| Universal Sentence Encoder | 85.10 |
| **SBERT-NLI-base** | **87.41** |
| **SBERT-NLI-large** | **87.69** |

**Key Finding:** SBERT achieves new state-of-the-art on SentEval, with +2.1 points improvement over InferSent and +2.6 points over Universal Sentence Encoder.

## Ablation Study

### Impact of Pooling Strategy

**On NLI classification:**
- MEAN: 80.78
- MAX: 79.07
- CLS: 79.80

**On STSb regression:**
- MEAN: 87.44
- MAX: 69.92 (significantly worse)
- CLS: 86.62

**Conclusion:** MEAN pooling performs best overall; MAX pooling particularly poor for regression.

### Impact of Concatenation Mode (Classification)

| Concatenation | Score (NLI) |
|---------------|-------------|
| (u, v) | 66.04 |
| (\|u-v\|) | 69.78 |
| (u * v) | 70.54 |
| (\|u-v\|, u * v) | 78.37 |
| (u, v, u * v) | 77.44 |
| **(u, v, \|u-v\|)** | **80.78** |
| (u, v, \|u-v\|, u * v) | 80.44 |

**Key Finding:** Element-wise difference |u-v| is the most important component for training.

## Computational Efficiency

**Setup:**
- CPU: Intel i7-5820K @ 3.30GHz
- GPU: Nvidia Tesla V100
- Task: Compute embeddings for STS benchmark sentences

**Performance (sentences per second):**

| Model | CPU | GPU |
|-------|-----|-----|
| Avg. GloVe embeddings | 6,469 | - |
| InferSent | 137 | 1,876 |
| Universal Sentence Encoder | 67 | 1,318 |
| SBERT-base | 44 | 1,378 |
| **SBERT-base (smart batching)** | **83** | **2,042** |

**Smart Batching:** Groups sentences with similar lengths, reduces padding overhead
- 89% speedup on CPU
- 48% speedup on GPU
- 9% faster than InferSent on GPU
- 55% faster than Universal Sentence Encoder on GPU

## Use Cases Enabled by SBERT

### Semantic Search
- **Problem:** Finding similar questions in 40M Quora questions would take 50+ hours with BERT
- **Solution:** SBERT embeddings + optimized index structures → milliseconds
- **Approach:** Compute embeddings once, use cosine-similarity or optimized search (Johnson et al., 2017)

### Clustering
- **Problem:** Clustering 10,000 sentences with hierarchical clustering requires ~50M comparisons
- **BERT:** ~65 hours
- **SBERT:** ~5 seconds (compute 10,000 embeddings) + clustering

### Large-Scale Similarity Comparison
- Reduce from O(n²) comparisons to O(n) embedding computations
- Enable real-time applications

## Key Contributions

1. **SBERT Architecture:** Siamese/triplet network modification of BERT for sentence embeddings
2. **Significant Performance Improvement:**
   - +11.7 points over InferSent on STS tasks
   - +5.5 points over Universal Sentence Encoder
   - New state-of-the-art on SentEval
3. **Computational Efficiency:**
   - 65 hours → 5 seconds for 10,000 sentence similarity task
   - Enables previously infeasible applications
4. **Comprehensive Evaluation:** Multiple datasets, ablation studies, transfer learning

## Important Findings

### What Works
- Training on NLI data creates good general-purpose sentence embeddings
- MEAN pooling strategy is most robust
- Element-wise difference |u-v| crucial for classification training
- Smart batching significantly improves throughput

### What Doesn't Work
- Direct BERT output (averaging embeddings or CLS token) poor for cosine-similarity
- MAX pooling poor for regression tasks
- Adding element-wise product u*v decreases performance

### BERT vs SBERT Trade-offs
- **BERT:** Better for cross-encoder tasks with task-specific fine-tuning
- **SBERT:** Better for generating semantic sentence embeddings, clustering, semantic search
- **Performance:** BERT slightly better on supervised tasks, SBERT enables new use cases

## Limitations and Future Work

### Identified Limitations
1. **Complex Similarity Notions:** SBERT requires more training data than BERT for tasks requiring nuanced understanding (e.g., argument similarity)
2. **Cross-Topic Generalization:** Performance drops in cross-topic scenarios
3. **Not Optimal for Transfer Learning:** BERT fine-tuning preferred for new classification tasks

### Not Mentioned Limitations
- Primarily evaluated on English
- Computational cost still higher than simple embedding averaging
- Requires substantial training data

## Related Work Comparison

| Method | Approach | Training Data | Performance |
|--------|----------|--------------|-------------|
| Skip-Thought | Encoder-decoder | Unsupervised book corpus | Baseline |
| InferSent | BiLSTM max-pooling | SNLI + MultiNLI | 65.01 avg STS |
| Universal Sentence Encoder | Transformer | SNLI + other | 71.22 avg STS |
| **SBERT** | **Siamese BERT** | **SNLI + MultiNLI** | **74.89-76.55 avg STS** |

## Implementation Details

**Code Available:** https://github.com/UKPLab/sentence-transformers

**Training Time:** Less than 20 minutes for fine-tuning

**Model Sizes:**
- SBERT-base: 110M parameters
- SBERT-large: 340M parameters

**Supports:**
- Multiple languages (multilingual BERT)
- Custom fine-tuning on domain-specific data
- Both PyTorch and TensorFlow

## Conclusion

SBERT successfully addresses BERT's limitations for semantic similarity tasks by:
1. Generating meaningful sentence embeddings via siamese network architecture
2. Enabling efficient similarity computation via cosine-similarity
3. Maintaining high accuracy while reducing computational cost by orders of magnitude
4. Outperforming previous state-of-the-art sentence embedding methods

The method makes previously computationally infeasible tasks (large-scale semantic search, clustering) practical and efficient.
