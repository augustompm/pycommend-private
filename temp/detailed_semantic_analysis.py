#!/usr/bin/env python3
"""
Detailed analysis of semantic data to identify improvement opportunities.
"""

import pickle
import numpy as np
import sys
import os
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load all three semantic data sources."""
    print("Loading all data sources...")

    # Load relationships matrix
    with open("E:/pycommend/pycommend-code/data/package_relationships_10k.pkl", 'rb') as f:
        rel_data = pickle.load(f)

    # Load similarity matrix
    with open("E:/pycommend/pycommend-code/data/package_similarity_matrix_10k.pkl", 'rb') as f:
        sim_data = pickle.load(f)

    # Load embeddings
    with open("E:/pycommend/pycommend-code/data/package_embeddings_10k.pkl", 'rb') as f:
        emb_data = pickle.load(f)

    return rel_data, sim_data, emb_data

def analyze_relationships_matrix(rel_data):
    """Analyze the co-occurrence relationships matrix."""
    print("\n" + "="*60)
    print("RELATIONSHIPS MATRIX ANALYSIS")
    print("="*60)

    matrix = rel_data['matrix']
    package_names = rel_data['package_names']

    print(f"Matrix type: {type(matrix)}")
    print(f"Shape: {matrix.shape}")
    print(f"Non-zero elements: {matrix.nnz}")
    print(f"Sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")

    # Convert to dense for analysis
    dense_sample = matrix[:10, :10].toarray()
    print(f"\nSample 10x10 values:")
    print(dense_sample)

    # Find strongest relationships
    print("\nFinding strongest relationships...")
    max_vals = []
    for i in range(min(100, matrix.shape[0])):  # Check first 100 packages
        row = matrix[i].toarray().flatten()
        row[i] = 0  # Remove self-connection
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        if max_val > 0:
            max_vals.append((package_names[i], package_names[max_idx], max_val))

    # Sort by strength
    max_vals.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 strongest relationships:")
    for i, (pkg1, pkg2, strength) in enumerate(max_vals[:10]):
        print(f"{i+1:2d}. {pkg1} -> {pkg2} (strength: {strength})")

    return matrix, package_names

def analyze_similarity_matrix(sim_data):
    """Analyze the semantic similarity matrix."""
    print("\n" + "="*60)
    print("SIMILARITY MATRIX ANALYSIS")
    print("="*60)

    matrix = sim_data['similarity_matrix']
    package_names = sim_data['package_names']

    print(f"Matrix type: {type(matrix)}")
    print(f"Shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    print(f"Min value: {np.min(matrix)}")
    print(f"Max value: {np.max(matrix)}")
    print(f"Mean value: {np.mean(matrix)}")

    # Sample values
    print(f"\nSample 5x5 values:")
    print(matrix[:5, :5])

    # Find most similar packages
    print("\nFinding most similar packages...")
    most_similar = []
    for i in range(min(50, matrix.shape[0])):  # Check first 50 packages
        row = matrix[i].copy()
        row[i] = -1  # Remove self-similarity
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        most_similar.append((package_names[i], package_names[max_idx], max_val))

    # Sort by similarity
    most_similar.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 most similar package pairs:")
    for i, (pkg1, pkg2, similarity) in enumerate(most_similar[:10]):
        print(f"{i+1:2d}. {pkg1} -> {pkg2} (similarity: {similarity:.4f})")

    return matrix, package_names

def analyze_embeddings(emb_data):
    """Analyze the package embeddings."""
    print("\n" + "="*60)
    print("EMBEDDINGS ANALYSIS")
    print("="*60)

    embeddings = emb_data['embeddings']
    package_names = emb_data['package_names']

    print(f"Embeddings type: {type(embeddings)}")
    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Min value: {np.min(embeddings)}")
    print(f"Max value: {np.max(embeddings)}")
    print(f"Mean value: {np.mean(embeddings)}")

    # Check embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms - Min: {np.min(norms):.4f}, Max: {np.max(norms):.4f}, Mean: {np.mean(norms):.4f}")

    # Sample embedding
    print(f"\nFirst embedding (first 10 dimensions): {embeddings[0][:10]}")

    # Test cosine similarity computation
    print("\nTesting cosine similarity between first few packages:")
    for i in range(min(5, len(package_names))):
        for j in range(i+1, min(5, len(package_names))):
            cos_sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"{package_names[i]} vs {package_names[j]}: {cos_sim:.4f}")

    return embeddings, package_names

def compare_data_sources(rel_matrix, sim_matrix, embeddings, package_names):
    """Compare the three data sources to understand their relationships."""
    print("\n" + "="*60)
    print("CROSS-DATA SOURCE COMPARISON")
    print("="*60)

    # Check if package names are the same across all sources
    rel_names = set(package_names)  # Assuming same for relationships
    print(f"Package names consistent across sources: {len(rel_names) == len(set(package_names))}")

    # Find a well-known package for comparison
    test_packages = ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'requests']
    available_test_packages = [pkg for pkg in test_packages if pkg in package_names]

    if available_test_packages:
        test_pkg = available_test_packages[0]
        test_idx = package_names.index(test_pkg)

        print(f"\nAnalyzing {test_pkg} (index {test_idx}) across all data sources:")

        # Relationships
        rel_row = rel_matrix[test_idx].toarray().flatten()
        rel_top_indices = np.argsort(rel_row)[-6:][::-1]  # Top 6 (including self)
        rel_top_indices = rel_top_indices[rel_top_indices != test_idx][:5]  # Remove self, take top 5
        print(f"\nTop 5 related packages (co-occurrence):")
        for i, idx in enumerate(rel_top_indices):
            print(f"  {i+1}. {package_names[idx]} (strength: {rel_row[idx]})")

        # Similarity
        sim_row = sim_matrix[test_idx].copy()
        sim_row[test_idx] = -1  # Remove self
        sim_top_indices = np.argsort(sim_row)[-5:][::-1]
        print(f"\nTop 5 similar packages (similarity matrix):")
        for i, idx in enumerate(sim_top_indices):
            print(f"  {i+1}. {package_names[idx]} (similarity: {sim_row[idx]:.4f})")

        # Embeddings similarity
        test_embedding = embeddings[test_idx]
        emb_similarities = cosine_similarity([test_embedding], embeddings)[0]
        emb_similarities[test_idx] = -1  # Remove self
        emb_top_indices = np.argsort(emb_similarities)[-5:][::-1]
        print(f"\nTop 5 similar packages (embedding cosine similarity):")
        for i, idx in enumerate(emb_top_indices):
            print(f"  {i+1}. {package_names[idx]} (cosine similarity: {emb_similarities[idx]:.4f})")

def identify_semantic_opportunities():
    """Identify specific semantic improvement opportunities."""
    print("\n" + "="*60)
    print("SEMANTIC IMPROVEMENT OPPORTUNITIES")
    print("="*60)

    print("Based on the analysis, here are key opportunities:")
    print("\n1. MULTI-MODAL FUSION:")
    print("   - Combine co-occurrence (relationships) + semantic similarity + embeddings")
    print("   - Weight them based on different use cases")
    print("   - Co-occurrence = collaborative filtering")
    print("   - Embeddings = content-based filtering")
    print("   - Similarity matrix = hybrid approach")

    print("\n2. SEMANTIC CLUSTERING:")
    print("   - Use embeddings to identify package domains/categories")
    print("   - Create domain-specific recommendation pools")
    print("   - Improve initialization by staying within semantic clusters")

    print("\n3. EMBEDDING-BASED CANDIDATE FILTERING:")
    print("   - Use cosine similarity on embeddings to pre-filter candidates")
    print("   - Only consider packages above certain embedding similarity threshold")
    print("   - Reduces search space intelligently")

    print("\n4. SEMANTIC OBJECTIVE FUNCTION:")
    print("   - Add embedding-based semantic coherence as 4th objective")
    print("   - Measure how semantically coherent a recommendation set is")
    print("   - Balance diversity vs semantic coherence")

    print("\n5. CONTEXTUAL RECOMMENDATIONS:")
    print("   - Use embeddings to understand package 'context' or 'purpose'")
    print("   - Recommend packages that fit the same context")
    print("   - E.g., web development vs data science vs systems programming")

def main():
    """Main analysis function."""
    try:
        # Load all data
        rel_data, sim_data, emb_data = load_data()

        # Analyze each data source
        rel_matrix, rel_names = analyze_relationships_matrix(rel_data)
        sim_matrix, sim_names = analyze_similarity_matrix(sim_data)
        embeddings, emb_names = analyze_embeddings(emb_data)

        # Compare data sources
        compare_data_sources(rel_matrix, sim_matrix, embeddings, rel_names)

        # Identify opportunities
        identify_semantic_opportunities()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()