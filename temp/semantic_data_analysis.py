#!/usr/bin/env python3
"""
Analyze the semantic data available in PyCommend to find improvement opportunities.
"""

import pickle
import numpy as np
import sys
import os

def analyze_pkl_file(filepath, name):
    """Analyze a pickle file and print its structure."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Type: {type(data)}")

        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
            print(f"Data type: {data.dtype}")

            if hasattr(data, 'nnz'):
                print(f"Non-zero elements: {data.nnz}")
                print(f"Sparsity: {(1 - data.nnz / (data.shape[0] * data.shape[1])) * 100:.2f}%")

        elif isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys")
            print(f"Keys (first 10): {list(data.keys())[:10]}")
            if data:
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                print(f"First value type: {type(first_value)}")
                if hasattr(first_value, 'shape'):
                    print(f"First value shape: {first_value.shape}")
                elif isinstance(first_value, (list, tuple)):
                    print(f"First value length: {len(first_value)}")
                    if first_value:
                        print(f"First element of first value: {first_value[0]}")

        elif isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            if data:
                print(f"First element type: {type(data[0])}")
                print(f"First element: {data[0]}")

        elif isinstance(data, np.ndarray):
            print(f"Array shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            if data.size > 0:
                print(f"Min value: {np.min(data)}")
                print(f"Max value: {np.max(data)}")
                print(f"Mean value: {np.mean(data)}")

        # Try to show a sample if it's a matrix
        if hasattr(data, 'shape') and len(data.shape) == 2 and data.shape[0] > 0:
            if hasattr(data, 'toarray'):  # Sparse matrix
                sample = data[:5, :5].toarray()
            else:
                sample = data[:5, :5]
            print(f"Sample 5x5:")
            print(sample)

        return data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    # Main data files - use absolute Windows paths
    data_files = [
        ("E:/pycommend/pycommend-code/data/package_relationships_10k.pkl", "Package Relationships Matrix"),
        ("E:/pycommend/pycommend-code/data/package_similarity_matrix_10k.pkl", "Package Similarity Matrix"),
        ("E:/pycommend/pycommend-code/data/package_embeddings_10k.pkl", "Package Embeddings"),
    ]

    print("SEMANTIC DATA ANALYSIS FOR PYCOMMEND")
    print("=" * 80)

    results = {}

    # Analyze main data files
    for filepath, name in data_files:
        if os.path.exists(filepath):
            data = analyze_pkl_file(filepath, name)
            results[name] = data
        else:
            print(f"\nFile not found: {filepath}")

    return results

if __name__ == "__main__":
    results = main()