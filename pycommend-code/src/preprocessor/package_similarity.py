# File path: package_similarity.py

import json
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_package_data(file_path='data/PyPI/top_10000_packages.json'):
    """Load PyPI package data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_package_texts(package_data):
    """Create text representations for packages by combining summary, description, and keywords"""
    package_texts = []
    package_names = []
    
    print("Processing package texts...")
    for package_info in tqdm(package_data):
        name = package_info['package']
        summary = package_info['data'].get('summary', '')
        description = package_info['data'].get('description', '')
        keywords = package_info['data'].get('keywords', '')
        
        # Combine text features, giving more weight to summary and keywords
        package_text = f"{summary} {summary} {keywords} {keywords} {description}"
        package_texts.append(package_text)
        package_names.append(name)
    
    return package_texts, package_names

def compute_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Compute embeddings for all package texts using SBERT"""
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def create_similarity_matrix(embeddings):
    """Create a matrix of all pairwise similarities"""
    n = embeddings.shape[0]
    similarity_matrix = np.zeros((n, n))
    
    print("Computing similarity matrix...")
    for i in tqdm(range(n)):
        # Normalize the embedding for package i
        emb_i = embeddings[i]
        norm_i = np.linalg.norm(emb_i)
        
        # Compute similarities for all packages at once (vectorized)
        # This is much faster than computing one by one
        dots = np.dot(embeddings, emb_i)
        norms = np.linalg.norm(embeddings, axis=1) * norm_i
        sims = dots / norms
        
        similarity_matrix[i, :] = sims
    
    return similarity_matrix

def compute_similarity(package1_idx, package2_idx, similarity_matrix=None, embeddings=None):
    """
    Compute semantic similarity between two packages.
    
    Args:
        package1_idx: Index of first package
        package2_idx: Index of second package
        similarity_matrix: Pre-computed similarity matrix (optional)
        embeddings: Matrix of package embeddings (optional)
        
    Returns:
        Similarity score between 0 and 1
    """
    if similarity_matrix is not None:
        return similarity_matrix[package1_idx, package2_idx]
    
    if embeddings is not None:
        # Get embeddings for both packages
        emb1 = embeddings[package1_idx]
        emb2 = embeddings[package2_idx]
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity
    
    raise ValueError("Either similarity_matrix or embeddings must be provided")

def save_data(embeddings, similarity_matrix, package_names):
    """Save embeddings, similarity matrix, and package names to disk"""
    print("Saving embeddings...")
    with open('package_embeddings_10k.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'package_names': package_names
        }, f)
    
    print("Saving similarity matrix...")
    with open('package_similarity_matrix_10k.pkl', 'wb') as f:
        pickle.dump({
            'similarity_matrix': similarity_matrix,
            'package_names': package_names
        }, f)

def load_similarity_data(file_path='package_similarity_matrix_10k.pkl'):
    """Load pre-computed similarity data"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['similarity_matrix'], data['package_names']

def main():
    # Load package data
    package_data = load_package_data('data/PyPI/top_10000_packages.json')
    
    # Create text representations
    package_texts, package_names = create_package_texts(package_data)
    
    # Compute embeddings
    embeddings = compute_embeddings(package_texts)
    
    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(embeddings)
    
    # Save data for future use
    save_data(embeddings, similarity_matrix, package_names)
    
    # Example usage:
    # Load pre-computed similarity data
    # sim_matrix, pkg_names = load_similarity_data()
    # 
    # Find index of a package
    # idx1 = pkg_names.index('requests')
    # idx2 = pkg_names.index('urllib3')
    # 
    # Compute similarity
    # sim = compute_similarity(idx1, idx2, similarity_matrix=sim_matrix)
    # print(f"Similarity between 'requests' and 'urllib3': {sim}")

if __name__ == "__main__":
    main()