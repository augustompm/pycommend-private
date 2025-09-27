# File: create_sparse_distance_matrix.py

import json
import os
import re
from pathlib import Path
import toml
import numpy as np
from tqdm import tqdm
from scipy import sparse
import pickle

def load_top_packages(file_path):
    """Load top packages data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_package_names(top_packages):
    """Extract package names from the top packages data."""
    return [pkg["package"] for pkg in top_packages]

def create_extras_matrix(top_packages, package_names):
    """Create sparse matrix based on PyPI relationships."""
    n = len(package_names)
    
    # Criar mapeamento de nome para índice (case insensitive)
    pkg_to_idx = {name.lower(): i for i, name in enumerate(package_names)}
    
    # Usar formato COO (Coordinate) para construção eficiente
    rows = []
    cols = []
    data = []
    
    for pkg in tqdm(top_packages, desc="Processing PyPI relationships"):
        pkg_name = pkg["package"].lower()
        
        if pkg_name not in pkg_to_idx:
            continue
            
        pkg_idx = pkg_to_idx[pkg_name]
        
        # Verificar dependências
        depends_on = [dep.lower() for dep in pkg["data"].get("depends_on", [])]
        for dep in depends_on:
            if dep in pkg_to_idx:
                dep_idx = pkg_to_idx[dep]
                # Adicionar relação nos dois sentidos (matriz simétrica)
                rows.extend([pkg_idx, dep_idx])
                cols.extend([dep_idx, pkg_idx])
                data.extend([1, 1])
        
        # Verificar "depended_by"
        depended_by = [dep.lower() for dep in pkg["data"].get("depended_by", [])]
        for dep in depended_by:
            if dep in pkg_to_idx:
                dep_idx = pkg_to_idx[dep]
                # Adicionar relação nos dois sentidos (matriz simétrica)
                rows.extend([pkg_idx, dep_idx])
                cols.extend([dep_idx, pkg_idx])
                data.extend([1, 1])
    
    # Criar matriz esparsa CSR (Compressed Sparse Row) para operações eficientes
    extras_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return extras_matrix

def parse_requirements_txt(file_path):
    """Parse a requirements.txt file and extract package names."""
    packages = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and options
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                # Remove any markers or requirements after semicolon
                if ';' in line:
                    line = line.split(';')[0]
                
                # Remove version specifiers
                line = re.split(r'[<>=!~]', line)[0].strip()
                
                # Handle editable installs with -e or --editable
                if line.startswith('-e') or line.startswith('--editable'):
                    parts = line.split()
                    for part in parts[1:]:  # Skip the -e or --editable
                        if '#egg=' in part:
                            # Extract package name from egg
                            egg_part = part.split('#egg=')[1]
                            package = egg_part.split('&')[0].strip().lower()  # Remove any extra parameters
                            packages.add(package)
                            break
                else:
                    # Regular requirement
                    package = line.split('#')[0].strip().lower()  # Remove comments
                    if package:
                        packages.add(package)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return packages

def extract_dependencies_from_pyproject(file_path):
    """Extract dependencies from a pyproject.toml file using regex for robustness."""
    packages = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Find dependencies sections using regex patterns
            dependency_sections = []
            
            # Poetry dependencies
            poetry_deps = re.findall(r'\[tool\.poetry\.dependencies\](.*?)(\[|\Z)', content, re.DOTALL)
            if poetry_deps:
                dependency_sections.extend(poetry_deps)
            
            # PEP 621 dependencies
            project_deps = re.findall(r'\[project\.dependencies\](.*?)(\[|\Z)', content, re.DOTALL)
            if project_deps:
                dependency_sections.extend(project_deps)
            
            # Extract all optional dependencies sections
            opt_deps = re.findall(r'\[(?:tool\.poetry\.dev-dependencies|project\.optional-dependencies\.[^\]]+)\](.*?)(\[|\Z)', content, re.DOTALL)
            if opt_deps:
                dependency_sections.extend(opt_deps)
            
            # Process each section to extract package names
            for section in dependency_sections:
                if isinstance(section, tuple):
                    section = section[0]  # Extract string from tuple
                
                # Find package names in the section
                # Look for patterns like: package_name = "version"
                package_lines = re.findall(r'([a-zA-Z0-9_.-]+)\s*=\s*["\'](.*?)["\']', section)
                for pkg, _ in package_lines:
                    packages.add(pkg.lower())
                
                # Also look for list items (dependencies in array format)
                list_items = re.findall(r'["\'](.*?)["\']', section)
                for item in list_items:
                    # Extract package name from list item (ignore version specifiers)
                    pkg_match = re.match(r'^([a-zA-Z0-9_.-]+)', item)
                    if pkg_match:
                        packages.add(pkg_match.group(1).lower())
    
    except Exception as e:
        print(f"Error extracting dependencies from {file_path}: {e}")
    
    return packages

def create_github_matrix(package_names, github_deps_path):
    """Create sparse matrix based on co-occurrence in GitHub dependencies."""
    n = len(package_names)
    
    # Criar mapeamento de nome para índice (case insensitive)
    pkg_to_idx = {name.lower(): i for i, name in enumerate(package_names)}
    
    # Usar formato COO para construção eficiente
    rows = []
    cols = []
    data = []
    
    # Processar todos os arquivos de dependência
    deps_dir = Path(github_deps_path)
    total_files = len(list(deps_dir.glob('*')))
    
    for file_path in tqdm(deps_dir.glob('*'), total=total_files, desc="Processing dependency files"):
        if file_path.is_file():
            packages = set()
            
            if file_path.name.endswith('.txt'):
                packages = parse_requirements_txt(file_path)
            elif file_path.name.endswith('.toml'):
                # Try using toml parser first
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data_toml = toml.load(f)
                        
                        # Try different possible locations for dependencies
                        dependencies = []
                        
                        # Poetry dependencies
                        if 'tool' in data_toml and 'poetry' in data_toml.get('tool', {}):
                            poetry = data_toml['tool'].get('poetry', {})
                            if 'dependencies' in poetry:
                                dependencies.extend(poetry['dependencies'].keys())
                            if 'dev-dependencies' in poetry:
                                dependencies.extend(poetry['dev-dependencies'].keys())
                        
                        # PEP 621 dependencies
                        if 'project' in data_toml:
                            if 'dependencies' in data_toml['project']:
                                if isinstance(data_toml['project']['dependencies'], list):
                                    dependencies.extend(data_toml['project']['dependencies'])
                                elif isinstance(data_toml['project']['dependencies'], dict):
                                    dependencies.extend(data_toml['project']['dependencies'].keys())
                            
                            # Optional dependencies
                            if 'optional-dependencies' in data_toml['project']:
                                for opt_deps in data_toml['project']['optional-dependencies'].values():
                                    if isinstance(opt_deps, list):
                                        dependencies.extend(opt_deps)
                                    elif isinstance(opt_deps, dict):
                                        dependencies.extend(opt_deps.keys())
                        
                        # Clean and add dependencies
                        for dep in dependencies:
                            # Extract package name (ignoring version specifiers)
                            pkg_match = re.match(r'^([a-zA-Z0-9_.-]+)', dep)
                            if pkg_match:
                                package = pkg_match.group(1).lower()
                                packages.add(package)
                
                except Exception as e:
                    # Fallback to regex-based parsing if toml parsing fails
                    packages = extract_dependencies_from_pyproject(file_path)
            
            # Encontrar pacotes da nossa lista que aparecem neste arquivo
            found_indices = [pkg_to_idx[pkg.lower()] for pkg in packages if pkg.lower() in pkg_to_idx]
            
            # Atualizar matriz de co-ocorrência
            for i, idx1 in enumerate(found_indices):
                for idx2 in found_indices[i+1:]:
                    rows.extend([idx1, idx2])
                    cols.extend([idx2, idx1])
                    data.extend([1, 1])
    
    # Criar matriz esparsa CSR
    github_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return github_matrix

def combine_matrices(extras_matrix, github_matrix, weight_extras=1.0, weight_github=1.0):
    """Combine the two sparse matrices with optional weights."""
    return weight_extras * extras_matrix + weight_github * github_matrix

def save_sparse_matrix(matrix, package_names, output_file):
    """Save the sparse matrix and package names for later use."""
    with open(output_file, 'wb') as f:
        pickle.dump({
            'matrix': matrix,
            'package_names': package_names
        }, f)
    print(f"Sparse matrix saved to {output_file}")

def main():
    # Paths
    top_packages_file = 'data/PyPI/top_10000_packages.json'
    github_deps_path = 'data/github/dependencies'
    output_file = 'package_relationships_10k.pkl'
    
    # Carregar dados
    print("Loading package data...")
    top_packages = load_top_packages(top_packages_file)
    package_names = extract_package_names(top_packages)
    
    print(f"Processing {len(package_names)} packages...")
    
    # Criar matrizes esparsas
    extras_matrix = create_extras_matrix(top_packages, package_names)
    github_matrix = create_github_matrix(package_names, github_deps_path)
    
    # Combinar matrizes (pesos iguais por padrão)
    print("Combining matrices...")
    combined_matrix = combine_matrices(extras_matrix, github_matrix)
    
    # Zerar diagonal (sem auto-relações)
    combined_matrix.setdiag(0)
    
    # Calcular estatísticas
    density = combined_matrix.nnz / (combined_matrix.shape[0] * combined_matrix.shape[1])
    print(f"Matrix density: {density:.6f} ({combined_matrix.nnz} non-zero elements)")
    
    # Salvar resultado
    save_sparse_matrix(combined_matrix, package_names, output_file)

if __name__ == "__main__":
    main()