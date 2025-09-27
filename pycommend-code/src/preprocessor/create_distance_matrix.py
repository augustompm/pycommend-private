# File: create_distance_matrix.py

import json
import os
import csv
import re
from pathlib import Path
import toml

def load_top_packages(file_path):
    """Load top packages data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_package_names(top_packages):
    """Extract package names from the top packages data."""
    return [pkg["package"] for pkg in top_packages]

def create_extras_matrix(top_packages, package_names):
    """Create matrix based on extras relationships."""
    n = len(package_names)
    extras_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Create a mapping from package name to index
    pkg_to_idx = {name.lower(): i for i, name in enumerate(package_names)}
    
    # For each package, check if it appears in another package's depends_on or depended_by
    for pkg in top_packages:
        pkg_name = pkg["package"].lower()
        
        if pkg_name not in pkg_to_idx:
            continue
            
        pkg_idx = pkg_to_idx[pkg_name]
        
        # Check if this package depends on other packages
        depends_on = [dep.lower() for dep in pkg["data"].get("depends_on", [])]
        for dep in depends_on:
            if dep in pkg_to_idx:
                dep_idx = pkg_to_idx[dep]
                extras_matrix[pkg_idx][dep_idx] += 1
                extras_matrix[dep_idx][pkg_idx] += 1  # Symmetric relationship
        
        # Check if other packages depend on this one
        depended_by = [dep.lower() for dep in pkg["data"].get("depended_by", [])]
        for dep in depended_by:
            if dep in pkg_to_idx:
                dep_idx = pkg_to_idx[dep]
                extras_matrix[pkg_idx][dep_idx] += 1
                extras_matrix[dep_idx][pkg_idx] += 1  # Symmetric relationship
    
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

def create_github_matrix(top_packages, package_names, github_deps_path):
    """Create matrix based on co-occurrence in GitHub dependencies."""
    n = len(package_names)
    github_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Create a mapping from package name to index (case insensitive)
    pkg_to_idx = {name.lower(): i for i, name in enumerate(package_names)}
    
    processed_count = 0
    # Process all dependency files
    deps_dir = Path(github_deps_path)
    for file_path in deps_dir.glob('*'):
        if file_path.is_file():
            packages = set()
            
            if file_path.name.endswith('.txt'):
                packages = parse_requirements_txt(file_path)
            elif file_path.name.endswith('.toml'):
                # Try using toml parser first
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = toml.load(f)
                        
                        # Try different possible locations for dependencies
                        dependencies = []
                        
                        # Poetry dependencies
                        if 'tool' in data and 'poetry' in data.get('tool', {}):
                            poetry = data['tool'].get('poetry', {})
                            if 'dependencies' in poetry:
                                dependencies.extend(poetry['dependencies'].keys())
                            if 'dev-dependencies' in poetry:
                                dependencies.extend(poetry['dev-dependencies'].keys())
                        
                        # PEP 621 dependencies
                        if 'project' in data:
                            if 'dependencies' in data['project']:
                                if isinstance(data['project']['dependencies'], list):
                                    dependencies.extend(data['project']['dependencies'])
                                elif isinstance(data['project']['dependencies'], dict):
                                    dependencies.extend(data['project']['dependencies'].keys())
                            
                            # Optional dependencies
                            if 'optional-dependencies' in data['project']:
                                for opt_deps in data['project']['optional-dependencies'].values():
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
            
            # Find packages from our list that appear in this file
            found_packages = [pkg for pkg in packages if pkg.lower() in pkg_to_idx]
            
            # Update co-occurrence matrix
            for i, pkg1 in enumerate(found_packages):
                for pkg2 in found_packages[i+1:]:
                    idx1 = pkg_to_idx[pkg1.lower()]
                    idx2 = pkg_to_idx[pkg2.lower()]
                    github_matrix[idx1][idx2] += 1
                    github_matrix[idx2][idx1] += 1  # Symmetric relationship
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} dependency files...")
    
    print(f"Total dependency files processed: {processed_count}")
    return github_matrix

def combine_matrices(extras_matrix, github_matrix, weight_extras=1.0, weight_github=1.0):
    """Combine the two matrices with optional weights."""
    n = len(extras_matrix)
    combined_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            combined_matrix[i][j] = (
                weight_extras * extras_matrix[i][j] + 
                weight_github * github_matrix[i][j]
            )
    
    return combined_matrix

def save_matrix_to_csv(matrix, package_names, output_file):
    """Save the matrix to a CSV file with row and column headers."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow([''] + package_names)
        # Write data rows
        for i, name in enumerate(package_names):
            writer.writerow([name] + matrix[i])

def main():
    # Paths
    top_packages_file = 'data/PyPI/top_10_packages.json'
    github_deps_path = 'data/github/dependencies'
    output_file = 'package_relationship_matrix.csv'
    
    # Load data
    top_packages = load_top_packages(top_packages_file)
    package_names = extract_package_names(top_packages)
    
    print(f"Processing {len(package_names)} packages...")
    
    # Create matrices
    extras_matrix = create_extras_matrix(top_packages, package_names)
    github_matrix = create_github_matrix(top_packages, package_names, github_deps_path)
    
    # Combine matrices (equal weights by default)
    combined_matrix = combine_matrices(extras_matrix, github_matrix)
    
    # Make diagonal elements zero (no self-relationships)
    for i in range(len(combined_matrix)):
        combined_matrix[i][i] = 0
    
    # Save result
    save_matrix_to_csv(combined_matrix, package_names, output_file)
    print(f"Matrix saved to {output_file}")

if __name__ == "__main__":
    main()