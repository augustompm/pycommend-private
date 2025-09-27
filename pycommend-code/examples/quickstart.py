#!/usr/bin/env python
"""
PyCommend Quick Start Example

This example shows how to get library recommendations using both algorithms.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from optimizer.nsga2 import NSGA2
from optimizer.moead import MOEAD


def get_recommendations_nsga2(package_name, top_n=5):
    """Get recommendations using NSGA-II algorithm"""
    print(f"\n=== NSGA-II Recommendations for '{package_name}' ===")

    # Initialize and run NSGA-II
    nsga2 = NSGA2(package_name, pop_size=30, max_gen=10)
    solutions = nsga2.run()

    # Format results
    df = nsga2.format_results(solutions)

    # Get top N solutions
    top_solutions = df.head(top_n)

    print(f"\nTop {top_n} recommendations:")
    for idx, row in top_solutions.iterrows():
        packages = row['recommended_packages'].split(',')
        print(f"\n{row['solution_id']}. Size: {int(row['size'])} packages")
        print(f"   Packages: {', '.join(packages[:5])}")
        if len(packages) > 5:
            print(f"            {', '.join(packages[5:])}")
        print(f"   Co-usage score: {row['linked_usage']:.2f}")
        print(f"   Similarity: {row['semantic_similarity']:.4f}")

    return df


def get_recommendations_moead(package_name, top_n=5):
    """Get recommendations using MOEA/D algorithm"""
    print(f"\n=== MOEA/D Recommendations for '{package_name}' ===")

    # Initialize and run MOEA/D
    moead = MOEAD(package_name, pop_size=30, n_neighbors=10, max_gen=10)
    solutions = moead.run()

    # Format results
    df = moead.format_results(solutions)

    # Get top N solutions
    top_solutions = df.head(top_n)

    print(f"\nTop {top_n} recommendations:")
    for idx, row in top_solutions.iterrows():
        packages = row['recommended_packages'].split(',')
        print(f"\n{row['solution_id']}. Size: {int(row['size'])} packages")
        print(f"   Packages: {', '.join(packages[:5])}")
        if len(packages) > 5:
            print(f"            {', '.join(packages[5:])}")
        print(f"   Co-usage score: {row['linked_usage']:.2f}")
        print(f"   Similarity: {row['semantic_similarity']:.4f}")

    return df


def compare_algorithms(package_name):
    """Compare recommendations from both algorithms"""
    print(f"\n{'='*60}")
    print(f"Comparing Algorithms for '{package_name}'")
    print('='*60)

    # Get recommendations from both
    nsga2_df = get_recommendations_nsga2(package_name, top_n=3)
    moead_df = get_recommendations_moead(package_name, top_n=3)

    # Compare metrics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print(f"\nNSGA-II:")
    print(f"  Total solutions: {len(nsga2_df)}")
    print(f"  Avg co-usage: {nsga2_df['linked_usage'].mean():.2f}")
    print(f"  Avg similarity: {nsga2_df['semantic_similarity'].mean():.4f}")
    print(f"  Avg size: {nsga2_df['size'].mean():.1f}")

    print(f"\nMOEA/D:")
    print(f"  Total solutions: {len(moead_df)}")
    print(f"  Avg co-usage: {moead_df['linked_usage'].mean():.2f}")
    print(f"  Avg similarity: {moead_df['semantic_similarity'].mean():.4f}")
    print(f"  Avg size: {moead_df['size'].mean():.1f}")


def main():
    """Main function with examples"""

    print("PyCommend - Python Library Recommendation System")
    print("=" * 60)

    # Example 1: Get recommendations for a single package
    package = "requests"

    try:
        # Try NSGA-II
        get_recommendations_nsga2(package, top_n=3)

    except FileNotFoundError:
        print("\nError: Data files not found!")
        print("Please ensure the following files exist:")
        print("  - data/package_relationships_10k.pkl")
        print("  - data/package_similarity_matrix_10k.pkl")
        return 1

    except ValueError as e:
        print(f"\nError: {e}")
        print("The package might not be in the dataset.")
        print("Try one of: numpy, pandas, requests, fastapi, scikit-learn")
        return 1

    # Example 2: Compare both algorithms (optional)
    print("\n" + "="*60)
    response = input("Compare with MOEA/D? (y/n): ")
    if response.lower() == 'y':
        compare_algorithms(package)

    print("\n" + "="*60)
    print("Quick start completed!")
    print("\nFor more examples, see the documentation.")

    return 0


if __name__ == "__main__":
    exit(main())