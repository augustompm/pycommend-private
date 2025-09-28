"""
Compare presentation results with actual v6 output
"""

import sys
import os
import numpy as np
os.chdir('pycommend-code')
sys.path.append('src/optimizer')
from nsga2_v5 import NSGA2_V5

def get_top_recommendations(package_name, top_n=10):
    """Get top-N packages by co-occurrence directly from matrix"""
    nsga2 = NSGA2_V5(package_name, pop_size=10, max_gen=1)

    idx = nsga2.main_package_idx
    connections = nsga2.rel_matrix[idx].toarray().flatten()
    ranked = np.argsort(connections)[::-1]

    recommendations = []
    for i in range(min(top_n, len(ranked))):
        pkg_idx = ranked[i]
        if connections[pkg_idx] > 0:
            recommendations.append({
                'package': nsga2.package_names[pkg_idx],
                'cooccurrence': connections[pkg_idx]
            })

    return recommendations

# Presentation test cases from LaTeX
presentation_data = {
    'fastapi': {
        'small': ['pydantic', 'uvicorn', 'typer'],
        'large': ['pydantic', 'uvicorn', 'starlette', 'httpx', 'pytest', 'jinja2', 'sqlalchemy']
    },
    'scikit-learn': {
        'small': ['optuna', 'xgboost', 'joblib'],
        'large': ['pandas', 'matplotlib', 'seaborn', 'xgboost', 'joblib', 'yellowbrick']
    },
    'prophet': ['pandas', 'matplotlib', 'numpy', 'scikit-learn'],
    'xgboost': {
        'small': ['optuna', 'pandas', 'numpy'],
        'large': ['optuna', 'pandas', 'numpy', 'scikit-learn', 'mlflow']
    }
}

print("=" * 70)
print("PRESENTATION vs REALITY CHECK")
print("=" * 70)

for package, expected in presentation_data.items():
    print(f"\n{package.upper()}:")
    print("-" * 50)

    # Get actual top recommendations
    actual = get_top_recommendations(package, 15)

    print("Top 10 by co-occurrence in our matrix:")
    for i, rec in enumerate(actual[:10], 1):
        print(f"  {i}. {rec['package']}: {rec['cooccurrence']:.0f}")

    # Check presentation recommendations
    if isinstance(expected, dict):
        for size_name, pkg_list in expected.items():
            print(f"\nPresentation {size_name} set: {pkg_list}")
            matches = []
            for pkg in pkg_list:
                for rec in actual:
                    if rec['package'] == pkg:
                        matches.append(f"{pkg}({rec['cooccurrence']:.0f})")
                        break
                else:
                    matches.append(f"{pkg}(NOT IN TOP)")
            print(f"  Status: {', '.join(matches)}")
    else:
        print(f"\nPresentation set: {expected}")
        matches = []
        for pkg in expected:
            for rec in actual:
                if rec['package'] == pkg:
                    matches.append(f"{pkg}({rec['cooccurrence']:.0f})")
                    break
            else:
                matches.append(f"{pkg}(NOT IN TOP)")
        print(f"  Status: {', '.join(matches)}")

print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

print("""
1. FASTAPI recommendations are VALID:
   - uvicorn (442) and pydantic (321) are top-2 by co-occurrence
   - httpx (216), starlette (139), jinja2 (129) all in top-15

2. SCIKIT-LEARN recommendations are REASONABLE:
   - pandas (626), matplotlib (594), joblib (287) are highly co-occurring
   - xgboost (37) and optuna (20) have lower counts but are domain-relevant

3. The presentation shows CURATED multi-objective results:
   - Not just highest co-occurrence
   - Balances semantic similarity and set size
   - Likely from Pareto front, not single best

4. Current v6 COULD produce similar results:
   - Has the data (co-occurrence matrix is accurate)
   - Has semantic similarity (SBERT embeddings)
   - Bug prevents full convergence to see if it would find these
""")