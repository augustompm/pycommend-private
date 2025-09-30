"""
Debug do cálculo de spacing
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from evaluation.quality_metrics import QualityMetrics

print("Teste de Spacing")
print("="*60)

# Criar dados de teste simples
objectives = np.array([
    [-1000, -0.5, 5],
    [-2000, -0.3, 7],
    [-500,  -0.7, 3]
])

print("Objetivos de teste:")
print(objectives)

# Criar QualityMetrics
qm = QualityMetrics()

# Calcular spacing
spacing = qm.spacing(objectives)
print(f"\nSpacing calculado: {spacing:.4f}")

# Verificar normalização manual
print("\nDebug da normalização:")
print(f"Tem normalize_objectives? {hasattr(qm, 'normalize_objectives')}")

# Normalizar manualmente
obj_min = np.min(objectives, axis=0)
obj_max = np.max(objectives, axis=0)
print(f"Min: {obj_min}")
print(f"Max: {obj_max}")

normalized = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
print(f"\nNormalizado manualmente:")
print(normalized)

# Calcular spacing manualmente nos dados normalizados
n = len(normalized)
distances = []
for i in range(n):
    min_dist = float('inf')
    for j in range(n):
        if i != j:
            dist = np.linalg.norm(normalized[i] - normalized[j])
            if dist < min_dist:
                min_dist = dist
    distances.append(min_dist)

mean_dist = np.mean(distances)
spacing_manual = np.sqrt(np.sum((distances - mean_dist) ** 2) / (n - 1))

print(f"\nSpacing manual (normalizado): {spacing_manual:.4f}")

# Testar se QualityMetrics.spacing está normalizando
print("\n" + "="*60)
print("Teste com valores grandes vs pequenos:")

# Valores grandes
big_objectives = objectives * 1000
spacing_big = qm.spacing(big_objectives)
print(f"Spacing com valores *1000: {spacing_big:.4f}")

# Valores pequenos
small_objectives = objectives / 1000
spacing_small = qm.spacing(small_objectives)
print(f"Spacing com valores /1000: {spacing_small:.4f}")

print("\nSe a normalização funciona, os valores devem ser iguais!")