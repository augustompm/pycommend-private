"""
Teste do bug de normalização - ideal/nadir points persistem
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from evaluation.quality_metrics import QualityMetrics

print("Bug de normalização - ideal/nadir persistem")
print("="*60)

# Criar nova instância
qm = QualityMetrics()

# Primeiro conjunto de dados
obj1 = np.array([
    [-100, -0.5, 5],
    [-200, -0.3, 7]
])

print("Primeira chamada com dados pequenos:")
print(obj1)
spacing1 = qm.spacing(obj1)
print(f"Spacing: {spacing1:.4f}")
print(f"ideal_point: {qm.ideal_point}")
print(f"nadir_point: {qm.nadir_point}")

# Segundo conjunto com valores muito diferentes
obj2 = np.array([
    [-10000, -0.9, 2],
    [-5000,  -0.1, 15]
])

print("\nSegunda chamada com dados grandes:")
print(obj2)
spacing2 = qm.spacing(obj2)
print(f"Spacing: {spacing2:.4f}")
print(f"ideal_point ainda: {qm.ideal_point}")  # BUG: não muda!
print(f"nadir_point ainda: {qm.nadir_point}")  # BUG: não muda!

print("\n*** BUG CONFIRMADO: ideal/nadir não atualizam! ***")

# Teste com nova instância
print("\nCom nova instância QualityMetrics:")
qm2 = QualityMetrics()
spacing2_novo = qm2.spacing(obj2)
print(f"Spacing com nova instância: {spacing2_novo:.4f}")
print(f"ideal_point novo: {qm2.ideal_point}")
print(f"nadir_point novo: {qm2.nadir_point}")