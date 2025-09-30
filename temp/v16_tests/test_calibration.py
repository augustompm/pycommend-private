"""
Teste de calibração: mais soluções = melhor spacing?
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("TESTE: EFEITO DO NÚMERO DE SOLUÇÕES NO SPACING")
print("="*70)

# Teste 1: MOVNS com poucas soluções (atual)
print("\n1. MOVNS Advanced padrão (poucas soluções)")
movns1 = MOVNS_Advanced('fastapi', archive_size=100, max_iterations=10, track_metrics=True)
sol1 = movns1.run()

qm1 = QualityMetrics()
obj1 = []
for s in sol1:
    obj1.append(movns1.evaluate_objectives(s['chromosome']))
obj1 = np.array(obj1)

spacing1 = qm1.spacing(obj1) if len(obj1) > 1 else float('inf')
hv1 = movns1.get_metrics_history()['hypervolume'][-1] if movns1.get_metrics_history() else 0

print(f"Soluções: {len(sol1)}")
print(f"HV: {hv1:.4f}")
print(f"Spacing: {spacing1:.4f}")

# Teste 2: MOEA/D para comparação
print("\n2. MOEA/D (100 soluções)")
moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=10, track_metrics=True)
sol2 = moead.run()

qm2 = QualityMetrics()
obj2 = []
for s in sol2:
    obj2.append(moead.evaluate_objectives(s['chromosome']))
obj2 = np.array(obj2)

spacing2 = qm2.spacing(obj2) if len(obj2) > 1 else float('inf')
hv2 = moead.get_metrics_history()['hypervolume'][-1] if moead.get_metrics_history() else 0

print(f"Soluções: {len(sol2)}")
print(f"HV: {hv2:.4f}")
print(f"Spacing: {spacing2:.4f}")

print("\n" + "="*70)
print("ANÁLISE")
print("="*70)
print(f"MOVNS tem {len(sol1)} soluções, spacing={spacing1:.4f}")
print(f"MOEA/D tem {len(sol2)} soluções, spacing={spacing2:.4f}")
print(f"\nMais soluções geralmente = menor spacing (melhor distribuição)")
print(f"MOVNS precisa manter mais soluções no arquivo!")