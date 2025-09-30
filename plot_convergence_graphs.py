"""
Plot convergence graphs from real tracking data
Shows HV, Spacing, Archive Size and Time evolution
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load median data
df = pd.read_csv('pycommend-code/convergence_tracking_median_20250930_020410.csv')

# Separate by algorithm
movns_data = df[df['algorithm'] == 'MOVNS'].copy()
moead_data = df[df['algorithm'] == 'MOEAD'].copy()

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Real Convergence Analysis - MOVNS vs MOEA/D (Median of 5 runs)', fontsize=14, fontweight='bold')

# 1. Hypervolume (currently 0 due to bug, but show for completeness)
ax1 = axes[0, 0]
ax1.plot(movns_data['iteration'], movns_data['hv'], 'b-', label='MOVNS', linewidth=2, marker='o', markersize=4)
ax1.plot(moead_data['iteration'], moead_data['hv'], 'r--', label='MOEA/D', linewidth=2, marker='s', markersize=4)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Hypervolume')
ax1.set_title('Hypervolume Convergence (Bug: Returns 0)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Spacing
ax2 = axes[0, 1]
ax2.plot(movns_data['iteration'], movns_data['spacing'], 'b-', label='MOVNS', linewidth=2, marker='o', markersize=4)
ax2.plot(moead_data['iteration'], moead_data['spacing'], 'r--', label='MOEA/D', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Spacing (Lower is Better)')
ax2.set_title('Spacing Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add convergence point for MOVNS
convergence_point = movns_data[movns_data['iteration'] == 10].iloc[0]
ax2.axvline(x=10, color='blue', linestyle=':', alpha=0.5, label='MOVNS Converged')
ax2.text(10, convergence_point['spacing'], f'Converged\n({convergence_point["spacing"]:.4f})',
         ha='left', va='bottom', fontsize=8)

# 3. Archive/Population Size
ax3 = axes[1, 0]
ax3.plot(movns_data['iteration'], movns_data['archive_size'], 'b-', label='MOVNS Archive', linewidth=2, marker='o', markersize=4)
ax3.plot(moead_data['iteration'], moead_data['archive_size'], 'r--', label='MOEA/D Population', linewidth=2, marker='s', markersize=4)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Number of Solutions')
ax3.set_title('Archive/Population Size Evolution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Execution Time
ax4 = axes[1, 1]
ax4.plot(movns_data['iteration'], movns_data['time'], 'b-', label='MOVNS', linewidth=2, marker='o', markersize=4)
ax4.plot(moead_data['iteration'], moead_data['time'], 'r--', label='MOEA/D', linewidth=2, marker='s', markersize=4)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Cumulative Time (seconds)')
ax4.set_title('Execution Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add speedup annotation
final_movns_time = movns_data.iloc[-1]['time']
final_moead_time = moead_data.iloc[-1]['time']
speedup = final_moead_time / final_movns_time
ax4.text(0.95, 0.95, f'Speedup: {speedup:.1f}x',
         transform=ax4.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('convergence_graphs.png', dpi=300, bbox_inches='tight')
plt.savefig('convergence_graphs.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("="*70)
print("CONVERGENCE ANALYSIS SUMMARY")
print("="*70)

print(f"\n1. SPACING (Lower is Better):")
print(f"   MOVNS Final: {movns_data.iloc[-1]['spacing']:.4f}")
print(f"   MOEA/D Final: {moead_data.iloc[-1]['spacing']:.4f}")
if movns_data.iloc[-1]['spacing'] < moead_data.iloc[-1]['spacing']:
    print(f"   Winner: MOVNS (better distribution)")
else:
    print(f"   Winner: MOEA/D (better distribution)")

print(f"\n2. ARCHIVE/POPULATION SIZE:")
print(f"   MOVNS: {movns_data.iloc[0]['archive_size']:.0f} → {movns_data.iloc[-1]['archive_size']:.0f} (adaptive)")
print(f"   MOEA/D: {moead_data.iloc[-1]['archive_size']:.0f} (fixed)")

print(f"\n3. EXECUTION TIME:")
print(f"   MOVNS: {final_movns_time:.2f}s")
print(f"   MOEA/D: {final_moead_time:.2f}s")
print(f"   Speedup: MOVNS is {speedup:.1f}x faster")

print(f"\n4. CONVERGENCE SPEED:")
movns_converged = movns_data[movns_data['archive_size'] == movns_data.iloc[-1]['archive_size']].iloc[0]
print(f"   MOVNS converged at iteration {movns_converged['iteration']}")
print(f"   MOEA/D: No early convergence (runs all 30 iterations)")

print("\n" + "="*70)
print("Graphs saved as 'convergence_graphs.png' and 'convergence_graphs.pdf'")