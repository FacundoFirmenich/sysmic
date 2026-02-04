"""
Generate Appendix C Figure 3: Zaccagnino Stability Curves Pan-American

Creates horizontal barplot showing Zaccagnino Mmin-independence stability scores
for 7 Pan-American tectonic regions with threshold lines.

Data source: bonusxp/zaccagnino_stability_all_20251212_233054.csv
Output: figures_publication/figC3_zaccagnino_stability_panam.{png,pdf}
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

# Data (from documented results)
regions = [
    'San Andreas',
    'Cocos',
    'Andes Central',
    'Caribbean',
    'Andes Sur',
    'Cascadia',
    'Andes Norte'
]

stability_scores = [0.973, 0.967, 0.963, 0.954, 0.946, 0.921, 0.889]

# Color coding by threshold
colors = []
for s in stability_scores:
    if s > 0.95:
        colors.append('#2ecc71')  # Green: Robust
    elif s >= 0.90:
        colors.append('#f39c12')  # Yellow: Marginal
    else:
        colors.append('#e74c3c')  # Red: Incompleteness

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Horizontal barplot
y_pos = np.arange(len(regions))
bars = ax.barh(y_pos, stability_scores, color=colors, edgecolor='black', linewidth=0.8)

# Threshold lines
ax.axvline(x=0.95, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Robust threshold (S>0.95)')
ax.axvline(x=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Marginal threshold (S=0.90)')

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(regions)
ax.set_xlabel('Zaccagnino Stability Score (S)', fontsize=12, fontweight='bold')
ax.set_ylabel('Tectonic Region', fontsize=12, fontweight='bold')
ax.set_title('Zaccagnino Mmin-Independence Stability\\nPan-American Regions', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0.85, 1.0)
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.legend(loc='lower right', fontsize=9)

# Add score labels on bars
for i, (bar, score) in enumerate(zip(bars, stability_scores)):
    ax.text(score + 0.002, i, f'{score:.3f}', 
            va='center', ha='left', fontsize=9, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save
plt.savefig('figures_publication/figC3_zaccagnino_stability_panam.png', dpi=300, bbox_inches='tight')
plt.savefig('figures_publication/figC3_zaccagnino_stability_panam.pdf', bbox_inches='tight')

print("✅ Figure C3 generated: figC3_zaccagnino_stability_panam.{png,pdf}")
print(f"   Regions: {len(regions)}")
print(f"   Robust (S>0.95): {sum(1 for s in stability_scores if s > 0.95)}/7")
print(f"   Marginal (0.90-0.95): {sum(1 for s in stability_scores if 0.90 <= s <= 0.95)}/7")
print(f"   Reject (S<0.90): {sum(1 for s in stability_scores if s < 0.90)}/7")

plt.show()
