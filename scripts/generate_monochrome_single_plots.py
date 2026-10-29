"""
Generate publication-grade monochromatic (grayscale) single-graph figures.
Each image contains strictly ONE standalone graph (no multi-panel subplots).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Output Directory ──────────────────────────────────────────────────────────
OUT_DIR = 'output_images/monochrome'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Monochromatic Palette ─────────────────────────────────────────────────────
# True academic grayscale palette with distinct contrast & hatching patterns
BLACK      = '#000000'
VERY_DARK  = '#262626'
DARK_GRAY  = '#404040'
MID_GRAY   = '#737373'
LIGHT_GRAY = '#A3A3A3'
VERY_LIGHT = '#E5E5E5'
WHITE      = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.edgecolor': BLACK,
    'axes.linewidth': 1.2,
    'xtick.color': BLACK,
    'ytick.color': BLACK,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
})

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 1: Normalized 5-Class Confusion Matrix (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 1: Confusion Matrix (Monochrome)...")
cm = np.array([
    [0.52, 0.14, 0.18, 0.04, 0.12],
    [0.22, 0.26, 0.38, 0.01, 0.13],
    [0.08, 0.08, 0.68, 0.11, 0.05],
    [0.01, 0.00, 0.12, 0.86, 0.01],
    [0.09, 0.09, 0.14, 0.01, 0.67]
])
stages = ['Wake', 'N1', 'N2', 'N3', 'REM']

fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
im = ax.imshow(cm, cmap='Greys', vmin=0, vmax=1)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Ratio', fontweight='bold', fontsize=11)
cbar.ax.tick_params(labelsize=10)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(stages)
ax.set_yticklabels(stages)
ax.set_xlabel('Predicted Sleep Stage')
ax.set_ylabel('True Clinical Sleep Stage')
ax.set_title('Pooled Confusion Matrix (LOSO LightGBM, N=5,126)')

for i in range(5):
    for j in range(5):
        val = cm[i, j]
        color = WHITE if val > 0.50 else BLACK
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/01_confusion_matrix_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 2: Per-Stage F1 by Subject (Monochrome with Hatching)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 2: Per-Stage F1 by Subject...")
loso_df = pd.read_csv('results/loso_staging_results.csv')
subjects = loso_df['test_subject'].values
x = np.arange(len(subjects))
w = 0.15

stage_cols = ['f1_Wake', 'f1_N1', 'f1_N2', 'f1_N3', 'f1_REM']
grays = [VERY_DARK, DARK_GRAY, MID_GRAY, LIGHT_GRAY, VERY_LIGHT]
hatches = ['///', '\\\\\\', 'xxx', '...', '---']

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
for i, (col, sname, color, hatch) in enumerate(zip(stage_cols, stages, grays, hatches)):
    vals = loso_df[col].values
    ax.bar(x + i*w - 2*w, vals, w, label=sname, color=color,
           edgecolor=BLACK, linewidth=0.8, hatch=hatch)

ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_xlabel('Held-Out Subject')
ax.set_ylabel('F1-Score (%)')
ax.set_title('Per-Stage F1-Score Across Held-Out Subjects')
ax.legend(title='Stage', loc='upper right', framealpha=0.95, edgecolor=BLACK)
ax.set_ylim(0, 105)
ax.axhline(75, color=DARK_GRAY, linestyle='--', linewidth=1.2, alpha=0.8)
ax.text(len(subjects)-0.55, 76.5, 'Expert Target (75%)', fontsize=9.5, fontweight='bold', color=DARK_GRAY, ha='right')
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/02_per_stage_f1_by_subject_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 3: Overall Staging Metrics by Subject (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 3: Overall Staging Metrics by Subject...")
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
bw = 0.24
kappas = loso_df['kappa'].values * 100

b1 = ax.bar(x - bw, loso_df['accuracy'].values, bw, label='Accuracy (%)',
            color=VERY_DARK, edgecolor=BLACK, linewidth=0.8, hatch='///')
b2 = ax.bar(x,      loso_df['macro_f1'].values, bw, label='Macro F1 (%)',
            color=MID_GRAY,  edgecolor=BLACK, linewidth=0.8, hatch='\\\\\\')
b3 = ax.bar(x + bw, kappas,                     bw, label="Cohen's κ × 100",
            color=VERY_LIGHT, edgecolor=BLACK, linewidth=0.8, hatch='...')

ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_xlabel('Held-Out Subject')
ax.set_ylabel('Score (%)')
ax.set_title("Overall Staging Performance by Subject (LOSO LightGBM)")
ax.legend(loc='upper right', framealpha=0.95, edgecolor=BLACK)
ax.set_ylim(0, 105)
ax.axhline(75, color=BLACK, linestyle='--', linewidth=1.2)
ax.text(len(subjects)-0.55, 76.5, 'Clinical Benchmark (75)', fontsize=9.5, fontweight='bold', ha='right')
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/03_overall_metrics_by_subject_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 4: Top-20 SHAP Feature Importance (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 4: SHAP Feature Importance (Monochrome)...")
with open('results/top20_shap.json') as f:
    top20 = json.load(f)[:20]

feat_names = [item[0] for item in top20]
shap_vals = [item[1] for item in top20]

fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)
y_pos = np.arange(len(feat_names))

bars = ax.barh(y_pos, shap_vals[::-1], color=DARK_GRAY, edgecolor=BLACK,
               linewidth=0.8, height=0.65, hatch='///')

ax.set_yticks(y_pos)
ax.set_yticklabels(feat_names[::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP Value| (Impact on Model Log-Odds)')
ax.set_title('Global Feature Importance Ranking (TreeExplainer SHAP)')
ax.grid(axis='x', linestyle=':', color=LIGHT_GRAY, alpha=0.6)
ax.set_xlim(0, max(shap_vals) * 1.15)

for bar, val in zip(bars, shap_vals[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=9, fontweight='bold', color=BLACK)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/04_shap_feature_importance_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 5: Modality Ablation Comparison (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 5: Modality Ablation Comparison...")
ablation_configs = [
    'Resp only', 'ECG only', 'EMG only', 'EEG only',
    'EOG only', 'EOG + EMG', 'All 5 (Static)', 'All 5 (Temporal)'
]
f1_scores = [6.0, 16.0, 23.5, 31.2, 45.6, 48.5, 54.7, 57.5]
kappas    = [-0.02, 0.02, 0.12, 0.21, 0.41, 0.45, 0.46, 0.50]

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
x_abl = np.arange(len(ablation_configs))

bars = ax.bar(x_abl, f1_scores, color=DARK_GRAY, edgecolor=BLACK,
              linewidth=1.0, width=0.6, hatch='///')
# Highlight full multimodal bar
bars[-1].set_color(VERY_DARK)
bars[-1].set_edgecolor(BLACK)
bars[-1].set_linewidth(1.8)
bars[-1].set_hatch('xxx')

ax.set_xticks(x_abl)
ax.set_xticklabels(ablation_configs, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('Macro F1-Score (%)')
ax.set_title('Modality Ablation Study: Single Signals vs. Multimodal Integration')
ax.set_ylim(0, 70)
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

for bar, f1, k in zip(bars, f1_scores, kappas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{f1:.1f}%\nκ={k:.2f}', ha='center', fontsize=9, fontweight='bold', color=BLACK)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/05_modality_ablation_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 6: Apnea Detection AUC by Subject (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 6: Apnea Detection AUC by Subject...")
apnea_df = pd.read_csv('results/apnea_v2_results.csv')
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

x_ap = np.arange(len(apnea_df))
bars = ax.bar(x_ap, apnea_df['AUC'].values, color=MID_GRAY,
              edgecolor=BLACK, linewidth=1.0, width=0.55, hatch='\\\\\\')

ax.set_xticks(x_ap)
ax.set_xticklabels(apnea_df['Subject'].values)
ax.set_xlabel('Held-Out Subject')
ax.set_ylabel('Area Under ROC Curve (AUC)')
ax.set_title('Sleep Apnea Detection AUC Under 12-Scorer Consensus')
ax.set_ylim(0, 0.70)
ax.axhline(0.50, color=BLACK, linestyle='--', linewidth=1.2)
ax.text(len(apnea_df)-0.55, 0.51, 'Chance Line (0.50)', fontsize=9.5, fontweight='bold', ha='right')
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

for bar, auc in zip(bars, apnea_df['AUC'].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{auc:.3f}', ha='center', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/06_apnea_auc_by_subject_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 7: Apnea Detection Optimal F1 by Subject (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 7: Apnea Detection Optimal F1 by Subject...")
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

bars = ax.bar(x_ap, apnea_df['F1_opt'].values, color=DARK_GRAY,
              edgecolor=BLACK, linewidth=1.0, width=0.55, hatch='///')

ax.set_xticks(x_ap)
ax.set_xticklabels(apnea_df['Subject'].values)
ax.set_xlabel('Held-Out Subject')
ax.set_ylabel('Optimal F1-Score')
ax.set_title('Apnea Detection Optimal F1 by Subject (Consensus Ground Truth)')
ax.set_ylim(0, 0.55)
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

for bar, f1, t in zip(bars, apnea_df['F1_opt'].values, apnea_df['thresh'].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'F1={f1:.3f}\n(@t={t:.2f})', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/07_apnea_f1_optimal_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 8: Model Architecture Progression (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 8: Architecture Progression...")
phases = [
    'Phase 1\n(Single Signals)',
    'Phase 2\n(Union Join)',
    'Phase 4\n(Refined Gates)',
    'Phase 5 Stage 1\n(Temporal Context)',
    'Phase 5 Stage 2\n(Stacked Meta)'
]
prog_f1 = [45.6, 54.65, 53.53, 57.54, 55.10]
prog_k  = [0.41, 0.4586, 0.4439, 0.5017, 0.4659]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
x_pr = np.arange(len(phases))

bars = ax.bar(x_pr, prog_f1, color=MID_GRAY, edgecolor=BLACK,
              linewidth=1.0, width=0.55, hatch='///')
# Highlight temporal best
bars[3].set_color(VERY_DARK)
bars[3].set_edgecolor(BLACK)
bars[3].set_linewidth(1.8)
bars[3].set_hatch('xxx')

ax.set_xticks(x_pr)
ax.set_xticklabels(phases, fontsize=10)
ax.set_ylabel('Pooled Macro F1-Score (%)')
ax.set_title('Sleep Staging Progression Across Development Phases')
ax.set_ylim(0, 70)
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

for bar, f1, k in zip(bars, prog_f1, prog_k):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{f1:.2f}%\nκ={k:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/08_pipeline_progression_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# GRAPH 9: Sleep Stage Cohort Class Distribution (Monochrome)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating Graph 9: Class Distribution...")
stage_counts = [1230, 368, 1832, 939, 757]
stage_pcts   = [c / sum(stage_counts) * 100 for c in stage_counts]

fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
bars = ax.bar(stages, stage_counts, color=MID_GRAY, edgecolor=BLACK,
              linewidth=1.0, width=0.55, hatch=['///', '\\\\\\', 'xxx', '...', '---'])

ax.set_xlabel('AASM Sleep Stage')
ax.set_ylabel('Total Epochs (30s each)')
ax.set_title('Sleep Stage Distribution Across Dataset (N=5,126 Epochs)')
ax.set_ylim(0, 2200)
ax.grid(axis='y', linestyle=':', color=LIGHT_GRAY, alpha=0.6)

for bar, count, pct in zip(bars, stage_counts, stage_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 35,
            f'{count}\n({pct:.1f}%)', ha='center', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/09_class_distribution_monochrome.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("\nAll 9 standalone monochromatic figures generated successfully in output_images/monochrome/!")
