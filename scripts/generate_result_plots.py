"""
Generate all result visualizations for Phase 3 final pipeline
"""
import warnings, os, json
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from collections import Counter

np.random.seed(42)

# ── Color palette ──────────────────────────────────────────────────────────
NAVY   = '#0F2942'
BLUE   = '#2F81CF'
CYAN   = '#22D3EE'
GREEN  = '#10B981'
AMBER  = '#F59E0B'
RED    = '#EF4444'
PURPLE = '#8B5CF6'
WHITE  = '#FFFFFF'
GRAY   = '#F8FAFC'
STAGE_COLORS = ['#6366F1','#F59E0B','#10B981','#3B82F6','#EC4899']
STAGE_NAMES  = ['Wake','N1','N2','N3','REM']

df = pd.read_parquet('features/fused_union.parquet')
META = ['subject','epoch_idx','hard_label','apnea_label','quality_ok',
        'soft_W','soft_N1','soft_N2','soft_N3','soft_REM']
FLAGS = ['eeg_present','eog_present','emg_present','ecg_present','resp_sao2_present']
FEATS = [c for c in df.columns if c not in META and c not in FLAGS]
subjects = sorted(df['subject'].unique())

# ── Re-run LOSO to collect pooled predictions ─────────────────────────────
print("Collecting pooled predictions for confusion matrix ...")
all_true, all_pred = [], []

for test_subj in subjects:
    tr = df[df['subject'] != test_subj].copy()
    te = df[df['subject'] == test_subj].copy()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[FEATS].fillna(0))
    X_te = scaler.transform(te[FEATS].fillna(0))
    y_tr = tr['hard_label'].values
    y_te = te['hard_label'].values
    mc = Counter(y_tr)
    k  = min(5, min(mc.values())-1) if min(mc.values()) > 1 else 1
    try:
        X_sm, y_sm = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_tr, y_tr)
    except Exception:
        X_sm, y_sm = X_tr, y_tr
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
        num_leaves=63, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced', random_state=42, verbose=-1)
    m.fit(X_sm, y_sm)
    y_pred = m.predict(X_te)
    all_true.extend(y_te.tolist())
    all_pred.extend(y_pred.tolist())
    print(f"  [{test_subj}] done")

all_true = np.array(all_true)
all_pred = np.array(all_pred)

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Confusion Matrix (normalised)
# ────────────────────────────────────────────────────────────────────────────
print("\nPlotting confusion matrix ...")
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
fig.patch.set_facecolor(WHITE)

cm = confusion_matrix(all_true, all_pred, labels=[0,1,2,3,4], normalize='true')
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(STAGE_NAMES, fontsize=12)
ax.set_yticklabels(STAGE_NAMES, fontsize=12)
ax.set_xlabel('Predicted Stage', fontsize=13, fontweight='bold')
ax.set_ylabel('True Stage', fontsize=13, fontweight='bold')
ax.set_title('Pooled LOSO Confusion Matrix\n(LightGBM, 5-fold, 5,126 epochs)', fontsize=13, fontweight='bold', pad=12)

for i in range(5):
    for j in range(5):
        v = cm[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=11, color='white' if v > 0.55 else NAVY, fontweight='bold')

plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/confusion_matrix.png")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 2: SHAP Top-20 Feature Importance
# ────────────────────────────────────────────────────────────────────────────
print("Plotting SHAP importance ...")
with open('results/top20_shap.json') as f:
    top20 = json.load(f)

names  = [x[0] for x in top20]
values = [x[1] for x in top20]

# Colour by modality
def modality_color(name):
    if name.startswith('eeg'): return BLUE
    if name.startswith('eog'): return PURPLE
    if name.startswith('emg') or name.startswith('leg'): return GREEN
    if any(x in name for x in ['hr','hrv','lf','hf','poincare']): return AMBER
    if any(x in name for x in ['resp','sao2','thoraco','chest','flow']): return CYAN
    return RED   # gates

colors = [modality_color(n) for n in names]

fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

bars = ax.barh(range(len(names)), values[::-1], color=colors[::-1], height=0.7, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n for n in reversed(names)], fontsize=10)
ax.set_xlabel('Mean |SHAP| Importance', fontsize=12, fontweight='bold')
ax.set_title('Global Feature Importance — SHAP (LightGBM LOSO)\nTop 20 of 74 multimodal features', fontsize=12, fontweight='bold', pad=10)

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, values[::-1])):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5, color=NAVY)

# Legend
legend_items = [
    mpatches.Patch(color=BLUE,   label='EEG (Brain waves)'),
    mpatches.Patch(color=PURPLE, label='EOG (Eye motion)'),
    mpatches.Patch(color=GREEN,  label='EMG (Muscle tone)'),
    mpatches.Patch(color=AMBER,  label='ECG/HRV (Heart)'),
    mpatches.Patch(color=CYAN,   label='Resp/SaO2'),
    mpatches.Patch(color=RED,    label='Clinical gates'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, max(values) * 1.18)
plt.tight_layout()
plt.savefig('results/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/shap_importance.png")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Per-subject benchmark bars (staging F1 breakdown)
# ────────────────────────────────────────────────────────────────────────────
print("Plotting per-subject benchmarks ...")
loso_df = pd.read_csv('results/loso_staging_results.csv')

x = np.arange(len(subjects))
w = 0.14
stage_cols = ['f1_Wake','f1_N1','f1_N2','f1_N3','f1_REM']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.patch.set_facecolor(WHITE)

# Left: stacked per-stage F1
ax = axes[0]
ax.set_facecolor(GRAY)
for i, (col, sname, color) in enumerate(zip(stage_cols, STAGE_NAMES, STAGE_COLORS)):
    vals = loso_df[col].values
    ax.bar(x + i*w - 2*w, vals, w, label=sname, color=color, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=11)
ax.set_ylabel('F1-Score (%)', fontsize=11)
ax.set_title('Per-Stage F1 by Subject\n(LOSO LightGBM)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 105)
ax.axhline(70, color=NAVY, linestyle='--', alpha=0.4, linewidth=1)
ax.text(4.6, 71.5, 'κ=0.75 target zone', fontsize=8, color=NAVY, alpha=0.7)
ax.grid(axis='y', alpha=0.3)

# Right: overall Acc / MacroF1 / Kappa×100
ax2 = axes[1]
ax2.set_facecolor(GRAY)
metrics = ['accuracy','macro_f1',]
kappa_scaled = loso_df['kappa'] * 100
ax2.bar(x - 0.2, loso_df['accuracy'].values, 0.2, label='Accuracy (%)', color=BLUE, edgecolor='white')
ax2.bar(x,       loso_df['macro_f1'].values,  0.2, label='Macro F1 (%)', color=GREEN, edgecolor='white')
ax2.bar(x + 0.2, kappa_scaled.values,         0.2, label="Cohen κ × 100", color=AMBER, edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(subjects, fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Overall Metrics by Subject\n(LOSO LightGBM)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylim(0, 105)
ax2.axhline(75, color=RED, linestyle='--', alpha=0.4, linewidth=1)
ax2.text(4.55, 76, 'Target', fontsize=8, color=RED, alpha=0.8)
ax2.grid(axis='y', alpha=0.3)

# Annotate SN5 distribution-shift note
ax2.annotate('SN5: 74% Wake\n(distribution shift)',
             xy=(4, 22), xytext=(3.2, 45),
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
             fontsize=8.5, color=RED, ha='center')

plt.suptitle('PSG Sleep Staging — LOSO Benchmark Results (LightGBM, 5 Subjects)', 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/per_subject_benchmark.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/per_subject_benchmark.png")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Modality ablation comparison (updating with real numbers from brain.md)
# ────────────────────────────────────────────────────────────────────────────
print("Plotting modality ablation ...")
ablation_data = {
    'Configuration':     ['EEG only','EOG only','EMG only','ECG only','Resp only','EOG+EMG','ALL 5 Signals\n(LightGBM)'],
    'Macro F1 (%)':      [31.2,       45.6,       23.5,      16.0,      6.0,        48.5,    54.65],
    'Cohen κ':           [0.21,       0.41,       0.12,      0.02,     -0.02,       0.45,    0.4586],
}
ab_df = pd.DataFrame(ablation_data)

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(GRAY)

bar_colors = [BLUE, PURPLE, GREEN, AMBER, CYAN, '#6366F1', RED]
bars = ax.bar(range(len(ab_df)), ab_df['Macro F1 (%)'],
              color=bar_colors, edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(ab_df)))
ax.set_xticklabels(ab_df['Configuration'], fontsize=10)
ax.set_ylabel('Macro F1-Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Modality Ablation Study — Single Signal vs. Fusion\n(LOSO LightGBM, 5,126 epochs)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3)

for bar, val, kap in zip(bars, ab_df['Macro F1 (%)'], ab_df['Cohen κ']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%\nκ={kap:.2f}', ha='center', fontsize=9, fontweight='bold', color=NAVY)

ax.axhline(48.5, color=NAVY, linestyle='--', alpha=0.5, linewidth=1.2)
ax.text(6.5, 49.5, 'EOG+EMG\nbest prev.', fontsize=7.5, color=NAVY, ha='right')

# Highlight best bar
bars[-1].set_edgecolor(NAVY)
bars[-1].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('results/modality_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/modality_ablation.png")

print("\n✅ All 4 plots saved to results/")
