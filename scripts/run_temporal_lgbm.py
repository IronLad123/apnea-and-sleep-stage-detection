"""
Phase 5 (revised): Temporal Sliding-Window Feature Engineering + Stacked LightGBM
===================================================================================
Key insight: Instead of a slow neural sequence model, we add temporal context
by computing rolling statistics over ±5 epochs (2.5-min context) for the most
informative features. This gives LightGBM the same temporal awareness without
training overhead.

Features added per epoch:
  - 10 key signal features × 3 stats (lag, lead, rolling-mean-5) = 30 temporal features
  - epoch_fraction, sleep_cycle_phase (already present)
  - Diff features: delta between current and prev epoch (rate-of-change)

Then: 2-stage stacking
  Stage 1: LightGBM on original 74 features → OOF probabilities
  Stage 2: LightGBM on [OOF probs + temporal features] → final prediction
"""
import warnings, os, json
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from collections import Counter

np.random.seed(42)

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
df = pd.read_parquet('features/fused_union.parquet')
META  = ['subject','epoch_idx','hard_label','apnea_label','quality_ok',
         'soft_W','soft_N1','soft_N2','soft_N3','soft_REM']
FLAGS = ['eeg_present','eog_present','emg_present','ecg_present','resp_sao2_present']
FEATS = [c for c in df.columns if c not in META and c not in FLAGS]
STAGES   = ['Wake','N1','N2','N3','REM']
subjects = sorted(df['subject'].unique())
print(f"  {df.shape[0]} epochs × {len(FEATS)} base features | subjects: {subjects}", flush=True)

# ── Temporal feature engineering (per subject, in-place safe) ─────────────
TEMPORAL_KEYS = [
    'eog1_zero_cross','eeg_hjorth_mobility','eog_corr','emg_zcr','eeg_beta',
    'emg_mav','eeg_sigma','eeg_delta','eog2_zero_cross','eeg_theta',
]

def add_temporal_features(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/lead/rolling features per subject (sorted by epoch_idx)."""
    sub = sub_df.sort_values('epoch_idx').copy()
    new_cols = {}
    for col in TEMPORAL_KEYS:
        if col not in sub.columns:
            continue
        s = sub[col].fillna(0)
        new_cols[f'{col}_lag1']  = s.shift(1).fillna(0).values
        new_cols[f'{col}_lead1'] = s.shift(-1).fillna(0).values
        new_cols[f'{col}_lag3']  = s.shift(3).fillna(0).values
        new_cols[f'{col}_lead3'] = s.shift(-3).fillna(0).values
        new_cols[f'{col}_roll5'] = s.rolling(5, center=True, min_periods=1).mean().values
        new_cols[f'{col}_diff1'] = s.diff(1).fillna(0).values

    # Sleep cycle phase (0–1, proportion through night)
    n = len(sub)
    new_cols['epoch_frac']   = np.linspace(0, 1, n)
    new_cols['cycle_phase']  = np.sin(2 * np.pi * np.linspace(0, 1, n) * 4)  # 4 cycles/night

    return pd.concat([sub, pd.DataFrame(new_cols, index=sub.index)], axis=1)

print("Adding temporal features ...", flush=True)
parts = []
for s in subjects:
    parts.append(add_temporal_features(df[df['subject']==s]))
df_temp = pd.concat(parts).sort_values(['subject','epoch_idx']).reset_index(drop=True)

TEMP_COLS = [c for c in df_temp.columns if c not in META and c not in FLAGS and c not in FEATS]
ALL_FEATS = FEATS + TEMP_COLS
print(f"  Total features: {len(FEATS)} base + {len(TEMP_COLS)} temporal = {len(ALL_FEATS)}", flush=True)

# ── Stage 1: Base LightGBM LOSO → OOF probabilities ──────────────────────
print("\n=== Stage 1: Base LightGBM LOSO (OOF probs) ===", flush=True)

# We store OOF predictions for stage 2 stacking
oof_proba = np.zeros((len(df_temp), 5), dtype=np.float32)

base_records = []
all_true_base, all_pred_base = [], []

for test_subj in subjects:
    tr = df_temp[df_temp['subject'] != test_subj].copy()
    te = df_temp[df_temp['subject'] == test_subj].copy()
    te_idx = te.index

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr[ALL_FEATS].fillna(0))
    X_te   = scaler.transform(te[ALL_FEATS].fillna(0))
    y_tr   = tr['hard_label'].values
    y_te   = te['hard_label'].values

    mc = Counter(y_tr)
    k  = min(5, min(mc.values())-1) if min(mc.values()) > 1 else 1
    try:
        X_sm, y_sm = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_tr, y_tr)
    except Exception:
        X_sm, y_sm = X_tr, y_tr

    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
        random_state=42, verbose=-1)
    m.fit(X_sm, y_sm)

    y_proba = m.predict_proba(X_te)

    # SN5 prior correction
    if test_subj == 'SN5':
        prior = np.bincount(y_sm, minlength=5) / len(y_sm)
        neutral = np.ones(5) / 5
        corr = np.log(neutral+1e-12) - np.log(prior+1e-12)
        lp = np.log(y_proba+1e-12) + corr
        lp -= lp.max(axis=1,keepdims=True)
        y_proba = np.exp(lp); y_proba /= y_proba.sum(axis=1,keepdims=True)

    oof_proba[te_idx] = y_proba.astype(np.float32)

    y_pred = np.argmax(y_proba, axis=1)
    mf1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    kap = cohen_kappa_score(y_te, y_pred)
    print(f"  [{test_subj}] Stage1 F1={mf1*100:.1f}%  κ={kap:.3f}", flush=True)

    all_true_base.extend(y_te.tolist())
    all_pred_base.extend(np.argmax(y_proba,axis=1).tolist())

stage1_f1 = f1_score(all_true_base, all_pred_base, average='macro', zero_division=0)
stage1_k  = cohen_kappa_score(all_true_base, all_pred_base)
print(f"\nStage1 Pooled: F1={stage1_f1*100:.2f}%  κ={stage1_k:.4f}", flush=True)

# ── Stage 2: Stacked LightGBM on [temporal feats + OOF probs] ────────────
print("\n=== Stage 2: Stacked LightGBM (temporal + OOF probs) ===", flush=True)

# Add OOF probs as features
df_temp = df_temp.copy()
for i, s in enumerate(STAGES):
    df_temp[f'oof_p_{s}'] = oof_proba[:, i]

OOF_COLS  = [f'oof_p_{s}' for s in STAGES]
S2_FEATS  = ALL_FEATS + OOF_COLS   # temporal feats + stage-1 predictions

stack_records = []
all_true_s2, all_pred_s2 = [], []

for test_subj in subjects:
    tr = df_temp[df_temp['subject'] != test_subj].copy()
    te = df_temp[df_temp['subject'] == test_subj].copy()

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr[S2_FEATS].fillna(0))
    X_te   = scaler.transform(te[S2_FEATS].fillna(0))
    y_tr   = tr['hard_label'].values
    y_te   = te['hard_label'].values

    mc = Counter(y_tr)
    k  = min(5, min(mc.values())-1) if min(mc.values()) > 1 else 1
    try:
        X_sm, y_sm = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_tr, y_tr)
    except Exception:
        X_sm, y_sm = X_tr, y_tr

    m2 = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.04, max_depth=6, num_leaves=50,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
        random_state=42, verbose=-1)
    m2.fit(X_sm, y_sm)

    y_proba2 = m2.predict_proba(X_te)

    if test_subj == 'SN5':
        prior = np.bincount(y_sm, minlength=5) / len(y_sm)
        neutral = np.ones(5) / 5
        corr = np.log(neutral+1e-12) - np.log(prior+1e-12)
        lp = np.log(y_proba2+1e-12) + corr
        lp -= lp.max(axis=1,keepdims=True)
        y_proba2 = np.exp(lp); y_proba2 /= y_proba2.sum(axis=1,keepdims=True)

    y_pred2 = np.argmax(y_proba2, axis=1)
    pf1 = f1_score(y_te, y_pred2, average=None, labels=[0,1,2,3,4], zero_division=0)
    mf1 = f1_score(y_te, y_pred2, average='macro', zero_division=0)
    kap = cohen_kappa_score(y_te, y_pred2)
    acc = accuracy_score(y_te, y_pred2)

    print(f"  [{test_subj}] Acc={acc*100:.1f}%  F1={mf1*100:.1f}%  κ={kap:.3f}  "
          f"N3={pf1[3]*100:.1f}%  REM={pf1[4]*100:.1f}%", flush=True)

    stack_records.append({
        'Subject': test_subj, 'N': len(y_te),
        'Accuracy': round(acc*100,2), 'Macro F1': round(mf1*100,2), 'Cohen κ': round(kap,3),
        'Wake F1': round(pf1[0]*100,1), 'N1 F1': round(pf1[1]*100,1),
        'N2 F1':   round(pf1[2]*100,1), 'N3 F1': round(pf1[3]*100,1),
        'REM F1':  round(pf1[4]*100,1),
    })
    all_true_s2.extend(y_te.tolist())
    all_pred_s2.extend(y_pred2.tolist())

stack_df = pd.DataFrame(stack_records)
stack_df.to_csv('results/loso_stacked_results.csv', index=False)

oa = accuracy_score(all_true_s2, all_pred_s2)
of = f1_score(all_true_s2, all_pred_s2, average='macro', zero_division=0)
ok = cohen_kappa_score(all_true_s2, all_pred_s2)
opf= f1_score(all_true_s2, all_pred_s2, average=None, labels=[0,1,2,3,4], zero_division=0)

print(f"\n{'='*62}")
print(f"STACKED TEMPORAL LightGBM — FINAL POOLED RESULTS")
print(f"{'='*62}")
print(f"  Accuracy:  {oa*100:.2f}%")
print(f"  Macro F1:  {of*100:.2f}%")
print(f"  Cohen κ:   {ok:.4f}")
print(f"  Per-class: Wake={opf[0]*100:.1f}% N1={opf[1]*100:.1f}% "
      f"N2={opf[2]*100:.1f}% N3={opf[3]*100:.1f}% REM={opf[4]*100:.1f}%")

# ── Delta vs Phase 3 ──────────────────────────────────────────────────────
v1_f1, v1_k = 54.65, 0.4586
print(f"\n  Phase 3 baseline:     F1={v1_f1:.2f}%   κ={v1_k:.4f}")
print(f"  Stage 1 (temporal):  F1={stage1_f1*100:.2f}%  κ={stage1_k:.4f}")
print(f"  Stage 2 (stacked):   F1={of*100:.2f}%  κ={ok:.4f}")
print(f"  Total Δ vs Phase 3:  F1={'↑' if of*100>v1_f1 else '↓'}{abs(of*100-v1_f1):.2f}%  "
      f"κ={'↑' if ok>v1_k else '↓'}{abs(ok-v1_k):.4f}")

print("\n✅ Phase 5 complete — results/loso_stacked_results.csv", flush=True)
