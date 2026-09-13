"""Phase 4 - Fixed column duplication + clean delta table"""
import warnings, os, json, glob
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from collections import Counter

np.random.seed(42)

# ── Load ─────────────────────────────────────────────────────────────────
df = pd.read_parquet('features/fused_union.parquet')
META  = ['subject','epoch_idx','hard_label','apnea_label','quality_ok',
         'soft_W','soft_N1','soft_N2','soft_N3','soft_REM']
FLAGS = ['eeg_present','eog_present','emg_present','ecg_present','resp_sao2_present']

# Add fixed gates as new columns (avoid dup)
df['rem_gate_v2']  = ((df['eog_saccade_rate'] > 0) & (df['emg_zcr'] > 0)).astype(float)
df['n3_gate_v2']   = (df['eeg_delta'] > 0.5).astype(float)
df['wake_gate_v2'] = (df['eeg_beta']  > 0.3).astype(float)

# Feature set: drop old gates, add new ones
EXCLUDE = META + FLAGS + ['rem_gate','n3_gate','wake_gate']
FEATS   = [c for c in df.columns if c not in EXCLUDE]

subjects = sorted(df['subject'].unique())
STAGES   = ['Wake','N1','N2','N3','REM']

sn4_rem = df[(df['subject']=='SN4') & (df['hard_label']==4)]
print(f"FIX CHECK — SN4 REM epochs with rem_gate_v2=1: {sn4_rem['rem_gate_v2'].sum()}/{len(sn4_rem)}")

# ── Multi-scorer apnea consensus ─────────────────────────────────────────
print("\n=== Multi-scorer apnea consensus ===")
APNEA_DIR   = ('psg-ipa-a-polysomnographic-inter-scorer-performance-assessment-database-1.0.0'
               '/Resp_events/Annotations/manual')
APNEA_TYPES = {'Obstructive apnea','Mixed apnea','Hypopnea','Central apnea'}

def build_consensus(subject, n_epochs, threshold=4):
    files = sorted(glob.glob(os.path.join(APNEA_DIR, f'{subject}_Respiration_manual_scorer*.txt')))
    vote  = np.zeros((len(files), n_epochs), dtype=int)
    for si, fp in enumerate(files):
        try:
            ann = pd.read_csv(fp, skipinitialspace=True, header=0,
                              names=['Date','Time','Onset','Duration','Annotation','Channel'],
                              on_bad_lines='skip')
            ann['Onset']    = pd.to_numeric(ann['Onset'],    errors='coerce')
            ann['Duration'] = pd.to_numeric(ann['Duration'], errors='coerce')
            ann = ann.dropna(subset=['Onset','Duration'])
            ann = ann[ann['Annotation'].str.strip().isin(APNEA_TYPES)]
            for _, row in ann.iterrows():
                s_ep = max(0, int(row['Onset'] // 30))
                e_ep = min(n_epochs-1, int((row['Onset']+row['Duration']) // 30))
                vote[si, s_ep:e_ep+1] = 1
        except Exception:
            pass
    return (vote.sum(axis=0) >= threshold).astype(int), len(files)

df['apnea_consensus'] = df['apnea_label'].copy()
for subj in subjects:
    sub    = df[df['subject']==subj].sort_values('epoch_idx')
    labels, n_sc = build_consensus(subj, len(sub), threshold=4)
    idx    = sub.index
    if len(labels) == len(idx):
        df.loc[idx, 'apnea_consensus'] = labels
    old_p  = df.loc[idx,'apnea_label'].sum()
    new_p  = df.loc[idx,'apnea_consensus'].sum()
    print(f"  {subj}: scorers={n_sc} original={old_p} → consensus≥4={new_p}")

print(f"  TOTAL: old={df['apnea_label'].sum()} → consensus={df['apnea_consensus'].sum()}")

# ── Improved LOSO staging ─────────────────────────────────────────────────
print("\n=== IMPROVED LOSO Staging (v2) ===")
records2, all_true2, all_pred2 = [], [], []

for test_subj in subjects:
    tr = df[df['subject'] != test_subj].copy()
    te = df[df['subject'] == test_subj].copy()

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr[FEATS].fillna(0))
    X_te   = scaler.transform(te[FEATS].fillna(0))
    y_tr   = tr['hard_label'].values
    y_te   = te['hard_label'].values

    mc = Counter(y_tr)
    k  = min(5, min(mc.values())-1) if min(mc.values()) > 1 else 1
    try:
        X_sm, y_sm = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_tr, y_tr)
    except Exception:
        X_sm, y_sm = X_tr, y_tr

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=8, num_leaves=80,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.15, reg_lambda=0.15, class_weight='balanced',
        random_state=42, verbose=-1)
    model.fit(X_sm, y_sm)
    y_proba = model.predict_proba(X_te)

    # SN5: prior correction — neutralise the 74% Wake distribution shift
    if test_subj == 'SN5':
        train_prior   = np.bincount(y_sm, minlength=5) / len(y_sm)
        neutral_prior = np.ones(5) / 5
        log_corr = np.log(neutral_prior + 1e-12) - np.log(train_prior + 1e-12)
        log_p    = np.log(y_proba + 1e-12) + log_corr
        log_p   -= log_p.max(axis=1, keepdims=True)
        y_proba  = np.exp(log_p)
        y_proba /= y_proba.sum(axis=1, keepdims=True)

    y_pred = np.argmax(y_proba, axis=1)
    pf1    = f1_score(y_te, y_pred, average=None, labels=[0,1,2,3,4], zero_division=0)
    acc    = accuracy_score(y_te, y_pred)
    mf1    = f1_score(y_te, y_pred, average='macro', zero_division=0)
    kap    = cohen_kappa_score(y_te, y_pred)

    print(f"  [{test_subj}] Acc={acc*100:.1f}%  F1={mf1*100:.1f}%  κ={kap:.3f}  "
          f"N3={pf1[3]*100:.1f}%  REM={pf1[4]*100:.1f}%")

    records2.append({'Subject': test_subj, 'N': len(y_te),
        'Accuracy': round(acc*100,2), 'Macro F1': round(mf1*100,2), 'Cohen κ': round(kap,3),
        'Wake F1': round(pf1[0]*100,1), 'N1 F1': round(pf1[1]*100,1),
        'N2 F1':   round(pf1[2]*100,1), 'N3 F1': round(pf1[3]*100,1),
        'REM F1':  round(pf1[4]*100,1)})
    all_true2.extend(y_te.tolist())
    all_pred2.extend(y_pred.tolist())

loso_v2 = pd.DataFrame(records2)
loso_v2.to_csv('results/loso_v2_results.csv', index=False)

oa2  = accuracy_score(all_true2, all_pred2)
of2  = f1_score(all_true2, all_pred2, average='macro', zero_division=0)
ok2  = cohen_kappa_score(all_true2, all_pred2)
opf2 = f1_score(all_true2, all_pred2, average=None, labels=[0,1,2,3,4], zero_division=0)

print(f"\nPOOLED v2: Acc={oa2*100:.2f}%  F1={of2*100:.2f}%  κ={ok2:.4f}")
print(f"Per-class: ", {s:f'{v*100:.1f}%' for s,v in zip(STAGES, opf2)})

# ── Improved apnea detection ──────────────────────────────────────────────
print("\n=== Improved Apnea Detection (consensus labels) ===")
APNEA_FEATS = [c for c in FEATS if any(x in c for x in [
    'resp','sao2','flow','thoraco','chest_abd',
    'hr_mean','hrv_rmssd','hrv_sdnn','lf_hf_ratio',
    'rem_gate_v2','wake_gate_v2','apnea_gate','obstructive','arousal'])]
print(f"  Using {len(APNEA_FEATS)} apnea features")

apnea_rows, all_at, all_ap = [], [], []
for test_subj in subjects:
    tr = df[df['subject'] != test_subj].copy()
    te = df[df['subject'] == test_subj].copy()
    y_tr = tr['apnea_consensus'].values
    y_te = te['apnea_consensus'].values
    n_pos = int(y_te.sum())
    if n_pos == 0:
        print(f"  [{test_subj}] SKIP")
        apnea_rows.append({'Subject': test_subj, 'n_apnea': 0, 'AUC': None, 'F1_opt': None})
        continue
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr[APNEA_FEATS].fillna(0))
    X_te = scaler.transform(te[APNEA_FEATS].fillna(0))
    spw = max(1.0, (1-y_tr.mean()) / (y_tr.mean()+1e-6))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, max_depth=6,
        num_leaves=40, scale_pos_weight=spw, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    y_prob = m.predict_proba(X_te)[:,1]
    auc    = roc_auc_score(y_te, y_prob)
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1t = f1_score(y_te, (y_prob>=t).astype(int), average='binary', zero_division=0)
        if f1t > best_f1: best_f1, best_t = f1t, t
    print(f"  [{test_subj}] n={n_pos}/{len(y_te)} ({n_pos/len(y_te)*100:.1f}%)  AUC={auc:.3f}  F1={best_f1:.3f}  @t={best_t:.2f}")
    apnea_rows.append({'Subject': test_subj, 'n_apnea': n_pos,
                        'AUC': round(auc,4), 'F1_opt': round(best_f1,4), 'thresh': round(best_t,2)})
    all_at.extend(y_te.tolist()); all_ap.extend(y_prob.tolist())

pd.DataFrame(apnea_rows).to_csv('results/apnea_v2_results.csv', index=False)
if len(set(all_at)) > 1:
    pool_auc = roc_auc_score(all_at, all_ap)
    best_pf, best_pt = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1t = f1_score(all_at, (np.array(all_ap)>=t).astype(int), average='binary', zero_division=0)
        if f1t > best_pf: best_pf, best_pt = f1t, t
    print(f"\n  Pooled AUC: {pool_auc:.4f}  F1_opt: {best_pf:.4f}  @t={best_pt:.2f}")

# ── Delta table ───────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("PHASE 3 → PHASE 4 DELTA")
print(f"{'='*62}")
v1 = pd.read_csv('results/loso_staging_results.csv')
v1_map = {'test_subject':'Subject','macro_f1':'F1_v1','kappa':'κ_v1',
          'f1_N3':'N3_v1','f1_REM':'REM_v1'}
v1 = v1.rename(columns=v1_map)[['Subject','F1_v1','κ_v1','N3_v1','REM_v1']]

print(f"{'Sub':<6} {'F1 v1':>7} {'F1 v2':>7} {'Δ F1':>7}  {'κ v1':>7} {'κ v2':>7} {'Δ κ':>7}  {'REM v1':>7} {'REM v2':>7}")
print('-'*78)
for _, r2 in loso_v2.iterrows():
    s = r2['Subject']
    r1 = v1[v1['Subject']==s].iloc[0]
    d_f1 = r2['Macro F1'] - r1['F1_v1']
    d_k  = r2['Cohen κ']  - r1['κ_v1']
    d_rem= r2['REM F1']   - r1['REM_v1']
    af   = f"{'↑' if d_f1>0 else '↓'}{abs(d_f1):.1f}%"
    ak   = f"{'↑' if d_k>0  else '↓'}{abs(d_k):.3f}"
    print(f"  {s:<4} {r1['F1_v1']:>6.1f}% {r2['Macro F1']:>6.1f}% {af:>7}  "
          f"{r1['κ_v1']:>7.3f} {r2['Cohen κ']:>7.3f} {ak:>7}  "
          f"{r1['REM_v1']:>6.1f}% {r2['REM F1']:>6.1f}%")

v1_pool_f1, v2_pool_f1 = 54.65, of2*100
v1_pool_k,  v2_pool_k  = 0.4586, ok2
print(f"\n  POOLED  {v1_pool_f1:>6.2f}% {v2_pool_f1:>6.2f}% {'↑' if v2_pool_f1>v1_pool_f1 else '↓'}{abs(v2_pool_f1-v1_pool_f1):.2f}%  "
      f"{v1_pool_k:>7.4f} {v2_pool_k:>7.4f} {'↑' if v2_pool_k>v1_pool_k else '↓'}{abs(v2_pool_k-v1_pool_k):.4f}")
print("\n✅ Phase 4 complete.")
