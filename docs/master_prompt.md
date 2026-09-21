# Master Engineering Prompt: Multi-Modal PSG Sleep Staging + Apnea Detection
**Dataset:** PhysioNet PSG-IPA v1.0.0 (20 recordings, 12 independent expert scorers per subject)
**Targets (per 30s epoch):** (1) Sleep stage — 5-class AASM (Wake, N1, N2, N3, REM); (2) Apnea/hypopnea presence — binary

---

## 0. Ground-Truth Contract (implement before any modeling)

**Sleep stage** — build BOTH label sets from the 12-scorer array and carry both through every notebook:
- **Hard:** `y_hard = mode(scorer_1..scorer_12)`. Define a tie-break rule (e.g. priority order Scorer 1 → 4 → 5) and log how many epochs hit the tie-break.
- **Soft:** `p = [p_Wake, p_N1, p_N2, p_N3, p_REM]`, the empirical distribution across the 12 scorers per epoch — used as the KL-divergence training target.
- Every LOSO evaluation in Notebooks 1–6 reports metrics against **both** targets (soft target evaluated via `argmax(p)`). State which target each number refers to — don't average them together.

**Apnea presence** — derived from PSG-IPA's respiratory event annotations (onset + duration), **not** from the 12 stage-scorers. Confirm the event format first, then: an epoch is positive if the annotated event overlaps ≥50% of its 30s window. This is a single fixed ground truth — there's no 12-scorer analog for it, so don't run it through the Phase-consensus logic built for staging.

---

## 1. Shared Infrastructure — `psg_utils/` (build first, import everywhere)

| Module | Responsibility |
|---|---|
| `channel_resolver.py` | Cz-M1 → C4-M1 fallback (per original spec). Also **writes** resolved channel name to a per-subject metadata file — don't just use it silently. |
| `epoching.py` | Assert `fs == 256.0`, 30s/7680-sample windows, truncate to `floor(L/7680)`. |
| `quality.py` | Artifact gate **before** feature extraction: flat-line detection (rolling std ≈ 0), clipping/saturation (samples pinned at ADC min/max), amplitude-outlier epochs (z-score vs subject's whole-night median). Output a per-epoch quality flag; decide once whether flagged epochs are dropped or down-weighted, apply that decision everywhere. |
| `labels.py` | Hard-vote + soft-consensus stage label builder; apnea event→epoch overlap labeler. |
| `splits.py` | Subject-level LOSO generator + the leakage assert: `assert len(set(train_df.subject) & set(test_df.subject)) == 0`. |
| `feature_io.py` | `save_features(df, signal_name, subject_id)` → `features/{signal_name}/{subject_id}.parquet`, keyed by `(subject_id, epoch_idx)`. Fusion becomes a plain join on those keys. |
| `repro.py` | Central seed-fixing (numpy/torch/sklearn) + package-version logging, called from every notebook's first cell. |

---

## 2. Per-Signal Notebook Contract (applies to all 5 signal notebooks below)

Every signal notebook uses this structure so they're directly comparable and fusion is mechanical, not bespoke:

1. **Setup** — import `psg_utils`, fix seeds, log package versions.
2. **Load + resolve + quality-gate.**
3. **Feature extraction** (table per notebook below) + a markdown explanation cell: what each feature captures physiologically and which stage(s)/task it discriminates.
4. **EDA** — per-stage feature boxplots, correlation heatmap.
5. **LOSO modeling** — Random Forest baseline (`class_weight='balanced'`) + GRU/BiLSTM sequence model (`seq_len=5`).
6. **Metrics** — accuracy, macro-F1, per-class F1, Cohen's κ, confusion matrix, against both hard-vote and soft-consensus stage targets.
7. **`save_features(...)` call** + closing markdown: this signal's standalone diagnostic power and where it's expected to be weak.

---

## 3. Notebook 1 — Central EEG · `01_EEG_Staging.ipynb`

25 features: delta/theta/alpha/**sigma (12–16Hz)**/beta absolute + 5 relative powers, mean/std/min/max/rms/range, **Hjorth (activity, mobility, complexity)**, **spectral entropy**, zcr.

```
Activity   = Var(x(t))
Mobility   = sqrt( Var(x'(t)) / Var(x(t)) )
Complexity = Mobility(x'(t)) / Mobility(x(t))
H_spectral = -Σ p_k · log2(p_k),   p_k = P(f_k) / Σ P(f_j)
```

Verification target: LOSO Random Forest N2 sensitivity improves ≥5pp over a delta/theta/alpha/beta-only baseline once sigma + Hjorth + entropy are added.

---

## 4. Notebook 2 — Dual EOG · `02_EOG_Sleep_Staging.ipynb`

34 features as originally scoped (mean/std/min/max/rms per channel, ocular delta/theta/alpha/beta, 8 relative powers, 6 cross-band ratios). Keep the ocular-specific bands (0.5–2 / 2–4 / 4–8 / 8–15Hz) — flag them explicitly in the explanation cell as **non-standard AASM bands, chosen for ocular dynamics**, distinct from the true AASM bands used in the EEG notebook.

Add a Cohen's κ cell. Existing benchmarks (RF 65.36% acc / 54.04% macro-F1; GRU seq=5: 66.10% / 57.54%) are the regression floor to beat, not a target to stop at.

---

## 5. Notebook 3 — EMG: Chin + Leg · `03_EMG_Staging.ipynb`

**Chin (8 features, primary — REM atonia / Wake motor bursts):** rms, mav, std, atonia_index, high_freq_power (15–100Hz), burst_count, zcr, p95.

**Leg — LAT/RAT (secondary — PLM/arousal proxy, ~5 features):** compute per-channel then combine as `max(LAT, RAT)` per epoch (either leg can trigger a PLM event — standard PSG convention; document this choice in the explanation cell).
- `leg_rms`
- `leg_burst_count` — amplitude > 2× baseline, 0.5–10s duration (WASM PLM morphology criteria)
- `leg_plm_index` — PLM events per hour
- `leg_burst_interval_regularity` — variance of inter-burst interval; PLMs occur in trains at 5–90s spacing, so low variance flags a genuine PLM train vs. incidental movement
- `leg_high_freq_power`

Filtering as original: 4th-order zero-phase Butterworth 10–100Hz + 50Hz notch.

Explanation cell must state: chin EMG drives stage classification (esp. REM), leg EMG is not expected to meaningfully move stage accuracy — its value shows up in arousal/PLM context, not in this notebook's own LOSO numbers.

Target: EMG-alone REM F1 ≥75% (chin-driven, as original).

---

## 6. Notebook 4 — Lead-II ECG / HRV · `04_ECG_Staging.ipynb`

10 features as originally scoped: hr_mean, hrv_sdnn, hrv_rmssd, hrv_pnn50, vlf/lf/hf power, lf_hf_ratio, **poincare_sd1, poincare_sd2** — implement the Poincaré features in the notebook itself, not just in the summary table (they were listed in the feature dictionary but missing from the original Phase 4 build list).

QRS: Pan-Tompkins or `scipy.signal.find_peaks(distance=150)`. RR outlier filter: reject RR <300ms or >2000ms. Interpolate RR to 4Hz (cubic spline) before Welch PSD for VLF/LF/HF.

Target: N3 detection driven by high HF power + low LF/HF ratio; REM by elevated LF/HF and irregular RR.

---

## 7. Notebook 5 — Respiration + SaO2 · `05_Resp_SaO2_Staging.ipynb` (dual-purpose)

**Stage features (8, as original):** resp_rate, resp_regularity, resp_amplitude_std, thoraco_abdominal_coherence, sao2_mean, sao2_min, sao2_desat_count (+ sao2_std to match the feature dictionary).

**Apnea-specific features (new — this is the primary apnea-discriminative signal):**
- Airflow amplitude reduction flag: ≥90% drop = apnea criterion, ≥30% = hypopnea criterion
- Event duration ≥10s flag
- Desaturation-linked-to-flow-drop co-occurrence (SaO2 drop within a lag window of a flow-drop event)

Run **two separate LOSO evaluations** in this one notebook:
(a) 5-class stage classification (expect this signal to underperform EEG/EOG — that's fine, report it honestly)
(b) binary apnea classification (expect this signal to lead all others)

Explanation cell must state which features feed which task.

---

## 8. Notebook 6 — Multimodal Fusion · `06_Multimodal_Fusion.ipynb` (multi-task)

Feature matrix via `feature_io` join on `(subject_id, epoch_idx)` across all 5 signal notebooks (~98 features: 25 EEG + 34 EOG + 13 EMG + 10 ECG + 12 Resp/SaO2, includes the added leg-EMG and apnea-specific respiration features).

**Architecture — shared representation, two heads, not two independent models:**
- *Tabular baseline:* two RF/LGBM models sharing the same feature matrix, one per target (acceptable first cut).
- *Sequential:* shared BiGRU/BiLSTM trunk (5-epoch window, input `(B, 5, 98)`) → softmax(5) stage head + sigmoid(1) apnea head. Joint loss: `L = CE(stage) + λ · BCE(apnea)`, tune λ.
- *Stretch:* end-to-end 1D-CNN on raw `[EEG, EOG, EMG]` (3×7680) → BiLSTM, same dual-head output.

**Ground-truth comparison (per confirmed decision):** train/evaluate the stage head against **both** hard-vote and soft-KL targets; report which performs better and by how much as an actual result, not just an implementation footnote.

**Ablation table (required):** drop each of the 5 modalities one at a time, report Δaccuracy/ΔF1 for **both** tasks. A flat apnea-F1 ablation when Resp+SaO2 is dropped would indicate a labeling or leakage bug — check before reporting results, since Resp+SaO2 should dominate the apnea task by construction.

---

## 9. Notebook 7 — Multi-Scorer Benchmark · `07_Multi_Scorer_Benchmark.ipynb`

Applies to the **stage** prediction only — there's no 12-scorer analog for apnea (single fixed ground truth from event annotations), so keep the two evaluations separate rather than reporting one blended κ.

- Cohen's κ of model vs. each of the 12 individual scorers; target κ ≥0.75, i.e. within the inter-human agreement band (baseline inter-scorer κ ≈0.76).
- For apnea: report sensitivity/specificity/AHI-agreement against the fixed ground truth instead — not a multi-scorer κ.

---

## 10. Feature Dictionary (updated)

| Modality | Count | Notes |
|---|---|---|
| EEG (Cz/C4) | 25 | unchanged from original spec |
| EOG (E1, E2) | 34 | unchanged |
| EMG (chin + leg) | 13 | 8 chin + 5 leg (LAT/RAT combined) — **updated for leg scope** |
| ECG | 10 | includes poincare_sd1/sd2 — **now actually implemented, not just tabled** |
| Respiration + SaO2 | 12 | 8 stage features + 4 apnea-specific — **updated for apnea scope** |
| **Total** | **94** | up from 85 in the original single-task spec |

---

## 11. Verification Checklist

- [ ] All 5 signal notebooks + fusion + benchmark notebook execute end-to-end
- [ ] Channel resolver correctly handles SN1 (Cz-M1) vs. others (C4-M1)
- [ ] Leakage assert passes in every train/test split
- [ ] Quality-gate epoch-drop rate reported per subject (flag if any subject loses >X% of epochs — likely upstream bug, not real artifact rate)
- [ ] Stage metrics: accuracy, macro-F1, per-class F1, κ — reported against **both** hard-vote and soft-KL targets
- [ ] Apnea metrics: accuracy, F1, sensitivity, specificity, AHI correlation vs. ground truth
- [ ] Ablation table present in fusion notebook, both tasks
- [ ] Target: fusion κ ≥0.75 on stage (hard-vote target); soft-KL-target κ reported as secondary
