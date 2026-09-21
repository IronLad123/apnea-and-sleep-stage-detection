# 🔬 Master Engineering Implementation Plan: Polysomnography (PSG) Sleep Staging

An end-to-end, clinically rigorous, multi-modal machine learning system for 5-class sleep staging ($\text{Wake}, \text{N1}, \text{N2}, \text{N3}, \text{REM}$) on PhysioNet PSG-IPA v1.0.0.

---

## User Review Required

> [!IMPORTANT]
> **Central EEG Channel Discrepancy Across Subjects**:
> In `Sleep_stages/PSG/`:
> * **SN1** uses `EEG Cz-M1` (midline central).
> * **SN2, SN3, SN4, SN5** use `EEG C4-M1` (right central).
> Both are standard AASM central derivations referenced to contralateral mastoid ($M_1$), but hardcoding either name causes a `KeyError`. The implementation plan incorporates an automatic **Channel Normalization Resolver** that dynamically selects the available central derivation (`Cz-M1` or `C4-M1`).

> [!WARNING]
> **EOG Frequency Band Standardisation**:
> In `02_EOG_Sleep_Staging.ipynb`, the frequency bands were shifted ($0.5-2\,\text{Hz}$ delta, $2-4\,\text{Hz}$ theta, $4-8\,\text{Hz}$ alpha, $8-15\,\text{Hz}$ beta). This successfully captures slow eye movements ($<2\,\text{Hz}$) and rapid saccades ($2-5\,\text{Hz}$). In this plan, we formalize these as **EOG-specific ocular dynamics bands** while maintaining true AASM frequency bands for EEG.

---

## Open Questions

> [!NOTE]
> 1. **Multi-Scorer Ground Truth Strategy**: The dataset includes 12 independent medical scorers per subject. For single-signal baselines, we use **Scorer 1** (or Scorer 4/5 where Scorer 1 is uncalibrated). For the final Multimodal Fusion model, do you prefer **Hard Majority Voting** ($\text{mode}$) or **Soft Probabilistic Consensus** (training with KL-divergence on the 12-scorer probability distribution)?
> 2. **EMG Channel Set**: For Signal 3, do you want to analyze `EMG chin` only (standard for sleep staging atonia), or also extract features from `EMG LAT` / `EMG RAT` (anterior tibialis leg muscles, primarily used for Periodic Limb Movements)?

---

## Proposed Changes & Execution Roadmap

The implementation follows a strict **Signal-by-Signal Deep Dive $\to$ Multimodal Fusion** architecture.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      MODULAR SIGNAL-BY-SIGNAL ARCHITECTURE                      │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────────┤
│ Signal 1: EEG   │ Signal 2: EOG   │ Signal 3: EMG   │ Signal 4: ECG (HRV)       │
│ psg_analysis    │ 02_EOG_Staging  │ 03_EMG_Staging  │ 04_ECG_Staging            │
│ [Cz/C4 Spindles]│ [Ocular Saccade]│ [Chin Atonia]   │ [Sympathovagal Balance]   │
└────────┬────────┴────────┬────────┴────────┬────────┴─────────────┬─────────────┘
         │                 │                 │                      │
         └─────────────────┴────────┬────────┴──────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Signal 5 & 6: Respiration & SaO2 Dynamics               │
       │ 05_Resp_SaO2_Staging.ipynb (Airflow, Effort, SpO2 Dips) │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Signal 7: Multimodal Fusion & Sequential Deep Learning  │
       │ 07_Multimodal_Fusion.ipynb (Tabular RF/LGBM + 1D-CNN)   │
       └─────────────────────────────────────────────────────────┘
```

---

### Phase 0: Biosignal Ingestion & Channel Normalization Engine
Standardizes diverse EDF headers, electrode montages, and temporal alignment across all 20 recordings.

#### [NEW] [channel_resolver.py](file:///Users/omsrivastava/Documents/healthcare_project/utils/channel_resolver.py)
* **Objective**: Automatically map subject-specific channel variations to standardized physiological derivations.
* **Explanation & Clinical Rationale**: Different sleep labs place electrodes at $C_z$ or $C_4$ depending on scalp prep. Both capture Central sensorimotor rhythms (K-complexes, spindles).
* **Mathematical & Signal Rules**:
  * Sampling rate assertion: Verify $f_s = 256.0\,\text{Hz}$ across all channels.
  * Epoch segmentation: Exactly $N = 30 \times 256 = 7,680$ samples per epoch.
  * Truncation rule: Truncate signal to integer multiple of epoch length:
    $$N_{\text{epochs}} = \left\lfloor \frac{L}{7680} \right\rfloor$$
* **Failure Modes**: Missing channel in EDF header. Solved by falling back to alternate derivation ($F_4 \to C_z \to C_4$).

---

### Phase 1: Signal 1 (Central EEG) — Spindle & Entropy Enhancement
Upgrades `psg_analysis.ipynb` with clinically critical spectral bands and non-linear complexity metrics.

#### [MODIFY] [psg_analysis.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/psg_analysis.ipynb)
* **Objective**: Add Sigma band ($12-16\,\text{Hz}$), Hjorth parameters, and Spectral Entropy to single-subject and multi-subject EEG pipelines.
* **Explanation & Clinical Rationale**:
  * **Sigma Band ($12-16\,\text{Hz}$)**: Neurophysiological marker of **Sleep Spindles** generated by thalamocortical loops during **Stage N2**. Without Sigma, N1 and N2 are frequently confused.
  * **Hjorth Activity, Mobility, Complexity**:
    $$\text{Activity} = \sigma_x^2 = \text{Var}(x(t))$$
    $$\text{Mobility} = \sqrt{\frac{\text{Var}(x'(t))}{\text{Var}(x(t))}} \quad (\text{mean frequency proxy})$$
    $$\text{Complexity} = \frac{\text{Mobility}(x'(t))}{\text{Mobility}(x(t))} \quad (\text{frequency variation proxy})$$
  * **Spectral Entropy**: Quantifies rhythm disorder. High in desynchronized Wake and REM; low in synchronized deep N3 delta waves:
    $$H_{\text{spectral}} = -\sum_{k} p_k \log_2(p_k), \quad p_k = \frac{P(f_k)}{\sum_j P(f_j)}$$
* **Verification**: Run 5-subject LOSO Random Forest on EEG. Verify that N2 sensitivity increases by $\ge 5\%$.

---

### Phase 2: Signal 2 (Dual EOG) — Ocular Dynamics Consolidation
Verifies and locks in the existing EOG pipeline benchmarks.

#### [MODIFY] [02_EOG_Sleep_Staging.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/02_EOG_Sleep_Staging.ipynb)
* **Objective**: Document the 34-feature ocular feature table, ensure reproducible random seeds, and compute Cohen's Kappa ($\kappa$).
* **Explanation & Clinical Rationale**:
  * Out-of-phase deflections between $E_1-M_2$ and $E_2-M_2$ indicate horizontal conjugate eye movements.
  * Slow rolling movements ($<1\,\text{Hz}$) define sleep onset (Stage N1).
  * Rapid jerky saccades ($>2\,\text{Hz}$) accompanied by EEG theta indicate Stage REM.
* **Benchmarks Achieved**:
  * Random Forest LOSO: **65.36% Accuracy**, **54.04% Macro F1**.
  * GRU ($sequence\_length=5$): **66.10% Accuracy**, **57.54% Macro F1**.
* **Action**: Add Cohen's $\kappa$ metric calculation cell to formalize clinical agreement.

---

### Phase 3: Signal 3 (Submental Chin EMG) — Muscle Tone & REM Atonia
Creates the dedicated pipeline for detecting Stage REM atonia and Wake motor bursts.

#### [NEW] [03_EMG_Sleep_Staging.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/03_EMG_Sleep_Staging.ipynb)
* **Objective**: Build an independent EMG sleep staging pipeline on `EMG chin`.
* **Explanation & Clinical Rationale**:
  * **The REM Atonia Rule**: Under AASM scoring, REM cannot be confirmed without demonstrating lowest submental muscle tone of the entire night.
  * **Wake Motor Activation**: Awake patients talk, swallow, and move, producing high-amplitude, wideband EMG spikes.
* **Signal Processing & Feature Extraction (8 features per epoch)**:
  1. **Filtering**: 4th-order zero-phase Butterworth bandpass $10.0-100.0\,\text{Hz}$ + $50\,\text{Hz}$ notch filter to eliminate powerline hum:
     $$y[n] = \text{filtfilt}(b, a, x[n])$$
  2. **`emg_rms`**: Root-mean-square energy:
     $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2}$$
  3. **`emg_mav`**: Mean Absolute Value: $\frac{1}{N} \sum |x[n]|$.
  4. **`emg_atonia_index`**: Compute RMS in 1-second mini-windows. The atonia index is the fraction of 1-second sub-epochs where $\text{RMS} < 1.5 \times \text{baseline\_noise}$.
  5. **`emg_high_freq_power`**: Integrated Welch PSD in the true muscle firing band ($15-100\,\text{Hz}$).
  6. **`emg_burst_count`**: Number of amplitude envelope peaks exceeding $3 \times \text{median}(\text{envelope})$. Detects phasic muscle twitches in REM vs tonic tone in Wake.
  7. **`emg_zcr`**: Zero-crossing rate: counts transitions through zero level. High for genuine high-frequency muscle activation; low for baseline wander.
  8. **`emg_p95`**: 95th percentile amplitude, robust to single isolated transient artifacts.
* **Model & LOSO Validation**:
  * Train Random Forest and GRU ($seq=5$) across subjects SN1–SN5.
  * Target: Benchmark EMG-alone REM F1-score ($\ge 75\%$).

---

### Phase 4: Signal 4 (Lead-II ECG) — Autonomic Heart Rate Variability (HRV)
Extracts sympathovagal autonomic dynamics across sleep stages.

#### [NEW] [04_ECG_Sleep_Staging.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/04_ECG_Sleep_Staging.ipynb)
* **Objective**: Build an ECG sleep staging pipeline using R-peak detection and Heart Rate Variability (HRV).
* **Explanation & Clinical Rationale**:
  * The autonomic nervous system shifts with sleep depth:
    * **Stage N3**: Vagal (parasympathetic) dominance $\to$ heart rate slows, RR intervals become highly uniform, High Frequency (HF) power surges.
    * **Stage REM**: Autonomic storm $\to$ irregular tachycardia/bradycardia fluctuations, High sympathetic bursts, elevated LF/HF ratio.
    * **Wake**: High heart rate, sympathetic dominance.
* **Signal Processing & Feature Extraction (10 features per epoch)**:
  1. **QRS Detection**: Bandpass $0.5-40\,\text{Hz}$, derivative filter, squaring, moving window integration (Pan-Tompkins) or `scipy.signal.find_peaks(distance=150)` ($\ge 250\,\text{ms}$ refractory period).
  2. **Physiological RR Outlier Filtering**: Reject physiologically impossible intervals ($RR < 300\,\text{ms}$ or $RR > 2000\,\text{ms}$).
  3. **Time-Domain HRV**:
     * `hr_mean`: Average heart rate in beats per minute ($60 / \overline{RR}$).
     * `hrv_sdnn`: Standard deviation of normal-to-normal intervals (overall HRV):
       $$\text{SDNN} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (RR_k - \overline{RR})^2}$$
     * `hrv_rmssd`: Root mean square of successive RR differences (parasympathetic/vagal marker):
       $$\text{RMSSD} = \sqrt{\frac{1}{K-1} \sum_{k=1}^{K-1} (RR_{k+1} - RR_k)^2}$$
     * `hrv_pnn50`: Percentage of consecutive RR intervals differing by $>50\,\text{ms}$.
  4. **Frequency-Domain HRV**:
     * Interpolate uneven RR series to 4 Hz uniform grid using cubic spline.
     * Welch PSD:
       * **VLF ($0.0033-0.04\,\text{Hz}$)**: Thermoregulation, renin-angiotensin.
       * **LF ($0.04-0.15\,\text{Hz}$)**: Baroreceptor reflex (sympathetic + parasympathetic).
       * **HF ($0.15-0.40\,\text{Hz}$)**: Respiratory Sinus Arrhythmia (vagal parasympathetic).
       * **`lf_hf_ratio`**: Sympathovagal balance index (high in Wake/REM, lowest in N3).
* **Model & LOSO Validation**: Evaluate 5-subject LOSO classification. Verify N3 detection via high HF power and low LF/HF ratio.

---

### Phase 5: Signal 5 & 6 (Respiration & SaO2 Dynamics)
Captures ventilatory drive stability and oxygenation drops.

#### [NEW] [05_Resp_SaO2_Sleep_Staging.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/05_Resp_SaO2_Sleep_Staging.ipynb)
* **Objective**: Extract breathing patterns from nasal airflow, chest/abdominal bands, and SaO2 oximetry.
* **Explanation & Clinical Rationale**:
  * **N3 Sleep**: Metabolic breathing driven strictly by $P_{\text{CO}_2}$; breathing rate is clockwork regular with zero variation.
  * **REM Sleep**: Behavioral breathing with rapid, shallow, irregular respiratory amplitude.
  * **SaO2**: Stable high saturation in N3 ($96-98\%$); repetitive desaturation drops in sleep-disordered breathing.
* **Features Extracted (8 features per epoch)**:
  * `resp_rate`: Dominant respiratory frequency (FFT peak in $0.1-0.5\,\text{Hz}$, breaths/min).
  * `resp_regularity`: Variance of peak-to-peak breath durations (lowest in N3).
  * `resp_amplitude_std`: Depth variation across the 30-second epoch.
  * `thoraco_abdominal_coherence`: Correlation between `Resp chest` and `Resp abdomen`. In-phase during normal sleep; out-of-phase (paradoxical) during airway obstruction.
  * `sao2_mean`, `sao2_min`: Oxygenation level.
  * `sao2_desat_count`: Count of drops $\ge 3\%$ below rolling baseline.
* **Model & LOSO Validation**: Evaluate capacity of respiration to isolate deep N3 and fragmented sleep.

---

### Phase 6: Signal 7 — Multimodal Fusion & Deep Sequential Learning
Merges all modalities into a unified multimodal classifier.

#### [NEW] [07_Multimodal_Fusion.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/07_Multimodal_Fusion.ipynb)
* **Objective**: Fuse all engineered features across modalities into an 85+ feature matrix and train SOTA sequential architectures.
* **Feature Schema**:
  $$\mathbf{X}_{\text{multimodal}} = \left[ \mathbf{X}_{\text{EEG}}^{(25)} \parallel \mathbf{X}_{\text{EOG}}^{(34)} \parallel \mathbf{X}_{\text{EMG}}^{(8)} \parallel \mathbf{X}_{\text{ECG}}^{(10)} \parallel \mathbf{X}_{\text{Resp}}^{(8)} \right] \in \mathbb{R}^{N_{\text{epochs}} \times 85}$$
* **Model Zoo & Progression**:
  1. **Early Tabular Fusion**:
     * **Random Forest** (200 trees, balanced weights, LOSO).
     * **LightGBM / XGBoost** (with feature importance rankings across all 6 modalities).
  2. **Sequential Deep Learning (GRU / BiLSTM)**:
     * Input shape: $(B, 5, 85)$ (5 consecutive 30-second epochs).
     * Bidirectional GRU (64 units) + Temporal Attention layer + Softmax output.
  3. **End-to-End 1D-CNN Waveform Fusion (Advanced SOTA)**:
     * Raw 3-channel biosignal input: `[EEG, EOG, EMG]` ($3 \times 7680$ samples).
     * Multi-scale 1D-CNN filters ($50\,\text{ms}$ and $200\,\text{ms}$ kernels) for intra-epoch feature extraction $\to$ BiLSTM for inter-epoch transitions.

---

### Phase 7: Multi-Scorer Consensus & Clinical Validation
Benchmarking against the 12 expert human scorers in PSG-IPA.

#### [NEW] [08_Multi_Scorer_Benchmark.ipynb](file:///Users/omsrivastava/Documents/healthcare_project/08_Multi_Scorer_Benchmark.ipynb)
* **Objective**: Evaluate model predictions against all 12 human expert scorers.
* **Explanation & Clinical Rationale**:
  * Human expert agreement on PSG-IPA is not 100% (inter-scorer Cohen's $\kappa \approx 0.76$).
  * Training against a single scorer forces the model to learn one individual's subjective biases.
* **Experiments**:
  1. **Majority Voting Ground Truth**:
     $$y_{\text{consensus}} = \operatorname{mode}(y_{\text{scorer}_1}, y_{\text{scorer}_2}, \dots, y_{\text{scorer}_{12}})$$
  2. **Soft Label Consensus & Uncertainty**:
     Compute the empirical probability vector $\mathbf{p} = [p_W, p_{N1}, p_{N2}, p_{N3}, p_{REM}]$ across the 12 scorers for each epoch.
     Train the sequential neural network using **Kullback-Leibler (KL) Divergence Loss**:
     $$\mathcal{L}_{\text{KL}}(\mathbf{p}, \hat{\mathbf{p}}) = \sum_{c=0}^4 p_c \log \left(\frac{p_c}{\hat{p}_c}\right)$$
  3. **Clinical Agreement Benchmarking**:
     * Compute Cohen's $\kappa$ of our model vs each individual human scorer.
     * Show that the AI model's agreement with Scorer $k$ falls within the inter-human agreement distribution.

---

## Comprehensive Feature Dictionary (85 Total Features)

| Modality | Features | Count | Physiological Diagnostic Purpose |
|---|---|---|---|
| **EEG (`Cz/C4`)** | `delta`, `theta`, `alpha`, `sigma`, `beta`, `total`, 5 relative powers, `mean`, `std`, `min`, `max`, `rms`, `range`, Hjorth (3), `spectral_entropy`, `zcr` | 25 | Delta $\to$ N3 deep sleep; Sigma $\to$ N2 spindles; Alpha $\to$ Wake; Spectral entropy $\to$ EEG synchronization. |
| **EOG (`E1`, `E2`)** | `mean`, `std`, `min`, `max`, `rms` (x2), `delta`, `theta`, `alpha`, `beta` (x2), 8 relative powers, 6 cross-band frequency ratios (`theta/delta`, etc.) | 34 | Out-of-phase saccades $\to$ REM; slow eye drifts $\to$ N1 sleep onset; blinks $\to$ Wake. |
| **EMG (`chin`)** | `rms`, `mav`, `std`, `high_freq_power` ($15-100\,\text{Hz}$), `atonia_index`, `burst_count`, `zcr`, `p95` | 8 | Complete motor atonia $\to$ REM; tonic muscle tone $\to$ Wake/NREM; transient twitches. |
| **ECG** | `hr_mean`, `hrv_sdnn`, `hrv_rmssd`, `hrv_pnn50`, `vlf`, `lf`, `hf`, `lf_hf_ratio`, `poincare_sd1`, `poincare_sd2` | 10 | Parasympathetic dominance (high HF, low LF/HF) $\to$ N3; autonomic instability $\to$ REM; high HR $\to$ Wake. |
| **Respiration** | `resp_rate`, `resp_regularity`, `resp_amplitude_std`, `thoraco_abdominal_coherence` | 4 | Regular metabolic breathing $\to$ N3; irregular rate $\to$ REM; paradoxical effort $\to$ Apnea. |
| **SaO2** | `sao2_mean`, `sao2_min`, `sao2_std`, `sao2_desat_count` | 4 | Baseline oxygenation; desaturation drops $\ge 3\%$ indicate sleep fragmentation. |
| **Total** | | **85** | Complete clinical multi-modal coverage. |

---

## Verification Plan

### Automated Verification Protocols
1. **Pipeline Execution Tests**:
   * Each notebook must execute end-to-end without errors using virtualenv Python:
     `/Users/omsrivastava/Documents/healthcare_project/psg_env/bin/python`
2. **Channel Resolution Test**:
   * Confirm that `channel_resolver.py` seamlessly handles `SN1` (`EEG Cz-M1`) and `SN2-SN5` (`EEG C4-M1`).
3. **Data Leakage Assertions**:
   * Include strict automated assert checks in every script:
     ```python
     assert len(set(train_df["subject"]).intersection(set(test_df["subject"]))) == 0
     ```
4. **Clinical Metric Verification**:
   * Compute multi-class metrics on Leave-One-Subject-Out predictions:
     * Overall Accuracy
     * Macro F1-score
     * Per-Class F1 for Wake, N1, N2, N3, REM
     * **Cohen's Kappa ($\kappa$)** using `sklearn.metrics.cohen_kappa_score`
     * Target: Multimodal Fusion $\kappa \ge 0.75$ (substantial human-equivalent clinical agreement).
