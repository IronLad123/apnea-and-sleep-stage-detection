# Output Images Catalog

This directory consolidates all high-resolution figures, benchmark visualizations, architectural diagrams, and exploratory data analysis plots produced by the project.

---

## 1. Primary Benchmark & Diagnostic Visualizations

| Image File | Description | Clinical / Research Purpose |
|:---|:---|:---|
| [`01_final_dashboard.png`](01_final_dashboard.png) | **Master 4-Panel Benchmark Dashboard** | Displays (A) Per-stage F1 across all 5 subjects, (B) Overall accuracy, Macro F1, and Cohen's Kappa, (C) Full 8-stage modality ablation study, and (D) 12-scorer consensus apnea detection ROC curves and optimal F1 points. |
| [`02_per_subject_benchmark.png`](02_per_subject_benchmark.png) | **Per-Subject Performance Profiles** | Dual-panel comparison showing stage-wise breakdown (Wake, N1, N2, N3, REM) alongside overall metrics for Subjects SN1–SN5 under Leave-One-Subject-Out (LOSO) validation. Highlights Subject SN3 reaching near-human consensus ($\kappa = 0.724$). |
| [`03_confusion_matrix.png`](03_confusion_matrix.png) | **Normalized 5-Class Confusion Matrix** | Evaluates pooled classification across all 5,126 epochs. Demonstrates strong deep sleep recognition (N3 = 86%, N2 = 68%, REM = 67%) and visualizes transitional boundary ambiguity in Stage N1. |
| [`04_shap_feature_importance.png`](04_shap_feature_importance.png) | **Global SHAP Feature Importance** | Top-20 features color-coded by physiological modality (`EOG`, `EEG`, `EMG`, `ECG`, `Resp`). Mathematically confirms that decisions are governed by AASM-aligned biomarkers (EOG zero-crossing rate, EEG Hjorth mobility, conjugate correlation, and EMG atonia). |
| [`05_modality_ablation.png`](05_modality_ablation.png) | **Systematic Modality Ablation Study** | Compares unimodal baselines against paired and full multimodal fusion. Highlights EOG as the single strongest standalone sensor (45.6% Macro F1) and demonstrates the +26.3% F1 gain of full multimodal integration over central EEG alone. |

---

## 2. Theoretical & Architectural Diagrams

| Image File | Description | Purpose in Papers & Presentations |
|:---|:---|:---|
| [`06_clinical_problem_overview.png`](06_clinical_problem_overview.png) | **Clinical Problem & Diagnostic Backlog** | Visualizes the overnight hospital PSG setup, the manual scoring bottleneck (2–4 hours per patient), human inter-rater subjectivity ($\kappa = 0.65\text{--}0.78$), and the global diagnostic gap (80% undiagnosed OSA). |
| [`07_traditional_vs_proposed_framework.png`](07_traditional_vs_proposed_framework.png) | **Traditional vs. Proposed Comparison** | Contrasts conventional pipelines (inner-join data discarding, single-scorer boundary errors, random shuffling leakage) against our proposed framework (fault-tolerant union join, 12-scorer consensus, strict LOSO). |
| [`08_dataset_cohort_breakdown.png`](08_dataset_cohort_breakdown.png) | **PhysioNet PSG-IPA Cohort Breakdown** | Breaks down the 5,126 epochs across 5 subjects, showing stage distributions, clinical AHI severity (from mild 3.9 to severe 22.0), and the 12-technologist scoring matrix. |
| [`09_pipeline_architecture.png`](09_pipeline_architecture.png) | **End-to-End Pipeline Architecture** | Step-by-step schematic from raw 256 Hz multi-channel signals, through quality gating, per-signal feature engines, union fusion, temporal context engineering (136 features), to dual-task prediction heads. |
| [`10_performance_comparison_progression.png`](10_performance_comparison_progression.png) | **Model Evolution Across Development Phases** | Tracks iterative benchmark progression: Phase 1 (Single signals: 6–45% F1) $\to$ Phase 2 (Static fusion: 54.7% F1) $\to$ Phase 5 (Temporal context: 57.5% F1, $\kappa = 0.502$). |

---

## 3. Exploratory Data Analysis & Per-Signal Diagnostics (`output_images/eda/`)

| Image File | Modality | Description |
|:---|:---:|:---|
| [`eda/ecg_eda_boxplots.png`](eda/ecg_eda_boxplots.png) | ECG | Per-stage distributions of heart rate, HRV time-domain metrics (RMSSD, SDNN), and frequency-domain sympathovagal power (LF/HF ratio). |
| [`eda/ecg_xgb_confusion.png`](eda/ecg_xgb_confusion.png) | ECG | Standalone 5-class confusion matrix for ECG-only baseline. |
| [`eda/ecg_xgb_importance.png`](eda/ecg_xgb_importance.png) | ECG | Feature importance ranking for isolated heart rate variability features. |
| [`eda/emg_eda_boxplots.png`](eda/emg_eda_boxplots.png) | EMG | Submental and anterior tibialis muscle tone distributions across stages (MAV, ZCR, high-frequency power, atonia index). |
| [`eda/emg_feature_importance.png`](eda/emg_feature_importance.png) | EMG | Feature importance ranking for submental atonia and motor burst features. |
| [`eda/emg_rf_confusion.png`](eda/emg_rf_confusion.png) | EMG | Standalone 5-class confusion matrix for EMG-only baseline. |
| [`eda/resp_eda_boxplots.png`](eda/resp_eda_boxplots.png) | Resp / $\text{SaO}_2$ | Distributions of nasal airflow amplitude, thoracoabdominal phase angle ($\phi_{\text{thoraco}}$), and oxygen desaturation slope across normal vs. apneic epochs. |
