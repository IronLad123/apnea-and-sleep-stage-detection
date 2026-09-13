# Apnea and Sleep Stage Detection System

A clinically grounded, multi-modal machine learning pipeline for **5-class sleep staging** ($\text{Wake}, \text{N1}, \text{N2}, \text{N3}, \text{REM}$) and **sleep apnea detection** on the PhysioNet PSG-IPA database. Evaluated under strict **Leave-One-Subject-Out (LOSO)** cross-validation with zero patient data leakage.

---

## 📁 Repository Directory Structure

```
├── 07_Final_Benchmark.ipynb        # 🏆 Master reproducible benchmark notebook
├── psg_analysis.ipynb              # Signal 1: Central EEG exploration & baseline
├── 02_EOG_Sleep_Staging.ipynb      # Signal 2: EOG ocular dynamics & saccades
├── 03_EMG_Staging.ipynb            # Signal 3: Submental EMG atonia analysis
├── 04_ECG_Staging.ipynb            # Signal 4: ECG heart rate variability (HRV)
│
├── presentation.pptx               # 9-slide refined executive presentation deck
├── presentation.html               # Interactive browser-based presentation
│
├── brain.md                        # Project knowledge engine & clinical notes
├── output.md                       # Comprehensive numerical benchmarks & logs
│
├── features/                       # Processed Parquet caches (fused_union.parquet)
│   ├── eeg/                        # EEG spectral & complexity features
│   ├── eog/                        # EOG saccades & correlation features
│   ├── emg/                        # EMG submental & tibialis features
│   ├── ecg/                        # ECG HRV time & frequency features
│   ├── resp_sao2/                  # Airflow, effort, and SpO2 features
│   ├── fused_union.parquet         # 5,126 epochs × 89 cols (Union join)
│   └── labels.parquet              # Hard, soft (12-scorer), and apnea labels
│
├── results/                        # Evaluation metrics & visualizations
│   ├── final_dashboard.png         # 4-panel master benchmark dashboard
│   ├── confusion_matrix.png        # Normalized 5-class confusion matrix
│   ├── shap_importance.png         # Top-20 SHAP feature importance plot
│   ├── modality_ablation.png       # Single vs. multimodal ablation chart
│   ├── per_subject_benchmark.png   # Individual patient F1 profiles
│   ├── loso_stacked_results.csv    # Final Phase 5 temporal benchmark metrics
│   ├── loso_staging_results.csv    # Static LightGBM baseline metrics
│   └── apnea_v2_results.csv        # 12-scorer consensus apnea detection metrics
│
├── figures/                        # High-resolution vector-grade figures
│   ├── fig1_problem_scenario.png   # Clinical context & diagnostic workflow
│   ├── fig2_traditional_vs_proposed.png # Architecture comparison
│   ├── fig3_dataset_breakdown.png  # Class balance & patient profiles
│   ├── fig4_pipeline_architecture.png   # End-to-end ML data flow
│   ├── fig5_performance_comparison.png # Benchmark bar charts
│   └── eda/                        # Exploratory data analysis boxplots
│
├── scripts/                        # Production execution scripts
│   ├── run_temporal_lgbm.py        # Phase 5 temporal sliding-window model
│   ├── run_phase4_fixed.py         # Phase 4 interaction gates & apnea consensus
│   ├── generate_result_plots.py    # Script generating results/ figures
│   ├── fix_all_overlapping_figures.py # Script generating presentation figures
│   ├── build_refined_ppt.py        # Script building presentation.pptx
│   └── build_final_notebook.py     # Script generating 07_Final_Benchmark.ipynb
│
├── psg_utils/                      # Core shared Python modules (signal IO, metrics)
└── psg-ipa-*/                      # PhysioNet PSG-IPA v1.0.0 raw database
```

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
Activate the environment:
```bash
source psg_env/bin/activate
```

### 2. Run the Master Benchmark
To execute the final temporal machine learning model:
```bash
python scripts/run_temporal_lgbm.py
```

### 3. Generate Visualizations & Presentation Deck
To rebuild the figures and PowerPoint presentation:
```bash
python scripts/fix_all_overlapping_figures.py
python scripts/build_refined_ppt.py
```

### 4. Interactive Notebook
Launch Jupyter to explore `07_Final_Benchmark.ipynb`:
```bash
jupyter notebook 07_Final_Benchmark.ipynb
```

---

## 📊 Key Results Summary

* **Validation Strategy**: Strict Leave-One-Subject-Out (LOSO) across all 5 subjects ($5,126$ epochs).
* **Final Multimodal Model**: **$60.10\%$ Pooled Accuracy**, **$57.54\%$ Macro F1**, **Cohen's $\kappa = 0.5017$**.
* **Near-Human Consensus**: Patient **SN3** reached **$\kappa = 0.724$** ($74.0\%$ Macro F1), approaching human inter-rater agreement ($\kappa \ge 0.75$).
* **Deep Sleep (N3)**: Diagnosed with **$84.5\%$ F1** via delta power and spindle density.
* **Apnea Consensus Engine**: Reconstructed ground truth from **12 independent clinical scorers**, recovering $816$ true apnea epochs ($15.9\%$ prevalence).
