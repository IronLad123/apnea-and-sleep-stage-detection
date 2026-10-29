# Monochromatic Single-Graph Figures (Publication Ready)

This directory contains standalone, **single-graph monochromatic (grayscale) figures** rendered at 300 DPI with black, white, and hatched patterns. Every image contains exactly **one plot** (no multi-panel subplots), making them drop-in ready for academic journals (IEEE, Springer, Elsevier) and black-and-white print submissions.

---

## Catalog of Standalone Monochromatic Figures

| File Name | Figure Title | Academic Caption / Description |
|:---|:---|:---|
| [`01_confusion_matrix_monochrome.png`](01_confusion_matrix_monochrome.png) | **Pooled Confusion Matrix** | Normalized 5-class confusion matrix across all 5,126 epochs evaluated under Leave-One-Subject-Out (LOSO) cross-validation. Values represent row-normalized proportions. |
| [`02_per_stage_f1_by_subject_monochrome.png`](02_per_stage_f1_by_subject_monochrome.png) | **Per-Stage F1 by Subject** | Stage-wise classification performance ($F_1$-score) across held-out subjects SN1 through SN5, with hatched patterns differentiating Wake, N1, N2, N3, and REM. |
| [`03_overall_metrics_by_subject_monochrome.png`](03_overall_metrics_by_subject_monochrome.png) | **Overall Staging Metrics** | Independent bar comparison of Accuracy, Macro $F_1$, and Cohen's $\kappa \times 100$ per subject. Illustrates Subject SN3 reaching near-human concordance ($\kappa = 0.724$). |
| [`04_shap_feature_importance_monochrome.png`](04_shap_feature_importance_monochrome.png) | **Global SHAP Feature Attribution** | Horizontal bar chart ranking the top-20 features by mean absolute SHAP value across all five folds, highlighting EOG zero-crossing rate and EEG Hjorth mobility. |
| [`05_modality_ablation_monochrome.png`](05_modality_ablation_monochrome.png) | **Modality Ablation Comparison** | Standalone bar chart illustrating the diagnostic contribution of each sensor modality, from Respiration-only (6.0% $F_1$) to the full temporal multimodal ensemble (57.5% $F_1$). |
| [`06_apnea_auc_by_subject_monochrome.png`](06_apnea_auc_by_subject_monochrome.png) | **Apnea Detection AUC-ROC** | Area Under the ROC Curve for sleep apnea detection across held-out subjects evaluated against 12-expert majority consensus ($\ge 4/12$ agreement). |
| [`07_apnea_f1_optimal_monochrome.png`](07_apnea_f1_optimal_monochrome.png) | **Apnea Optimal F1-Score** | Optimal $F_1$-score and corresponding operating threshold ($t^*$) per subject, showing robust detection on moderate-to-severe disease (SN3 $F_1 = 0.425$ at $t = 0.05$). |
| [`08_pipeline_progression_monochrome.png`](08_pipeline_progression_monochrome.png) | **Architecture Progression** | Iterative benchmark advancement from unimodal baselines through static union fusion to temporal lag/lead feature engineering ($\kappa = 0.5017$). |
| [`09_class_distribution_monochrome.png`](09_class_distribution_monochrome.png) | **Cohort Sleep Stage Distribution** | Bar chart detailing epoch counts and class proportions across the 5,126 standardized 30-second epochs. |

---

## Technical Specifications
* **Color Space:** Grayscale (Black `#000000`, Dark Gray `#404040`, Mid Gray `#737373`, Light Gray `#A3A3A3`, White `#FFFFFF`).
* **Patterns:** Standard academic hatchings (`///`, `\\\\\\`, `xxx`, `...`, `---`) for clean distinction without color reliance.
* **Resolution:** 300 DPI (High-definition print standard).
* **Format:** PNG with tight bounding boxes (`bbox_inches='tight'`).
