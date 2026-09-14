# Phase 3 Explainability and Business Decision Analysis

The primary model is the saved Phase 2 XGBoost model. ANN remains the recall-oriented comparison model. Phase 1 and Phase 2 artifacts are unchanged.

## Actual model results carried forward

- XGBoost ROC-AUC: 0.786647
- XGBoost PR-AUC: 0.288217
- XGBoost recall at the original 0.50 threshold: 0.666264
- ANN recall at 0.50: 0.724673

## Global SHAP drivers

SHAP values explain changes in the XGBoost raw margin (log-odds). Positive SHAP values push predicted default risk upward; negative values push it downward.

| Rank | Feature | Mean absolute SHAP | Directional pattern |
|---:|---|---:|---|
| 1 | External source 2 | 0.33954 | higher values tend to decrease risk |
| 2 | External source 3 | 0.29669 | higher values tend to decrease risk |
| 3 | External source 1 | 0.13797 | higher values tend to decrease risk |
| 4 | Amount Credit | 0.13755 | higher values tend to increase risk |
| 5 | Amount Goods Price | 0.13480 | higher values tend to decrease risk |
| 6 | Amount Annuity | 0.11015 | higher values tend to increase risk |
| 7 | Installment Late Ratio | 0.09844 | higher values tend to increase risk |
| 8 | Days Birth | 0.09389 | higher values tend to increase risk |
| 9 | Installment Payment Amount Sum | 0.09281 | higher values tend to decrease risk |
| 10 | Days Employed | 0.09053 | higher values tend to increase risk |

Directional labels summarize association in the SHAP sample and are not causal claims. Mixed effects should be read from the beeswarm plot.

## Probability calibration

Calibration mappings were fitted from three-fold out-of-fold predictions using only the Phase 1 training IDs. Validation labels were not used to fit either mapping.

| Method | Brier score | Log loss | 10-bin ECE | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Platt scaling | 0.065657 | 0.236799 | 0.004197 | 0.786647 | 0.288217 |
| Isotonic regression | 0.065669 | 0.236810 | 0.004575 | 0.786567 | 0.281391 |
| Uncalibrated | 0.166011 | 0.502982 | 0.282561 | 0.786647 | 0.288217 |

Selected for decision analysis: **Platt scaling**, based on the lowest validation Brier score. This is post-hoc calibration, not a new ranking model.

## Threshold analysis

Decision analysis uses the selected calibrated probabilities. At threshold 0.50: TP=217, FP=134, TN=56,404, FN=4,748, recall=0.0437, precision=0.6182, specificity=0.9976, F1=0.0816.

The complete threshold grid is in `phase3_threshold_metrics.csv`.

## Illustrative business cost sensitivity

**These are hypothetical cost units, not Home Credit or bank economics.** False-positive cost is fixed to one unit and false-negative cost is varied.

| Assumed FN:FP cost | Cost-minimizing threshold | TP | FP | TN | FN | Recall | Precision | Cost units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2:1 | 0.35689 | 693 | 800 | 55,738 | 4,272 | 0.1396 | 0.4642 | 9,344 |
| 5:1 | 0.18443 | 2,066 | 5,072 | 51,466 | 2,899 | 0.4161 | 0.2894 | 19,567 |
| 10:1 | 0.09933 | 3,353 | 13,615 | 42,923 | 1,612 | 0.6753 | 0.1976 | 29,735 |
| 20:1 | 0.05459 | 4,187 | 25,434 | 31,104 | 778 | 0.8433 | 0.1414 | 40,994 |

There is no universal optimal threshold: it changes with the assumed relative cost of missing a default versus reviewing or declining a non-default applicant.

## Validation-derived risk bands

Search all validation decile boundaries with at least one decile per band; enforce monotonic observed default rates; maximize weighted between-band rate separation.

| Risk band | Applicants | Share | Observed defaults | Observed default rate | Probability range |
|---|---:|---:|---:|---:|---|
| Low | 43,052 | 70.00% | 1,514 | 3.52% | 0.0005–0.0920 |
| Medium | 12,300 | 20.00% | 1,564 | 12.72% | 0.0921–0.2008 |
| High | 6,151 | 10.00% | 1,887 | 30.68% | 0.2008–0.8180 |

Limitation: Descriptive validation-derived bands; confirm on a future holdout before use.

## Anonymized applicant case studies

The public case table contains aliases only. Positive SHAP values are risk drivers and negative values are protective drivers.

| Case | Band | Actual | Raw score | Calibrated probability | Main risk drivers | Main protective drivers |
|---|---|---:|---:|---:|---|---|
| L-01 | Low | 0 | 0.2328 | 0.0346 | Installment Payment Amount Sum [5e+04] (+0.175); Days Birth [-11798] (+0.122); Amount Goods Price [2.7e+05] (+0.121) | External source 2 [0.764] (-0.982); Amount Credit [2.7e+05] (-0.191); Installment Late Ratio [0] (-0.120) |
| L-02 | Low | 1 | 0.2328 | 0.0346 | External source 3 [0.401] (+0.147); External source 1 [missing] (+0.116); Amount Goods Price [2.7e+05] (+0.104) | External source 2 [0.695] (-0.527); Days Birth [-8880] (-0.235); Amount Credit [2.7e+05] (-0.153) |
| M-01 | Medium | 0 | 0.5705 | 0.1292 | External source 2 [0.163] (+0.481); Installment Late Ratio [0.342] (+0.196); Code Gender = F [absent] (+0.181) | External source 3 [0.75] (-0.391); Amount Annuity [1.53e+04] (-0.186); Previous Days Decision Max [-967] (-0.130) |
| M-02 | Medium | 1 | 0.5705 | 0.1292 | Credit card Balance Latest [1.42e+05] (+0.222); Credit card Drawing Count Sum [151] (+0.166); Credit card Utilization Max [1.05] (+0.143) | Amount Annuity [1.1e+04] (-0.186); Name Education Type = Higher Education [present] (-0.100); Bureau Days Credit Min [-2.91e+03] (-0.099) |
| H-01 | High | 0 | 0.7796 | 0.2760 | External source 3 [0.296] (+0.258); External source 1 [0.209] (+0.234); Bureau Debt To Credit Ratio [0.916] (+0.161) | POS Future Instalments Mean [5.5] (-0.095); Previous Credit To Application Mean [1.02] (-0.079); Amount Credit [3.41e+05] (-0.058) |
| H-02 | High | 1 | 0.7796 | 0.2760 | External source 2 [0.0605] (+0.949); External source 3 [0.264] (+0.323); Amount Credit [9.33e+05] (+0.148) | Amount Goods Price [7.78e+05] (-0.223); POS Record Count [89] (-0.148); Installment Payment Amount Sum [1.21e+06] (-0.119) |

Applicant IDs are stored only in a Git-ignored internal artifact and are not exposed in portfolio reports.
