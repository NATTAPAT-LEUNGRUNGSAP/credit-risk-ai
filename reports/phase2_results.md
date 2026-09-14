# Phase 2 Relational Feature Engineering Results

Phase 2 aggregates six historical tables into one row per `SK_ID_CURR`, then joins those features
to `application_train.csv`. It reuses the exact Phase 1 train and validation IDs.

## Feature set

| Feature family | Engineered features |
|---|---:|
| Application-only raw features | 120 |
| Bureau + bureau balance | 35 |
| Previous applications | 25 |
| Installment payments | 17 |
| POS cash balance | 14 |
| Credit card balance | 25 |
| **Enriched raw total** | **236** |
| Phase 1 transformed features | 241 |
| **Phase 2 transformed features** | **357** |

The relational features include history indicators, record/contract counts, active/closed ratios,
delinquency frequency/severity, outstanding debt and credit amounts, approval/refusal behavior,
installment lateness and underpayment, POS status/DPD, and credit-card utilization/payment behavior.

## Model results on the unchanged validation set

Metrics are measured on 61,503 validation applicants at threshold 0.50.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.773874 | 0.260109 | 0.697482 | 0.174011 | 0.278533 |
| XGBoost | **0.786647** | **0.288217** | 0.666264 | **0.199493** | **0.307050** |
| ANN | 0.775128 | 0.268835 | **0.724673** | 0.165852 | 0.269928 |

## Improvement over Phase 1

| Model | ROC-AUC delta | PR-AUC delta | Recall delta | Precision delta | F1 delta |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | +0.025199 | +0.032173 | +0.018933 | +0.012281 | +0.017330 |
| XGBoost | +0.023841 | +0.035000 | +0.008258 | +0.023251 | +0.029031 |
| ANN | +0.025628 | +0.041305 | +0.007452 | +0.010871 | +0.015042 |

## XGBoost source ablation

Each single-source row uses all application features plus the named source. The all-source row is
the final Phase 2 XGBoost model.

| Configuration | Features | ROC-AUC | Delta | PR-AUC | Delta |
|---|---:|---:|---:|---:|---:|
| Application only (Phase 1) | 241 | 0.762806 | — | 0.253218 | — |
| + Bureau and bureau balance | 276 | 0.767510 | +0.004704 | 0.265184 | +0.011966 |
| + Previous applications | 266 | 0.771845 | +0.009039 | 0.263141 | +0.009923 |
| + Installment payments | 258 | 0.769149 | +0.006343 | 0.263786 | +0.010568 |
| + POS cash balance | 255 | 0.772913 | +0.010107 | 0.267804 | +0.014586 |
| + Credit card balance | 266 | 0.767678 | +0.004873 | 0.259770 | +0.006552 |
| **+ All relational sources** | **357** | **0.786647** | **+0.023841** | **0.288217** | **+0.035000** |

POS history has the largest standalone lift for both ranking metrics in this ablation. The combined
model gains substantially more than any single source, indicating complementary signal across the
historical tables. These results describe predictive contribution, not causal effects.

## Data-quality and leakage checks

- Aggregated relational table: 307,511 rows and 307,511 unique applicant IDs.
- No duplicated aggregate IDs and no `TARGET` in relational features.
- One-to-one join preserved all 307,511 application rows.
- Train/validation intersection is empty.
- Split union exactly matches application IDs and manifest target labels match source labels.
- Validation ID order exactly matches Phase 1.
- No non-finite relational values remain after cleaning.

Machine-readable details are in `phase2_feature_quality.json`, `phase2_leakage_checks.json`,
`phase2_ablation.csv`, and `metrics/phase2_metrics.json`.

