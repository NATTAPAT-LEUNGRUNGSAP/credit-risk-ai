# Phase 1 Baseline Results

All models use the same stratified holdout: 246,008 training rows and 61,503 validation rows
(`random_state=42`). Preprocessing was fitted on training rows only. The applicant ID was excluded
from model inputs. The processed matrix contains 241 features.

## Validation metrics at threshold 0.50

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.748675 | 0.227935 | 0.678550 | 0.161730 | 0.261203 |
| XGBoost | 0.762806 | 0.253218 | 0.658006 | 0.176242 | 0.278019 |
| ANN | 0.749500 | 0.227530 | 0.717221 | 0.154981 | 0.254885 |

## Confusion matrices

| Model | True negative | False positive | False negative | True positive |
|---|---:|---:|---:|---:|
| Logistic Regression | 39,076 | 17,462 | 1,596 | 3,369 |
| XGBoost | 41,268 | 15,270 | 1,698 | 3,267 |
| ANN | 37,122 | 19,416 | 1,404 | 3,561 |

XGBoost provides the best ranking metrics and F1. ANN catches the most positive cases at the fixed
threshold, with a substantial false-positive cost. Threshold selection is deliberately deferred
until the project defines business costs and capacity constraints.

