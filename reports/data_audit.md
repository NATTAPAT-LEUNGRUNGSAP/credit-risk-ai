# Phase 1 Data Audit

## CSV inventory

| file | rows | columns | size_mb | encoding |
| --- | --- | --- | --- | --- |
| application_test.csv | 48744 | 121 | 25.34 | utf-8 |
| application_train.csv | 307511 | 122 | 158.44 | utf-8 |
| bureau.csv | 1716428 | 17 | 162.14 | utf-8 |
| bureau_balance.csv | 27299925 | 3 | 358.19 | utf-8 |
| credit_card_balance.csv | 3840312 | 23 | 404.91 | utf-8 |
| HomeCredit_columns_description.csv | 219 | 5 | 0.04 | cp1252 |
| installments_payments.csv | 13605401 | 8 | 689.62 | utf-8 |
| POS_CASH_balance.csv | 10001358 | 8 | 374.51 | utf-8 |
| previous_application.csv | 1670214 | 37 | 386.21 | utf-8 |
| sample_submission.csv | 48744 | 2 | 0.51 | utf-8 |

## application_train.csv

- Shape: 307,511 rows x 122 columns
- Repaid (TARGET=0): 282,686 (91.93%)
- Default/payment difficulty (TARGET=1): 24,825 (8.07%)
- Numerical features (ID included, target excluded): 105
- Categorical features: 16
- Columns with missing values: 67
- Columns over 50% missing: 41
- Duplicate SK_ID_CURR: 0
- Fully duplicate rows: 0
- Infinite numeric values: 0
- Constant columns: []

Detailed per-column schema and missingness are in `csv_schema.csv` and `application_train_missing_values.csv`.
