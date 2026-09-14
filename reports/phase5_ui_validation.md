# Phase 5 — Recruiter-facing UI validation

## Scope

Phase 5 changes presentation code and UI tests only. It does not retrain models or change saved
predictions, validation splits, calibration, thresholds, SHAP calculations, risk bands, or metrics.

## UI changes

- Overview: leads with the measured Phase 1 → Phase 2 XGBoost improvement, explains the source of
  the gain, and adds a plain-language system flow while retaining model and risk-band metrics.
- Applicant explorer: translates technical feature names and raw values into business-readable
  labels, ages, percentages, counts, and comma-formatted monetary units. The main tables show
  feature, applicant value, and model effect; original values and exact SHAP contributions are in
  a technical-details expander.
- Model insights: adds a selected-model summary, defaults to the 15 most important SHAP features,
  moves the full beeswarm into an expander, and keeps training-only calibration evidence visible.
- Business simulator: explains the selected FN:FP assumption in plain language, defines both error
  types, adds metric help text, and presents the threshold–recall–flagging relationship directly.
- Portfolio: introduces a six-stage case-study narrative, headline dataset facts, the measured
  ROC-AUC/PR-AUC improvement, calibration, SHAP, and cost-sensitive decision support. Dense
  validation views remain available in an expander.

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -q
```

Result on 2026-09-14: Ruff passed and **32 tests passed in 7.03 seconds**. AppTest exercised all
five pages, applicant selection, and the 20:1 simulator scenario. Presentation unit tests cover
age, money, ratio, score, missing-value, label, and SHAP-string translation.

The app was launched with:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501 --server.headless true
```

Local URL: `http://localhost:8501`

- `GET /_stcore/health`: HTTP 200, body `ok`
- `GET /`: HTTP 200, 7,459-byte response
- public-bundle identifier scan: clean
- public-bundle model-binary scan: clean
- Git-tracked sensitive-file scan: clean

## Preservation hashes

- `baseline_metrics.json`: `6c850975348faebca9918bc3fa2eb2507900edb76bfb6b6766beffcd86d4ceb3`
- `phase2_metrics.json`: `d0a26f2e4318596b9fa9dc4a00ec261a6fb34d98fcb0e74f4e80a80eb83cbfc8`
- `phase3_metrics.json`: `9ea6f4c231c0dc657758409154b4ec36b1e9c0608739b83043b70c4e05e717d9`
- Phase 4 demo manifest: `e4afe1c8ef0fbd78e8c9d3a80864454adb35c9f7d8e0833d4595b3da4fd94d38`
- Phase 4 validation report: `21a5513d63b6aeb17594bcc4136e2f5f808c7ffc41f28641513f8a42e4b2af71`
