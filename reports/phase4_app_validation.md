# Phase 4 — Streamlit Portfolio Application Validation

## Outcome

The public portfolio application uses only anonymized, precomputed validation outputs from
Phases 2–3. It does not load the Home Credit CSV files or trained model binaries at runtime and
does not retrain any model.

## Public demo bundle

- 12 payload artifacts, 453,917 bytes (excluding the manifest itself)
- six anonymized validation case studies
- aggregated probability histogram, risk-band outcomes, model metrics, threshold/cost results,
  SHAP importance and two existing Phase 3 figures
- no `SK_ID_CURR`, `SK_ID_PREV`, or `SK_ID_BUREAU` columns
- no raw dataset or model-file dependency
- SHA-256 checksums recorded in `app/demo_artifacts/manifest.json`

## Automated verification

Command:

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -q
```

Result on 2026-09-14: Ruff passed and **26 tests passed in 7.04 seconds**. The test suite covers
the prior phase contracts plus risk-band boundaries, invalid calibrated probabilities,
anonymized applicant lookup, identifier blocking, required artifact presence and checksums,
bundle size, all five Streamlit pages, applicant selection, and scenario control interaction.

## Local server smoke test

Launch command:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501 --server.headless true
```

Local URL: `http://localhost:8501`

- `GET /_stcore/health`: HTTP 200, body `ok`
- `GET /`: HTTP 200, 7,459-byte response

## Preservation evidence

No Phase 1–3 model or metrics file was modified. SHA-256 values after Phase 4 validation:

- `baseline_metrics.json`: `6c850975348faebca9918bc3fa2eb2507900edb76bfb6b6766beffcd86d4ceb3`
- `phase2_metrics.json`: `d0a26f2e4318596b9fa9dc4a00ec261a6fb34d98fcb0e74f4e80a80eb83cbfc8`
- `phase3_metrics.json`: `9ea6f4c231c0dc657758409154b4ec36b1e9c0608739b83043b70c4e05e717d9`
