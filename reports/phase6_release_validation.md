# Phase 6 — Release validation

## Release status

The repository is technically ready for an initial GitHub commit and Streamlit Community Cloud
deployment. No push or deployment was performed. The Git repository is on `main`, has no commits,
and has no remote configured.

One presentation item remains: the browser sandbox allowed visual screenshot capture and review
but did not provide a supported way to persist those captures into the workspace. README image
assets should therefore be captured from the final public URL immediately after the first
deployment. No broken image links are included.

## Phase 6 changes

- Corrected the Portfolio `Applicants` display format to Streamlit's documented `%,d` formatter.
- Added artifact-backed tests for Low 43,052, Medium 12,300, and High 6,151 applicants.
- Added a test proving the default Model Insights table is exactly ranks 1–15 from the preserved
  SHAP importance artifact.
- Reworked the README opening, verified results, architecture, methodology/leakage prevention,
  privacy boundary, limitations, repository structure, and run/deployment instructions.
- Added `DEPLOYMENT.md` for Streamlit Community Cloud.
- Reduced the locked default runtime to Streamlit, pandas, and Altair. Training libraries remain
  available through the optional `analysis` extra.
- Added release-candidate privacy, credential, path, binary, size, dependency, and README tests.

## Automated and runtime validation

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe -q
```

- Ruff: passed
- Pytest: **36 passed in 7.10 seconds**
- Streamlit health: HTTP 200, `ok`
- Browser and AppTest smoke tests: all five pages passed
- Fresh locked Python 3.12 environment: 38 packages installed; Streamlit 1.63.0, pandas 2.3.3,
  Altair 6.2.2; all five pages passed
- Release candidates: 89 files, 2,609,681 bytes
- Largest candidate: `reports/figures/phase3_shap_beeswarm.png`, 352,818 bytes

## Privacy and integrity validation

- Public demo identifier scan: clean
- Public demo model-binary scan: clean
- Release text credential/private-key/local-user-path scan: clean
- Raw data, processed data, models, `.env`, and Streamlit secrets: Git-ignored
- No release-candidate file exceeds 5 MB
- Demo manifest checksum validation: passed

Preserved artifact SHA-256 values:

- `baseline_metrics.json`: `6c850975348faebca9918bc3fa2eb2507900edb76bfb6b6766beffcd86d4ceb3`
- `phase2_metrics.json`: `d0a26f2e4318596b9fa9dc4a00ec261a6fb34d98fcb0e74f4e80a80eb83cbfc8`
- `phase3_metrics.json`: `9ea6f4c231c0dc657758409154b4ec36b1e9c0608739b83043b70c4e05e717d9`
- public demo manifest: `e4afe1c8ef0fbd78e8c9d3a80864454adb35c9f7d8e0833d4595b3da4fd94d38`
- Phase 4 validation report: `21a5513d63b6aeb17594bcc4136e2f5f808c7ffc41f28641513f8a42e4b2af71`
- Phase 5 validation report: `b0b1a0a5e83890f5091a4860bc73db754aed82b17d0ec69429c5886b7abf4550`
- risk-band artifact: `a0350668348cab9742bd0ca57609860a10164c8144329645390b7f4a8b3010aa`
- SHAP importance artifact: `d9b3ac11c2cd21454c1a70562f013d939e4b9bb21f6912649e61ab84bc33e4ad`
