# Streamlit Community Cloud deployment

The public app is designed to run directly from the repository without raw Home Credit files or
trained model binaries. Its entry point is `streamlit_app.py`, and its only runtime inputs are the
small versioned files in `app/demo_artifacts/`.

## Pre-deployment checks

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run streamlit run streamlit_app.py
```

Confirm that `app/demo_artifacts/manifest.json` passes the test suite and that no secrets are
required by the application.

## Deploy

1. Push the reviewed repository to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, then choose **Deploy a public app from GitHub**.
4. Select the GitHub repository and release branch.
5. Set **Main file path** to `streamlit_app.py`.
6. In advanced settings, select Python 3.12. Leave Secrets empty.
7. Deploy and verify all five navigation pages.

Community Cloud detects the root `uv.lock` and installs the locked default project dependencies:
Streamlit, pandas, and Altair. Heavy training libraries are isolated in the optional `analysis`
extra and are not required by the app. No `packages.txt` or system package is required.

## Post-deployment validation

- Confirm Overview metrics and risk-band counts against the repository artifacts.
- Open each anonymized applicant example and verify no original identifier appears.
- Exercise all four FN:FP scenarios.
- Confirm the Top 15 SHAP table and technical plot load.
- Re-run the privacy checks documented in `reports/phase6_release_validation.md` before each public
  release.
