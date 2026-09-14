from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_app() -> AppTest:
    app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20)
    app.run()
    assert not app.exception
    return app


def test_landing_page_smoke() -> None:
    app = _load_app()

    assert app.title[0].value == "Overview"
    assert len(app.metric) >= 4
    copy = " ".join(item.value for item in app.markdown)
    assert "ROC-AUC improved from 0.7628 → 0.7866" in copy
    assert "Applicant data + historical credit behavior" in copy


def test_applicant_explorer_smoke_and_selection() -> None:
    app = _load_app().switch_page("app_pages/applicant.py").run()
    assert not app.exception

    aliases = app.selectbox(key="applicant_alias").options
    app.selectbox(key="applicant_alias").select(aliases[-1]).run()

    assert not app.exception
    assert app.selectbox(key="applicant_alias").value == aliases[-1]
    displayed_factors = app.dataframe[0].value
    assert displayed_factors.columns.tolist() == ["Feature", "Applicant value", "Effect"]
    assert "SHAP contribution" not in displayed_factors.columns


def test_portfolio_simulator_and_model_pages_smoke() -> None:
    app = _load_app().switch_page("app_pages/portfolio.py").run()
    assert not app.exception
    portfolio_copy = " ".join(item.value for item in app.markdown)
    assert "Problem → Data → Feature engineering → Models" in portfolio_copy
    band_table = next(
        frame.value
        for frame in app.dataframe
        if "applicant_count" in frame.value.columns
    )
    validated_bands = pd.read_csv(
        PROJECT_ROOT / "app" / "demo_artifacts" / "risk_bands.csv"
    )
    assert band_table["risk_band"].tolist() == ["Low", "Medium", "High"]
    assert band_table["applicant_count"].tolist() == [43_052, 12_300, 6_151]
    assert band_table["applicant_count"].tolist() == validated_bands[
        "applicant_count"
    ].tolist()

    app = app.switch_page("app_pages/simulator.py").run()
    assert not app.exception
    app.segmented_control(key="fn_fp_ratio").set_value(20).run()
    assert not app.exception
    assert app.segmented_control(key="fn_fp_ratio").value == 20
    simulator_copy = " ".join(item.value for item in app.markdown)
    assert "costs 20× more" in simulator_copy

    app = app.switch_page("app_pages/model.py").run()
    assert not app.exception
    assert any(item.value == "Selected model: XGBoost" for item in app.subheader)
    top_features = next(frame.value for frame in app.dataframe if len(frame.value) == 15)
    validated_importance = pd.read_csv(
        PROJECT_ROOT / "app" / "demo_artifacts" / "shap_global_importance.csv"
    )
    assert top_features["rank"].tolist() == list(range(1, 16))
    assert top_features["rank"].tolist() == validated_importance.head(15)["rank"].tolist()
