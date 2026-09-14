import json

import numpy as np
import pandas as pd

from src.config import METRIC_DIR, PROCESSED_DIR, REPORT_DIR


def test_phase3_preserved_prior_artifacts_and_shap_is_additive() -> None:
    metrics = json.loads((METRIC_DIR / "phase3_metrics.json").read_text(encoding="utf-8"))

    assert metrics["preservation_checks"]["all_preserved"] is True
    assert metrics["shap_max_additivity_error_raw_margin"] < 1e-4
    assert metrics["primary_model"] == "Phase 2 XGBoost"


def test_calibration_is_training_only_and_selected_by_brier_score() -> None:
    metrics = json.loads((METRIC_DIR / "phase3_metrics.json").read_text(encoding="utf-8"))
    calibration = pd.DataFrame(metrics["calibration_results"])
    selected = calibration.loc[calibration["brier_score"].idxmin(), "method"]

    assert "training IDs only" in metrics["calibration_fit_source"]
    assert metrics["selected_calibration"] == selected
    assert calibration.loc[
        calibration["method"] == "Platt scaling", "brier_score"
    ].iloc[0] < calibration.loc[
        calibration["method"] == "Uncalibrated", "brier_score"
    ].iloc[0]


def test_threshold_table_matches_saved_decision_probabilities() -> None:
    predictions = pd.read_csv(PROCESSED_DIR / "phase3_validation_decision_probabilities.csv")
    thresholds = pd.read_csv(REPORT_DIR / "phase3_threshold_metrics.csv")
    row = thresholds[np.isclose(thresholds["threshold"], 0.5)].iloc[0]
    predicted = predictions["calibrated_probability"] >= 0.5
    actual = predictions["TARGET"].astype(bool)

    assert int((predicted & actual).sum()) == row["tp"]
    assert int((predicted & ~actual).sum()) == row["fp"]
    assert int((~predicted & actual).sum()) == row["fn"]
    assert int((~predicted & ~actual).sum()) == row["tn"]


def test_higher_fn_cost_selects_lower_threshold() -> None:
    costs = pd.read_csv(REPORT_DIR / "phase3_cost_sensitivity.csv")

    assert costs["assumed_fn_cost"].is_monotonic_increasing
    assert costs["threshold"].is_monotonic_decreasing
    assert (costs["assumed_fp_cost"] == 1).all()


def test_risk_bands_are_complete_and_monotonic() -> None:
    bands = pd.read_csv(REPORT_DIR / "phase3_risk_bands.csv")

    assert bands["applicant_count"].sum() == 61_503
    assert bands["observed_default_rate"].is_monotonic_increasing
    assert bands["risk_band"].tolist() == ["Low", "Medium", "High"]


def test_public_case_studies_are_anonymized() -> None:
    cases = pd.read_csv(REPORT_DIR / "phase3_case_studies.csv")

    assert "SK_ID_CURR" not in cases.columns
    assert len(cases) == 6
    assert set(cases["risk_band"]) == {"Low", "Medium", "High"}
    assert cases["case_alias"].is_unique

