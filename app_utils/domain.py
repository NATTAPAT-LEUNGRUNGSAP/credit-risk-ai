from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC_ID_COLUMNS = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}
REQUIRED_DEMO_ARTIFACTS = {
    "summary.json",
    "model_comparison.csv",
    "xgboost_confusion_matrix.csv",
    "anonymized_applicants.csv",
    "risk_bands.csv",
    "cost_sensitivity.csv",
    "threshold_metrics.csv",
    "probability_histogram.csv",
    "shap_global_importance.csv",
    "calibration_comparison.csv",
    "figures/shap_beeswarm.png",
    "figures/calibration_curve.png",
    "manifest.json",
}


def validated_probability(value: float) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Calibrated probability must be finite and between 0 and 1")
    return probability


def assign_risk_band(probability: float, low_cut: float, medium_cut: float) -> str:
    probability = validated_probability(probability)
    if not 0 <= low_cut < medium_cut <= 1:
        raise ValueError("Risk-band cut points must be ordered inside [0, 1]")
    if probability <= low_cut:
        return "Low"
    if probability <= medium_cut:
        return "Medium"
    return "High"


def get_anonymized_applicant(cases: pd.DataFrame, alias: str) -> pd.Series:
    if PUBLIC_ID_COLUMNS & set(cases.columns):
        raise ValueError("Public applicant data contains a prohibited identifier")
    matches = cases.loc[cases["case_alias"] == alias]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicated anonymized applicant alias: {alias}")
    row = matches.iloc[0]
    validated_probability(row["calibrated_probability"])
    return row


def missing_demo_artifacts(demo_dir: Path) -> set[str]:
    return {
        relative_path
        for relative_path in REQUIRED_DEMO_ARTIFACTS
        if not (demo_dir / relative_path).is_file()
    }

