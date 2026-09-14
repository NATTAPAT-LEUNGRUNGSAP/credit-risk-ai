"""Export a minimal, anonymous, model-free artifact bundle for the public demo."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRIC_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORT_DIR

DEMO_DIR = PROJECT_ROOT / "app" / "demo_artifacts"
FIGURE_DIR = DEMO_DIR / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = read_json(METRIC_DIR / "baseline_metrics.json")
    phase2 = read_json(METRIC_DIR / "phase2_metrics.json")
    phase3 = read_json(METRIC_DIR / "phase3_metrics.json")
    audit = read_json(REPORT_DIR / "application_train_audit.json")

    summary = {
        "project_title": "Credit risk decision-support demo",
        "primary_model": "Phase 2 XGBoost",
        "roc_auc": phase2["metrics_at_threshold_0_5"]["XGBoost"]["roc_auc"],
        "pr_auc": phase2["metrics_at_threshold_0_5"]["XGBoost"]["pr_auc"],
        "dataset_default_rate": audit["target_rates"]["1"],
        "validation_rows": phase2["validation_rows"],
        "phase1_xgboost_roc_auc": phase1["metrics_at_threshold_0_5"]["XGBoost"][
            "roc_auc"
        ],
        "phase1_xgboost_pr_auc": phase1["metrics_at_threshold_0_5"]["XGBoost"]["pr_auc"],
        "calibration_method": phase3["selected_calibration"],
        "is_educational_demo": True,
        "is_automated_approval_system": False,
    }
    (DEMO_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_rows = []
    for name, metrics in phase2["metrics_at_threshold_0_5"].items():
        model_rows.append(
            {
                "Model": name,
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "F1": metrics["f1"],
            }
        )
    pd.DataFrame(model_rows).to_csv(DEMO_DIR / "model_comparison.csv", index=False)

    confusion = phase2["metrics_at_threshold_0_5"]["XGBoost"]["confusion_matrix"]
    pd.DataFrame(
        [
            {"Actual": "No default", "Predicted": "No default", "Count": confusion["tn"]},
            {"Actual": "No default", "Predicted": "Default", "Count": confusion["fp"]},
            {"Actual": "Default", "Predicted": "No default", "Count": confusion["fn"]},
            {"Actual": "Default", "Predicted": "Default", "Count": confusion["tp"]},
        ]
    ).to_csv(DEMO_DIR / "xgboost_confusion_matrix.csv", index=False)

    cases = pd.read_csv(REPORT_DIR / "phase3_case_studies.csv")
    if "SK_ID_CURR" in cases.columns:
        raise ValueError("Applicant identifiers must not enter the demo bundle")
    cases.to_csv(DEMO_DIR / "anonymized_applicants.csv", index=False)

    bands = pd.read_csv(REPORT_DIR / "phase3_risk_bands.csv")
    bands.to_csv(DEMO_DIR / "risk_bands.csv", index=False)
    costs = pd.read_csv(REPORT_DIR / "phase3_cost_sensitivity.csv")
    costs.to_csv(DEMO_DIR / "cost_sensitivity.csv", index=False)
    thresholds = pd.read_csv(REPORT_DIR / "phase3_threshold_metrics.csv")
    thresholds.to_csv(DEMO_DIR / "threshold_metrics.csv", index=False)
    importance = pd.read_csv(REPORT_DIR / "phase3_shap_global_importance.csv").head(25)
    importance.to_csv(DEMO_DIR / "shap_global_importance.csv", index=False)
    calibration = pd.read_csv(REPORT_DIR / "phase3_calibration_comparison.csv")
    calibration.to_csv(DEMO_DIR / "calibration_comparison.csv", index=False)

    predictions = pd.read_csv(
        PROCESSED_DIR / "phase3_validation_decision_probabilities.csv",
        usecols=["TARGET", "calibrated_probability"],
    )
    edges = np.linspace(0, 1, 41)
    predictions["bin"] = pd.cut(
        predictions["calibrated_probability"], bins=edges, include_lowest=True
    )
    histogram = predictions.groupby("bin", observed=False).agg(
        Applicant_count=("TARGET", "size"),
        Observed_default_rate=("TARGET", "mean"),
        Mean_probability=("calibrated_probability", "mean"),
    )
    histogram["Probability_lower"] = [interval.left for interval in histogram.index]
    histogram["Probability_upper"] = [interval.right for interval in histogram.index]
    histogram = histogram.reset_index(drop=True).fillna(0)
    histogram.to_csv(DEMO_DIR / "probability_histogram.csv", index=False)

    source_figures = {
        "shap_beeswarm.png": REPORT_DIR / "figures" / "phase3_shap_beeswarm.png",
        "calibration_curve.png": REPORT_DIR / "figures" / "phase3_calibration_curve.png",
    }
    for name, source in source_figures.items():
        shutil.copy2(source, FIGURE_DIR / name)

    artifact_paths = sorted(
        path for path in DEMO_DIR.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "purpose": "Minimal anonymous precomputed bundle for the public Streamlit demo",
        "requires_raw_dataset": False,
        "requires_model_files": False,
        "contains_private_applicant_identifiers": False,
        "artifact_count": len(artifact_paths),
        "total_size_bytes": sum(path.stat().st_size for path in artifact_paths),
        "artifacts": [
            {
                "path": path.relative_to(DEMO_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
