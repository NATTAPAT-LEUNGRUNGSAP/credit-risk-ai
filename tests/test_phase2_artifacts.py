import json
from pathlib import Path

import pandas as pd

from src.config import ID_COLUMN, METRIC_DIR, PROCESSED_DIR, TARGET


def test_relational_features_have_one_row_per_application_and_no_target() -> None:
    features = pd.read_parquet(PROCESSED_DIR / "phase2_relational_features.parquet")

    assert len(features) == 307_511
    assert features[ID_COLUMN].is_unique
    assert TARGET not in features.columns
    assert features.shape[1] - 1 == 116


def test_behavioral_feature_families_are_present() -> None:
    columns = set(pd.read_parquet(PROCESSED_DIR / "phase2_relational_features.parquet").columns)
    expected = {
        "BUREAU_ACTIVE_RATIO",
        "BUREAU_BALANCE_DELINQUENT_RATIO",
        "BUREAU_DEBT_TO_CREDIT_RATIO",
        "PREV_APPROVED_RATIO",
        "PREV_REFUSED_RATIO",
        "INST_LATE_RATIO",
        "INST_UNDERPAY_RATIO",
        "POS_DELINQUENT_RATIO",
        "POS_COMPLETED_RATIO",
        "CC_UTILIZATION_MEAN",
        "CC_PAYMENT_TO_MIN_RATIO",
        "CC_DELINQUENT_RATIO",
    }

    assert expected <= columns


def test_phase2_predictions_preserve_phase1_validation_ids() -> None:
    manifest = pd.read_csv(PROCESSED_DIR / "validation_split_ids.csv")
    predictions = pd.read_csv(PROCESSED_DIR / "phase2_validation_predictions.csv")

    assert predictions[ID_COLUMN].tolist() == manifest[ID_COLUMN].tolist()
    assert predictions[TARGET].tolist() == manifest[TARGET].tolist()
    assert predictions[ID_COLUMN].is_unique


def test_phase1_and_phase2_metrics_are_separate() -> None:
    phase1_path = METRIC_DIR / "baseline_metrics.json"
    phase2_path = METRIC_DIR / "phase2_metrics.json"
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    phase2 = json.loads(phase2_path.read_text(encoding="utf-8"))

    assert phase1_path != phase2_path
    assert phase1["transformed_features"] == 241
    assert phase2["phase2_transformed_feature_count"] == 357
    assert phase2["split_source"].startswith("Phase 1")


def test_all_source_aggregate_files_have_unique_applicant_ids() -> None:
    paths = sorted(Path(PROCESSED_DIR).glob("phase2_*_features.parquet"))
    assert len(paths) == 6
    for path in paths:
        frame = pd.read_parquet(path, columns=[ID_COLUMN])
        assert frame[ID_COLUMN].is_unique, path.name

