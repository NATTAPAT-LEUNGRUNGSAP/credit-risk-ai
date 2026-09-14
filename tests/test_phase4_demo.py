from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app_utils.data import DEMO_DIR
from app_utils.domain import (
    PUBLIC_ID_COLUMNS,
    assign_risk_band,
    get_anonymized_applicant,
    missing_demo_artifacts,
    validated_probability,
)


def test_risk_band_assignment_includes_validated_boundaries() -> None:
    low_cut = 0.10
    medium_cut = 0.25

    assert assign_risk_band(0.00, low_cut, medium_cut) == "Low"
    assert assign_risk_band(low_cut, low_cut, medium_cut) == "Low"
    assert assign_risk_band(0.10001, low_cut, medium_cut) == "Medium"
    assert assign_risk_band(medium_cut, low_cut, medium_cut) == "Medium"
    assert assign_risk_band(0.25001, low_cut, medium_cut) == "High"


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_calibrated_probability_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="Calibrated probability"):
        validated_probability(value)


def test_anonymized_applicant_lookup_is_unique_and_validated() -> None:
    cases = pd.read_csv(DEMO_DIR / "anonymized_applicants.csv")
    alias = cases.loc[0, "case_alias"]

    applicant = get_anonymized_applicant(cases, alias)

    assert applicant["case_alias"] == alias
    assert 0 <= applicant["calibrated_probability"] <= 1
    assert not (PUBLIC_ID_COLUMNS & set(cases.columns))
    with pytest.raises(KeyError):
        get_anonymized_applicant(cases, "not-a-public-case")


def test_anonymized_lookup_blocks_private_identifier_columns() -> None:
    cases = pd.DataFrame(
        {
            "case_alias": ["L-01"],
            "calibrated_probability": [0.02],
            "SK_ID_CURR": [123456],
        }
    )

    with pytest.raises(ValueError, match="prohibited identifier"):
        get_anonymized_applicant(cases, "L-01")


def test_required_demo_bundle_is_complete_small_and_checksum_valid() -> None:
    manifest = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert missing_demo_artifacts(DEMO_DIR) == set()
    assert manifest["requires_raw_dataset"] is False
    assert manifest["requires_model_files"] is False
    assert manifest["contains_private_applicant_identifiers"] is False
    assert manifest["total_size_bytes"] < 1_000_000
    assert manifest["artifact_count"] == len(manifest["artifacts"])

    for artifact in manifest["artifacts"]:
        path = DEMO_DIR / artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_size == artifact["size_bytes"]
        assert digest == artifact["sha256"]


def test_demo_band_definitions_match_case_assignments() -> None:
    bands = pd.read_csv(DEMO_DIR / "risk_bands.csv")
    cases = pd.read_csv(DEMO_DIR / "anonymized_applicants.csv")
    low_cut = bands.loc[bands["risk_band"] == "Low", "probability_max"].iloc[0]
    medium_cut = bands.loc[bands["risk_band"] == "Medium", "probability_max"].iloc[0]

    assigned = cases["calibrated_probability"].map(
        lambda value: assign_risk_band(value, low_cut, medium_cut)
    )
    assert assigned.tolist() == cases["risk_band"].tolist()
