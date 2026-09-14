from __future__ import annotations

import pytest

from app_utils.presentation import format_applicant_value, parse_driver_list


@pytest.mark.parametrize(
    ("feature", "raw_value", "expected"),
    [
        ("Days Birth", "-11798", "32 years"),
        ("Amount Credit", "2.7e+05", "270,000 monetary units"),
        ("Installment Late Ratio", "0.342", "34.2%"),
        ("External source 2", "0.764", "0.764"),
        ("External source 1", "missing", "Not available"),
    ],
)
def test_business_value_formatting(feature: str, raw_value: str, expected: str) -> None:
    assert format_applicant_value(feature, raw_value) == expected


def test_driver_translation_separates_business_and_technical_fields() -> None:
    result = parse_driver_list(
        "Amount Credit [2.7e+05] (-0.191); Installment Late Ratio [0.342] (+0.196)",
        "Changes predicted risk",
    )

    assert result["Feature"].tolist() == [
        "Requested credit amount",
        "Historical late-payment ratio",
    ]
    assert result["Applicant value"].tolist() == [
        "270,000 monetary units",
        "34.2%",
    ]
    assert result["SHAP contribution"].tolist() == [-0.191, 0.196]
