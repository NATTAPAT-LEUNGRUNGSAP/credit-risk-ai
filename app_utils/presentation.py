from __future__ import annotations

import re

import pandas as pd

FEATURE_LABELS = {
    "Amount Annuity": "Loan annuity",
    "Amount Credit": "Requested credit amount",
    "Amount Goods Price": "Financed goods price",
    "Bureau Days Credit Max": "Most recent bureau credit",
    "Bureau Days Credit Min": "Oldest bureau credit",
    "Bureau Debt To Credit Ratio": "Historical debt-to-credit ratio",
    "Code Gender = F": "Female gender indicator",
    "Code Gender = M": "Male gender indicator",
    "Credit card Balance Latest": "Latest credit-card balance",
    "Credit card Drawing Count Sum": "Credit-card drawing count",
    "Credit card Utilization Max": "Peak credit-card utilization",
    "Days Birth": "Age",
    "Days Employed": "Employment history",
    "External source 1": "External credit score #1",
    "External source 2": "External credit score #2",
    "External source 3": "External credit score #3",
    "Installment Late Ratio": "Historical late-payment ratio",
    "Installment Payment Amount Sum": "Historical installment payments",
    "Name Education Type = Higher Education": "Higher-education indicator",
    "Name Family Status = Married": "Married-status indicator",
    "POS Future Instalments Max": "Maximum remaining POS installments",
    "POS Future Instalments Mean": "Average remaining POS installments",
    "POS Record Count": "POS account-history records",
    "Previous Credit To Application Mean": "Previous credit-to-request ratio",
    "Previous Days Decision Max": "Most recent previous application",
    "Previous Refused Ratio": "Previous application rejection ratio",
}

MONEY_FEATURES = {
    "Amount Annuity",
    "Amount Credit",
    "Amount Goods Price",
    "Credit card Balance Latest",
    "Installment Payment Amount Sum",
}
RATIO_FEATURES = {
    "Bureau Debt To Credit Ratio",
    "Credit card Utilization Max",
    "Installment Late Ratio",
    "Previous Refused Ratio",
}
COUNT_FEATURES = {"Credit card Drawing Count Sum", "POS Record Count"}
RELATIVE_DAY_FEATURES = {
    "Bureau Days Credit Max",
    "Bureau Days Credit Min",
    "Previous Days Decision Max",
}
DRIVER_PATTERN = re.compile(
    r"^(?P<feature>.+?) \[(?P<value>.+?)\] \((?P<contribution>[+-]?\d+(?:\.\d+)?)\)$"
)


def human_feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def format_applicant_value(feature: str, raw_value: str) -> str:
    if raw_value == "missing":
        return "Not available"
    if raw_value in {"present", "absent"}:
        return raw_value.capitalize()

    value = float(raw_value)
    if feature == "Days Birth":
        return f"{round(abs(value) / 365.25):,} years"
    if feature == "Days Employed":
        return f"{abs(value) / 365.25:,.1f} years"
    if feature in RELATIVE_DAY_FEATURES:
        return f"{abs(value) / 365.25:,.1f} years before application"
    if feature in MONEY_FEATURES:
        return f"{value:,.0f} monetary units"
    if feature in RATIO_FEATURES:
        return f"{value:.1%}"
    if feature == "Previous Credit To Application Mean":
        return f"{value:.2f}×"
    if feature in COUNT_FEATURES:
        return f"{value:,.0f} records"
    if "Instalments" in feature:
        return f"{value:,.1f} installments"
    if feature.startswith("External source"):
        return f"{value:.3f}"
    return f"{value:,.3g}"


def parse_driver_list(driver_text: str, effect: str) -> pd.DataFrame:
    rows = []
    for item in str(driver_text).split("; "):
        match = DRIVER_PATTERN.match(item)
        if match is None:
            raise ValueError(f"Unexpected SHAP driver format: {item}")
        feature = match.group("feature")
        raw_value = match.group("value")
        rows.append(
            {
                "Feature": human_feature_label(feature),
                "Applicant value": format_applicant_value(feature, raw_value),
                "Effect": effect,
                "Original feature": feature,
                "Raw value": raw_value,
                "SHAP contribution": float(match.group("contribution")),
            }
        )
    return pd.DataFrame(rows)
