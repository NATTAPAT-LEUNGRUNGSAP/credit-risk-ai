from __future__ import annotations

import altair as alt
import pandas as pd

RISK_COLORS = {"Low": "#0F9D76", "Medium": "#D97706", "High": "#DC2626"}


def risk_band_count_chart(bands: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(bands)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("risk_band:N", title=None, sort=["Low", "Medium", "High"]),
            y=alt.Y("applicant_count:Q", title="Applicants"),
            color=alt.Color(
                "risk_band:N",
                title="Risk band",
                scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("risk_band:N", title="Risk band"),
                alt.Tooltip("applicant_count:Q", title="Applicants", format=","),
                alt.Tooltip("applicant_share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=300)
    )


def risk_band_rate_chart(bands: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(bands)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("risk_band:N", title=None, sort=["Low", "Medium", "High"]),
            y=alt.Y(
                "observed_default_rate:Q",
                title="Observed default rate",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "risk_band:N",
                title="Risk band",
                scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("risk_band:N", title="Risk band"),
                alt.Tooltip(
                    "observed_default_rate:Q", title="Observed default rate", format=".2%"
                ),
            ],
        )
        .properties(height=300)
    )


def probability_histogram_chart(histogram: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(histogram)
        .mark_bar(color="#2563EB")
        .encode(
            x=alt.X("Probability_lower:Q", title="Calibrated default probability", bin="binned"),
            x2="Probability_upper:Q",
            y=alt.Y("Applicant_count:Q", title="Applicants"),
            tooltip=[
                alt.Tooltip("Probability_lower:Q", title="From", format=".3f"),
                alt.Tooltip("Probability_upper:Q", title="To", format=".3f"),
                alt.Tooltip("Applicant_count:Q", title="Applicants", format=","),
            ],
        )
        .properties(height=320)
    )


def model_comparison_chart(models: pd.DataFrame) -> alt.Chart:
    long = models.melt(
        id_vars="Model",
        value_vars=["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Score:Q", title="Validation score", scale=alt.Scale(domain=[0, 0.85])),
            color=alt.Color("Model:N", title="Model"),
            xOffset="Model:N",
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")],
        )
        .properties(height=340)
    )


def confusion_matrix_chart(confusion: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(confusion).encode(
        x=alt.X("Predicted:N", sort=["No default", "Default"], title="Predicted outcome"),
        y=alt.Y("Actual:N", sort=["Default", "No default"], title="Actual outcome"),
    )
    heatmap = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Count:Q", format=",")],
    )
    labels = base.mark_text(fontSize=17).encode(
        text=alt.Text("Count:Q", format=","),
        color=alt.condition(alt.datum.Count > 10_000, alt.value("white"), alt.value("#0F172A")),
    )
    return (heatmap + labels).properties(height=320)


def cost_sensitivity_chart(costs: pd.DataFrame) -> alt.Chart:
    display = costs.copy()
    display["Cost ratio"] = display["fn_to_fp_cost_ratio"]
    metrics = display.melt(
        id_vars="Cost ratio",
        value_vars=["recall", "precision"],
        var_name="Metric",
        value_name="Score",
    )
    return (
        alt.Chart(metrics)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Cost ratio:N", title="Hypothetical FN:FP cost ratio", sort=None),
            y=alt.Y("Score:Q", title="Metric", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title=None),
            tooltip=["Cost ratio:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=300)
    )

