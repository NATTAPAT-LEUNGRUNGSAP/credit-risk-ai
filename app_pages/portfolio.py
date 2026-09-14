import streamlit as st

from app_utils.charts import (
    confusion_matrix_chart,
    probability_histogram_chart,
    risk_band_count_chart,
    risk_band_rate_chart,
)
from app_utils.data import load_csv, load_json

summary = load_json("summary.json")
bands = load_csv("risk_bands.csv")
histogram = load_csv("probability_histogram.csv")
confusion = load_csv("xgboost_confusion_matrix.csv")

st.markdown(
    "A concise end-to-end case study for credit-risk ranking, explainability, and transparent "
    "business scenario analysis."
)
st.markdown(
    "**Problem → Data → Feature engineering → Models → Explainability → Business decision**"
)

case_study = [
    ("Problem", "Rank payment-difficulty risk in an imbalanced portfolio."),
    ("Data", "307,511 applications across 7 relational Home Credit tables."),
    ("Feature engineering", "Leakage-safe customer-level behavioral aggregates."),
    ("Models", "Logistic Regression vs XGBoost vs ANN."),
    ("Explainability", "Platt calibration, risk bands, and SHAP explanations."),
    ("Business decision", "Cost-sensitive thresholds without automated approval."),
]
for row in (case_study[:3], case_study[3:]):
    columns = st.columns(3)
    for column, (label, description) in zip(columns, row, strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{label}**")
            st.caption(description)

st.caption("Held-out Phase 2 validation portfolio · 61,503 anonymized applicants")
with st.container(horizontal=True):
    st.metric("Applications", "307,511", border=True)
    st.metric("Relational tables", "7", border=True)
    st.metric("ROC-AUC", f"{summary['roc_auc']:.4f}", border=True)
    st.metric("Default prevalence", f"{summary['dataset_default_rate']:.2%}", border=True)

with st.container(border=True):
    st.markdown("**Application-only → Behavioral model**")
    st.write(
        "XGBoost ROC-AUC **0.7628 → 0.7866** · PR-AUC **0.2532 → 0.2882** · "
        "probabilities calibrated with Platt scaling"
    )

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Applicants by risk band", icon=":material/groups:")
    st.altair_chart(risk_band_count_chart(bands))
with right.container(border=True, height="stretch"):
    st.subheader("Observed default rate by band", icon=":material/query_stats:")
    st.altair_chart(risk_band_rate_chart(bands))

with st.expander("Technical validation views", icon=":material/analytics:"):
    left, right = st.columns([3, 2])
    with left.container(border=True, height="stretch"):
        st.subheader("Calibrated probability distribution", icon=":material/monitoring:")
        st.altair_chart(probability_histogram_chart(histogram))
    with right.container(border=True, height="stretch"):
        st.subheader("XGBoost confusion matrix", icon=":material/grid_view:")
        st.altair_chart(confusion_matrix_chart(confusion))
        st.caption("Original class-weighted XGBoost threshold = 0.50, retained from Phase 2.")

    st.dataframe(
        bands[
            [
                "risk_band",
                "applicant_count",
                "applicant_share",
                "observed_default_rate",
                "probability_min",
                "probability_max",
            ]
        ],
        hide_index=True,
        column_config={
            "risk_band": "Risk band",
            "applicant_count": st.column_config.NumberColumn("Applicants", format="%,d"),
            "applicant_share": st.column_config.NumberColumn("Portfolio share", format="percent"),
            "observed_default_rate": st.column_config.NumberColumn(
                "Observed default rate", format="percent"
            ),
            "probability_min": st.column_config.NumberColumn(
                "Minimum probability", format="percent"
            ),
            "probability_max": st.column_config.NumberColumn(
                "Maximum probability", format="percent"
            ),
        },
    )
