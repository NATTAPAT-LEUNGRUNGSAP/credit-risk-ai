import streamlit as st

from app_utils.data import load_csv, load_json

summary = load_json("summary.json")
bands = load_csv("risk_bands.csv")

st.markdown(
    "A portfolio case study that ranks credit-default risk, calibrates model scores, and translates "
    "validation evidence into transparent decision scenarios."
)
with st.container(border=True):
    st.subheader("Historical behavior improved risk ranking", icon=":material/trending_up:")
    st.markdown(
        "**XGBoost ROC-AUC improved from 0.7628 → 0.7866 and PR-AUC from 0.2532 → 0.2882.**"
    )
    st.write(
        "The gain came from aggregating bureau, previous-application, installment, POS, and "
        "credit-card behavior instead of relying on application-form data alone."
    )

st.info(
    "This application is an educational decision-support demonstration. It does not approve or "
    "decline loans and does not represent Home Credit economics or policy.",
    icon=":material/info:",
)

with st.container(border=True):
    st.markdown("**What this system does**")
    st.write(
        "Applicant data + historical credit behavior → Default-risk model → Calibrated "
        "probability → Explainability → Business decision support"
    )

with st.container(horizontal=True):
    st.metric("XGBoost ROC-AUC", f"{summary['roc_auc']:.5f}", border=True)
    st.metric("XGBoost PR-AUC", f"{summary['pr_auc']:.5f}", border=True)
    st.metric("Dataset default rate", f"{summary['dataset_default_rate']:.2%}", border=True)
    st.metric("Validation applicants", f"{summary['validation_rows']:,}", border=True)

st.subheader("Observed risk-band outcomes", icon=":material/stacked_bar_chart:")
with st.container(horizontal=True):
    for row in bands.itertuples(index=False):
        st.metric(
            f"{row.risk_band} risk",
            f"{row.observed_default_rate:.2%}",
            f"{int(row.applicant_count):,} applicants",
            delta_color="off",
            border=True,
        )

st.subheader("Data and decision pipeline", icon=":material/account_tree:")
st.graphviz_chart(
    """
    digraph {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#E8F0FE", color="#1D4ED8"];
      raw [label="7 Home Credit\ntables"];
      agg [label="Leakage-safe\ncustomer aggregates"];
      model [label="Phase 2\nXGBoost"];
      calibration [label="Training-only\nPlatt calibration"];
      decision [label="Risk bands +\nscenario analysis"];
      explain [label="SHAP\nexplanations"];
      raw -> agg -> model -> calibration -> decision;
      model -> explain;
    }
    """
)

with st.container(border=True):
    st.markdown("**Why this matters**")
    st.write(
        "The model is evaluated on ranking quality and minority-class detection rather than "
        "accuracy alone. Business thresholds remain explicit assumptions, while SHAP makes the "
        "model's predictive contributions inspectable."
    )
