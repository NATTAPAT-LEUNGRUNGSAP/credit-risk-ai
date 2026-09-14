import pandas as pd
import streamlit as st

from app_utils.data import load_csv
from app_utils.domain import get_anonymized_applicant
from app_utils.presentation import parse_driver_list

cases = load_csv("anonymized_applicants.csv")
alias = st.selectbox(
    "Choose an anonymized validation applicant",
    cases["case_alias"].tolist(),
    key="applicant_alias",
    help="Aliases contain no original applicant identifier.",
)
applicant = get_anonymized_applicant(cases, alias)

st.caption("Precomputed example from the held-out validation set · Platt-calibrated probability")
prediction, band_card = st.columns([2, 1], vertical_alignment="center")
with prediction.container(border=True, height="stretch"):
    st.metric(
        "Calibrated probability of payment difficulty",
        f"{applicant['calibrated_probability']:.2%}",
    )
    st.caption(f"Underlying class-weighted XGBoost score: {applicant['xgboost_score']:.3f}")
with band_card.container(border=True, height="stretch"):
    band = applicant["risk_band"]
    colors = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(f":{colors[band]}-badge[{band} risk band]")
    st.write("Band derived from validation score distribution and observed default rates.")

with st.container(border=True):
    st.markdown("**Retrospective validation outcome**")
    outcome = "Payment difficulty observed" if applicant["actual_outcome"] == 1 else "No payment difficulty observed"
    st.write(outcome)
    st.caption(
        "This label is shown only for portfolio evaluation. It would not be available when making "
        "a real-time prediction and is not an input to the model."
    )

risk_factors = parse_driver_list(
    applicant["main_risk_drivers"], "Increases predicted risk"
)
protective_factors = parse_driver_list(
    applicant["main_protective_drivers"], "Decreases predicted risk"
)
business_factors = st.columns(2)
for column, title, icon, factors in [
    (business_factors[0], "Factors increasing risk", ":material/trending_up:", risk_factors),
    (
        business_factors[1],
        "Factors decreasing risk",
        ":material/trending_down:",
        protective_factors,
    ),
]:
    with column.container(border=True, height="stretch"):
        st.subheader(title, icon=icon)
        st.dataframe(
            factors[["Feature", "Applicant value", "Effect"]],
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature name"),
                "Applicant value": st.column_config.TextColumn("Applicant value"),
                "Effect": st.column_config.TextColumn("Model effect"),
            },
        )

with st.expander("Technical details", icon=":material/code:"):
    technical = pd.concat([risk_factors, protective_factors], ignore_index=True)
    st.caption("Exact values and SHAP contributions preserved from the Phase 3 case studies.")
    st.dataframe(
        technical[["Original feature", "Raw value", "SHAP contribution"]],
        hide_index=True,
        column_config={
            "SHAP contribution": st.column_config.NumberColumn(
                "SHAP contribution", format="%+.4f"
            )
        },
    )

st.warning(
    "SHAP values describe how features contributed to this model output. They do not establish "
    "causality, fairness, eligibility, or a reason to approve or decline credit.",
    icon=":material/warning:",
)
