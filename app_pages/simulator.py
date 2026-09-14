import streamlit as st

from app_utils.charts import cost_sensitivity_chart
from app_utils.data import load_csv

costs = load_csv("cost_sensitivity.csv")
ratios = costs["assumed_fn_cost"].astype(int).tolist()
ratio = st.segmented_control(
    "Assumed relative cost of a false negative",
    ratios,
    default=10,
    required=True,
    format_func=lambda value: f"{value}:1",
    key="fn_fp_ratio",
    help="False-positive cost is fixed to one illustrative unit.",
)
scenario = costs.loc[costs["assumed_fn_cost"] == ratio].iloc[0]

with st.container(border=True):
    st.markdown(
        f"**Assume missing a true defaulter costs {ratio}× more than flagging a "
        "non-defaulting applicant.**"
    )
    st.caption(
        "False negative: a future defaulter is missed. False positive: a non-defaulting "
        "applicant is flagged for additional review."
    )

st.warning(
    "All cost units are hypothetical sensitivity assumptions. They are not actual Home Credit, "
    "bank, insurance, loss-given-default, or customer-lifetime-value economics.",
    icon=":material/warning:",
)

with st.container(border=True):
    st.subheader("Recommended operating point under this assumption", icon=":material/tune:")
    with st.container(horizontal=True):
        st.metric("Calibrated threshold", f"{scenario['threshold']:.4f}", border=True)
        st.metric("Recall", f"{scenario['recall']:.1%}", border=True)
        st.metric("Precision", f"{scenario['precision']:.1%}", border=True)
        st.metric(
            "False positives",
            f"{int(scenario['fp']):,}",
            border=True,
            help="Non-defaulting validation applicants flagged at this threshold.",
        )
        st.metric(
            "False negatives",
            f"{int(scenario['fn']):,}",
            border=True,
            help="Defaulting validation applicants missed at this threshold.",
        )

st.info(
    "Higher cost of missed defaults → Lower threshold → Higher recall → More applicants flagged",
    icon=":material/arrow_forward:",
)

left, right = st.columns([3, 2])
with left.container(border=True, height="stretch"):
    st.subheader("Sensitivity across tested assumptions", icon=":material/swap_horiz:")
    st.altair_chart(cost_sensitivity_chart(costs))
with right.container(border=True, height="stretch"):
    st.subheader("Scenario details", icon=":material/receipt_long:")
    st.table(
        {
            "True positives": f"{int(scenario['tp']):,}",
            "True negatives": f"{int(scenario['tn']):,}",
            "Specificity": f"{scenario['specificity']:.1%}",
            "F1": f"{scenario['f1']:.3f}",
            "Illustrative total cost": f"{int(scenario['total_illustrative_cost_units']):,} units",
        },
        border="horizontal",
    )

st.info(
    "As false negatives become more expensive, the cost-minimizing threshold falls, recall rises, "
    "and more non-default applicants are flagged. A production threshold requires real financial, "
    "capacity, fairness, and regulatory inputs.",
    icon=":material/info:",
)
