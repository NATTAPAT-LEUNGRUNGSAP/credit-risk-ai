import streamlit as st

from app_utils.charts import model_comparison_chart
from app_utils.data import demo_figure, load_csv
from app_utils.presentation import human_feature_label

models = load_csv("model_comparison.csv")
importance = load_csv("shap_global_importance.csv")
calibration = load_csv("calibration_comparison.csv")

with st.container(border=True):
    st.subheader("Selected model: XGBoost", icon=":material/check_circle:")
    st.markdown("**ROC-AUC 0.7866 | PR-AUC 0.2882 | F1 0.3070**")
    st.write(
        "XGBoost was selected for the decision-support demo because it provided the strongest "
        "overall discrimination and precision-recall performance. ANN achieved the highest "
        "recall and remains an important comparison model."
    )

st.subheader("Phase 2 model comparison", icon=":material/leaderboard:")
st.altair_chart(model_comparison_chart(models))

with st.expander("Full model metrics", icon=":material/table_chart:"):
    st.dataframe(
        models,
        hide_index=True,
        column_config={
            metric: st.column_config.NumberColumn(metric, format="%.4f")
            for metric in ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
        },
    )

st.subheader("Global SHAP explanation", icon=":material/psychology:")
top_features = importance.head(15).copy()
top_features["feature"] = top_features["feature"].map(human_feature_label)
st.dataframe(
    top_features[["rank", "feature", "mean_absolute_shap"]],
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "feature": st.column_config.TextColumn("Predictive feature"),
        "mean_absolute_shap": st.column_config.ProgressColumn(
            "Relative SHAP importance",
            format="%.4f",
            min_value=0.0,
            max_value=float(top_features["mean_absolute_shap"].max()),
        ),
    },
)

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**Plain-language interpretation**")
    st.markdown(
        "- Higher external source scores generally reduce predicted risk.\n"
        "- Higher installment lateness and previous refusal ratios increase predicted risk.\n"
        "- Credit amount, annuity, and debt-to-credit behavior materially influence the score.\n"
        "- POS and repayment-history features add signal beyond the application form."
    )
with right.container(border=True, height="stretch"):
    st.markdown("**Calibration evidence**")
    platt = calibration.loc[calibration["method"] == "Platt scaling"].iloc[0]
    uncalibrated = calibration.loc[calibration["method"] == "Uncalibrated"].iloc[0]
    st.metric("Platt Brier score", f"{platt['brier_score']:.4f}")
    st.caption(
        f"Uncalibrated Brier score: {uncalibrated['brier_score']:.4f}. "
        "Lower is better; calibration was fitted on training data only."
    )

with st.expander("Full technical SHAP visualization", icon=":material/scatter_plot:"):
    st.image(demo_figure("shap_beeswarm.png"), width="stretch")
    st.caption(
        "Red points are higher feature values; blue points are lower values. Positive SHAP values "
        "push the model toward higher predicted risk. Effects are associations, not causality."
    )

with st.expander("Calibration curve and metrics", icon=":material/straighten:"):
    st.image(demo_figure("calibration_curve.png"), width="stretch")
    st.dataframe(
        calibration,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column.replace("_", " ").title(), format="%.4f")
            for column in calibration.columns
            if column != "method"
        },
    )
