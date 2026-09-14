"""Phase 3: SHAP, calibration, thresholds, costs, risk bands, and case studies."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    FIGURE_DIR,
    ID_COLUMN,
    METRIC_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    REPORT_DIR,
    TARGET,
    ensure_output_dirs,
)

SHAP_SAMPLE_SIZE = 5_000
CALIBRATION_FOLDS = 3
FN_FP_RATIOS = (2, 5, 10, 20)
PRESERVED_ARTIFACTS = (
    REPORT_DIR / "metrics" / "baseline_metrics.json",
    REPORT_DIR / "metrics" / "phase2_metrics.json",
    MODEL_DIR / "phase2_xgboost.json",
    MODEL_DIR / "phase2_preprocessor.joblib",
    PROCESSED_DIR / "phase2_validation_predictions.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", min_frequency=20, dtype=np.float32
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )


def make_xgboost(scale_pos_weight: float, n_estimators: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_phase2_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    application = pd.read_csv(RAW_DIR / "application_train.csv", low_memory=False)
    relational = pd.read_parquet(PROCESSED_DIR / "phase2_relational_features.parquet")
    relational_columns = [column for column in relational if column != ID_COLUMN]
    merged = application.merge(relational, on=ID_COLUMN, how="left", validate="one_to_one")
    merged[relational_columns] = (
        merged[relational_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    merged = merged.set_index(ID_COLUMN, verify_integrity=True)
    y = merged.pop(TARGET).astype(np.int8)
    train_ids = pd.read_csv(PROCESSED_DIR / "train_split_ids.csv")[ID_COLUMN].tolist()
    valid_ids = pd.read_csv(PROCESSED_DIR / "validation_split_ids.csv")[ID_COLUMN].tolist()
    return merged.loc[train_ids], y.loc[train_ids], merged.loc[valid_ids], y.loc[valid_ids]


def human_feature_names(preprocessor: ColumnTransformer, transformed: np.ndarray) -> list[str]:
    categorical = list(preprocessor.transformers_[1][2])
    labels = []
    for raw_name in transformed:
        name = raw_name.split("__", 1)[-1]
        if raw_name.startswith("cat__"):
            matched = False
            for column in categorical:
                prefix = f"{column}_"
                if name.startswith(prefix):
                    value = name[len(prefix) :]
                    name = f"{column} = {value}"
                    matched = True
                    break
            if not matched:
                name = name.replace("_", " ")
        label = name.replace("_", " ").title()
        replacements = {
            "Ext Source": "External source",
            "Amt ": "Amount ",
            "Cnt ": "Count ",
            "Dpd": "Days past due",
            "Cc ": "Credit card ",
            "Pos ": "POS ",
            "Prev ": "Previous ",
            "Inst ": "Installment ",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        labels.append(label)
    return labels


def analyze_shap(
    model: XGBClassifier,
    X_valid_t: sparse.csr_matrix,
    probabilities: np.ndarray,
    feature_labels: list[str],
    transformed_names: np.ndarray,
) -> tuple[shap.TreeExplainer, pd.DataFrame, float]:
    order = np.argsort(probabilities)
    sample_positions = np.linspace(0, len(order) - 1, SHAP_SAMPLE_SIZE, dtype=int)
    sample_indices = order[sample_positions]
    sample_sparse = X_valid_t[sample_indices]
    sample = sample_sparse.toarray().astype(np.float32)
    explainer = shap.TreeExplainer(model)
    # Preserve sparse representation: XGBoost routes absent CSR entries as missing values.
    # Explaining a dense copy would change those paths for one-hot zeroes.
    shap_values = np.asarray(explainer.shap_values(sample_sparse, check_additivity=False))
    margins = model.predict(sample_sparse, output_margin=True)
    reconstructed = float(np.asarray(explainer.expected_value).reshape(-1)[0]) + shap_values.sum(axis=1)
    additivity_error = float(np.max(np.abs(margins - reconstructed)))
    mean_abs = np.abs(shap_values).mean(axis=0)
    records = []
    for index in np.argsort(mean_abs)[::-1]:
        values = sample[:, index]
        contributions = shap_values[:, index]
        unique = np.unique(values)
        if len(unique) <= 2 and set(np.round(unique, 6)).issubset({0.0, 1.0}):
            present = contributions[values > 0.5]
            absent = contributions[values <= 0.5]
            present_mean = float(present.mean()) if len(present) else np.nan
            absent_mean = float(absent.mean()) if len(absent) else np.nan
            difference = present_mean - absent_mean
            direction = (
                "presence tends to increase risk"
                if difference > 0
                else "presence tends to decrease risk"
            )
            association = difference
        elif float(np.nanstd(values)) == 0.0 or float(np.nanstd(contributions)) == 0.0:
            association = 0.0
            direction = "constant in SHAP sample"
        else:
            correlation = spearmanr(values, contributions, nan_policy="omit").statistic
            association = float(correlation) if np.isfinite(correlation) else 0.0
            if association > 0.10:
                direction = "higher values tend to increase risk"
            elif association < -0.10:
                direction = "higher values tend to decrease risk"
            else:
                direction = "mixed or non-monotonic effect"
        records.append(
            {
                "rank": len(records) + 1,
                "feature": feature_labels[index],
                "transformed_feature": transformed_names[index],
                "mean_absolute_shap": float(mean_abs[index]),
                "directional_association": association,
                "direction": direction,
            }
        )
    importance = pd.DataFrame(records)
    importance.to_csv(REPORT_DIR / "phase3_shap_global_importance.csv", index=False)

    top = importance.head(20).sort_values("mean_absolute_shap")
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature"], top["mean_absolute_shap"], color="#7C3AED")
    axis.set(title="Phase 2 XGBoost — global SHAP importance", xlabel="Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_shap_global_importance.png", dpi=170)
    plt.close(fig)

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_labels,
        max_display=25,
        show=False,
    )
    plt.gcf().set_size_inches(11, 9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "phase3_shap_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.close()
    np.savez_compressed(
        PROCESSED_DIR / "phase3_shap_sample.npz",
        validation_positions=sample_indices,
        shap_values=shap_values,
        feature_values=sample,
    )
    return explainer, importance, additivity_error


def classification_at_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def threshold_analysis(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(np.concatenate(([0.0], np.linspace(0.01, 0.99, 99), [1.0])))
    table = pd.DataFrame(
        [classification_at_threshold(y_true, probabilities, value) for value in thresholds]
    )
    table.to_csv(REPORT_DIR / "phase3_threshold_metrics.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 5))
    for metric, color in (("precision", "#2563EB"), ("recall", "#DC2626"), ("f1", "#059669")):
        axis.plot(table["threshold"], table[metric], label=metric.title(), color=color)
    axis.set(xlabel="Decision threshold", ylabel="Metric", ylim=(0, 1))
    axis.set_title("Precision–recall trade-off across calibrated thresholds")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_precision_recall_threshold.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(table["threshold"], table["fp"], label="False positives", color="#D97706")
    axis.plot(table["threshold"], table["fn"], label="False negatives", color="#7C3AED")
    axis.set(xlabel="Decision threshold", ylabel="Applicant count")
    axis.set_title("Operational error trade-off")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_fp_fn_tradeoff.png", dpi=170)
    plt.close(fig)
    return table


def exact_cost_optimum(
    y_true: np.ndarray, probabilities: np.ndarray, fn_cost: int, fp_cost: int = 1
) -> dict[str, float | int]:
    ordered = pd.DataFrame({"probability": probabilities, "target": y_true}).sort_values(
        "probability", ascending=False
    )
    grouped = ordered.groupby("probability", sort=False)["target"].agg(["sum", "count"])
    tp = grouped["sum"].cumsum().to_numpy(dtype=np.int64)
    predicted_positive = grouped["count"].cumsum().to_numpy(dtype=np.int64)
    fp = predicted_positive - tp
    positives = int(np.sum(y_true))
    fn = positives - tp
    costs = fn_cost * fn + fp_cost * fp
    best = int(np.argmin(costs))
    result = classification_at_threshold(y_true, probabilities, float(grouped.index[best]))
    result.update(
        {
            "assumed_fn_cost": fn_cost,
            "assumed_fp_cost": fp_cost,
            "fn_to_fp_cost_ratio": f"{fn_cost}:{fp_cost}",
            "total_illustrative_cost_units": int(costs[best]),
            "cost_units_per_1000_applicants": float(costs[best] / len(y_true) * 1_000),
        }
    )
    return result


def cost_analysis(
    y_true: np.ndarray, probabilities: np.ndarray, threshold_table: pd.DataFrame
) -> pd.DataFrame:
    table = pd.DataFrame(
        [exact_cost_optimum(y_true, probabilities, ratio) for ratio in FN_FP_RATIOS]
    )
    table.to_csv(REPORT_DIR / "phase3_cost_sensitivity.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 5))
    for ratio in FN_FP_RATIOS:
        cost = ratio * threshold_table["fn"] + threshold_table["fp"]
        axis.plot(
            threshold_table["threshold"],
            cost / len(y_true) * 1_000,
            label=f"FN:FP = {ratio}:1",
        )
    axis.set(xlabel="Decision threshold", ylabel="Illustrative cost units per 1,000 applicants")
    axis.set_title("Sensitivity to hypothetical error costs")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_cost_sensitivity.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(table), figsize=(18, 4))
    for axis, row in zip(axes, table.itertuples(index=False)):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Oranges", cbar=False, ax=axis)
        axis.set(
            title=f"FN:FP {row.fn_to_fp_cost_ratio}\nthreshold={row.threshold:.3f}",
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_cost_optimal_confusion_matrices.png", dpi=170)
    plt.close(fig)
    return table


def calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"target": y_true, "probability": probabilities})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        count=("target", "size"), observed=("target", "mean"), predicted=("probability", "mean")
    )
    return float(
        np.sum(grouped["count"] / len(frame) * np.abs(grouped["observed"] - grouped["predicted"]))
    )


def fit_calibrators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_validation_probabilities: np.ndarray,
    y_valid: pd.Series,
    fixed_estimators: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    splitter = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train), dtype=np.float64)
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(X_train, y_train), start=1):
        print(f"Calibration OOF fold {fold}/{CALIBRATION_FOLDS}", flush=True)
        preprocessor = build_preprocessor(numerical, categorical)
        fit_matrix = preprocessor.fit_transform(X_train.iloc[fit_idx]).astype(np.float32)
        holdout_matrix = preprocessor.transform(X_train.iloc[holdout_idx]).astype(np.float32)
        positives = int(y_train.iloc[fit_idx].sum())
        negatives = len(fit_idx) - positives
        model = make_xgboost(negatives / positives, fixed_estimators)
        model.fit(fit_matrix, y_train.iloc[fit_idx], verbose=False)
        oof[holdout_idx] = model.predict_proba(holdout_matrix)[:, 1]

    epsilon = 1e-6
    oof_logit = np.log(np.clip(oof, epsilon, 1 - epsilon) / np.clip(1 - oof, epsilon, 1))
    valid_logit = np.log(
        np.clip(base_validation_probabilities, epsilon, 1 - epsilon)
        / np.clip(1 - base_validation_probabilities, epsilon, 1)
    )
    platt = LogisticRegression(C=1e6, solver="lbfgs", random_state=RANDOM_STATE)
    platt.fit(oof_logit.reshape(-1, 1), y_train)
    platt_valid = platt.predict_proba(valid_logit.reshape(-1, 1))[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(oof, y_train)
    isotonic_valid = isotonic.predict(base_validation_probabilities)
    joblib.dump(platt, MODEL_DIR / "phase3_platt_calibrator.joblib", compress=3)
    joblib.dump(isotonic, MODEL_DIR / "phase3_isotonic_calibrator.joblib", compress=3)
    np.save(PROCESSED_DIR / "phase3_calibration_oof_predictions.npy", oof)

    candidates = {
        "Uncalibrated": base_validation_probabilities,
        "Platt scaling": platt_valid,
        "Isotonic regression": isotonic_valid,
    }
    records = []
    for name, values in candidates.items():
        records.append(
            {
                "method": name,
                "brier_score": float(brier_score_loss(y_valid, values)),
                "log_loss": float(log_loss(y_valid, values)),
                "expected_calibration_error_10bin": calibration_error(y_valid.to_numpy(), values),
                "roc_auc": float(roc_auc_score(y_valid, values)),
                "pr_auc": float(average_precision_score(y_valid, values)),
            }
        )
    comparison = pd.DataFrame(records).sort_values("brier_score")
    comparison.to_csv(REPORT_DIR / "phase3_calibration_comparison.csv", index=False)
    selected = str(comparison.iloc[0]["method"])

    fig, axis = plt.subplots(figsize=(7, 6))
    for name, values in candidates.items():
        observed, predicted = calibration_curve(y_valid, values, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed default rate")
    axis.set_title("XGBoost probability calibration")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_calibration_curve.png", dpi=170)
    plt.close(fig)
    return candidates[selected], comparison, selected


def derive_risk_bands(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame({"target": y_true, "probability": probabilities})
    frame["decile"] = pd.qcut(
        frame["probability"].rank(method="first"), q=10, labels=False
    ).astype(int)
    deciles = frame.groupby("decile").agg(
        applicant_count=("target", "size"),
        observed_default_rate=("target", "mean"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
    )
    deciles.to_csv(REPORT_DIR / "phase3_risk_deciles.csv")

    best = None
    global_rate = float(frame["target"].mean())
    for low_end in range(8):
        for medium_end in range(low_end + 1, 9):
            groups = [
                frame[frame["decile"] <= low_end],
                frame[(frame["decile"] > low_end) & (frame["decile"] <= medium_end)],
                frame[frame["decile"] > medium_end],
            ]
            rates = [float(group["target"].mean()) for group in groups]
            if not (rates[0] < rates[1] < rates[2]):
                continue
            score = sum(len(group) * (rate - global_rate) ** 2 for group, rate in zip(groups, rates))
            if best is None or score > best[0]:
                best = (score, low_end, medium_end)
    if best is None:
        raise ValueError("Unable to derive monotonic risk bands from validation deciles")
    _, low_end, medium_end = best
    low_cut = float(deciles.loc[low_end, "probability_max"])
    medium_cut = float(deciles.loc[medium_end, "probability_max"])
    frame["risk_band"] = pd.cut(
        frame["probability"],
        bins=[-np.inf, low_cut, medium_cut, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    bands = frame.groupby("risk_band", observed=True).agg(
        applicant_count=("target", "size"),
        applicant_share=("target", lambda values: len(values) / len(frame)),
        observed_defaults=("target", "sum"),
        observed_default_rate=("target", "mean"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
        probability_mean=("probability", "mean"),
    )
    bands.to_csv(REPORT_DIR / "phase3_risk_bands.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(frame["probability"], bins=50, color="#2563EB", alpha=0.85)
    axes[0].axvline(low_cut, color="#059669", linestyle="--", label=f"Low ≤ {low_cut:.3f}")
    axes[0].axvline(
        medium_cut, color="#D97706", linestyle="--", label=f"Medium ≤ {medium_cut:.3f}"
    )
    axes[0].set(
        title="Calibrated validation risk distribution",
        xlabel="Calibrated default probability",
        ylabel="Applicants",
    )
    axes[0].legend()
    axes[1].bar(
        bands.index.astype(str),
        bands["observed_default_rate"],
        color=["#059669", "#D97706", "#DC2626"],
    )
    axes[1].set(
        title="Observed default rate by risk band",
        xlabel="Risk band",
        ylabel="Observed default rate",
        ylim=(0, float(bands["observed_default_rate"].max()) * 1.2),
    )
    for index, value in enumerate(bands["observed_default_rate"]):
        axes[1].text(index, value, f"{value:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase3_risk_bands.png", dpi=170)
    plt.close(fig)
    metadata = {
        "method": (
            "Search all validation decile boundaries with at least one decile per band; enforce "
            "monotonic observed default rates; maximize weighted between-band rate separation."
        ),
        "low_upper_probability": low_cut,
        "medium_upper_probability": medium_cut,
        "selected_low_end_decile_zero_based": low_end,
        "selected_medium_end_decile_zero_based": medium_end,
        "limitation": "Descriptive validation-derived bands; confirm on a future holdout before use.",
    }
    return bands, metadata


def create_case_studies(
    explainer: shap.TreeExplainer,
    X_valid_t: sparse.csr_matrix,
    y_valid: pd.Series,
    raw_probabilities: np.ndarray,
    decision_probabilities: np.ndarray,
    band_metadata: dict,
    feature_labels: list[str],
    transformed_names: np.ndarray,
    X_valid_raw: pd.DataFrame,
) -> pd.DataFrame:
    low_cut = band_metadata["low_upper_probability"]
    medium_cut = band_metadata["medium_upper_probability"]
    bands = np.select(
        [decision_probabilities <= low_cut, decision_probabilities <= medium_cut],
        ["Low", "Medium"],
        default="High",
    )
    selection = pd.DataFrame(
        {
            "position": np.arange(len(y_valid)),
            "actual": y_valid.to_numpy(),
            "raw_probability": raw_probabilities,
            "calibrated_probability": decision_probabilities,
            "risk_band": bands,
        }
    )
    chosen = []
    for band in ("Low", "Medium", "High"):
        band_frame = selection[selection["risk_band"] == band]
        median = float(band_frame["calibrated_probability"].median())
        for actual in (0, 1):
            candidates = band_frame[band_frame["actual"] == actual].copy()
            candidates["distance"] = (candidates["calibrated_probability"] - median).abs()
            chosen.append(candidates.sort_values("distance").iloc[0])
    chosen_frame = pd.DataFrame(chosen).reset_index(drop=True)
    chosen_positions = chosen_frame["position"].astype(int).to_numpy()
    case_sparse = X_valid_t[chosen_positions]
    values = case_sparse.toarray().astype(np.float32)
    explanations = np.asarray(explainer.shap_values(case_sparse, check_additivity=False))
    display_values = np.empty(values.shape, dtype=object)
    for feature_index, transformed in enumerate(transformed_names):
        transformed = str(transformed)
        if transformed.startswith("cat__"):
            display_values[:, feature_index] = np.where(
                values[:, feature_index] > 0.5, "present", "absent"
            )
        else:
            source_column = transformed.split("__", 1)[-1]
            display_values[:, feature_index] = X_valid_raw.iloc[chosen_positions][
                source_column
            ].to_numpy()
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    public_records = []
    internal_records = []
    counters = {"Low": 0, "Medium": 0, "High": 0}
    for row_number, row in chosen_frame.iterrows():
        band = str(row["risk_band"])
        counters[band] += 1
        alias = f"{band[0]}-{counters[band]:02d}"
        shap_row = explanations[row_number]
        positive = np.argsort(shap_row)[::-1]
        negative = np.argsort(shap_row)
        def driver_text(
            index: int,
            case_row_number: int = row_number,
            case_position: int = int(row["position"]),
            case_shap: np.ndarray = shap_row,
        ) -> str:
            transformed = str(transformed_names[index])
            if transformed.startswith("cat__"):
                displayed_value = "present" if values[case_row_number, index] > 0.5 else "absent"
            else:
                source_column = transformed.split("__", 1)[-1]
                raw_value = X_valid_raw.iloc[case_position][source_column]
                if pd.isna(raw_value):
                    displayed_value = "missing"
                elif isinstance(raw_value, (float, np.floating)):
                    displayed_value = f"{float(raw_value):.3g}"
                else:
                    displayed_value = str(raw_value)
            return f"{feature_labels[index]} [{displayed_value}] ({case_shap[index]:+.3f})"

        risk_drivers = [
            driver_text(index)
            for index in positive
            if shap_row[index] > 0
        ][:3]
        protective_drivers = [
            driver_text(index)
            for index in negative
            if shap_row[index] < 0
        ][:3]
        public_records.append(
            {
                "case_alias": alias,
                "risk_band": band,
                "actual_outcome": int(row["actual"]),
                "xgboost_score": float(row["raw_probability"]),
                "calibrated_probability": float(row["calibrated_probability"]),
                "main_risk_drivers": "; ".join(risk_drivers),
                "main_protective_drivers": "; ".join(protective_drivers),
            }
        )
        position = int(row["position"])
        internal_records.append(
            {"case_alias": alias, ID_COLUMN: int(y_valid.index[position]), "validation_position": position}
        )
        if counters[band] == 1:
            explanation = shap.Explanation(
                values=shap_row,
                base_values=expected_value,
                data=display_values[row_number],
                feature_names=feature_labels,
            )
            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.gcf().set_size_inches(10, 7)
            plt.tight_layout()
            plt.savefig(
                FIGURE_DIR / f"phase3_case_{band.lower()}_waterfall.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close()
    public = pd.DataFrame(public_records)
    public.to_csv(REPORT_DIR / "phase3_case_studies.csv", index=False)
    pd.DataFrame(internal_records).to_csv(
        PROCESSED_DIR / "phase3_case_study_internal_ids.csv", index=False
    )
    return public


def write_report(
    importance: pd.DataFrame,
    calibration: pd.DataFrame,
    selected_calibration: str,
    threshold_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    bands: pd.DataFrame,
    band_metadata: dict,
    cases: pd.DataFrame,
    primary_metrics: dict,
) -> None:
    at_half = threshold_table.iloc[(threshold_table["threshold"] - 0.5).abs().argmin()]
    top_features = importance.head(10)
    lines = [
        "# Phase 3 Explainability and Business Decision Analysis",
        "",
        (
            "The primary model is the saved Phase 2 XGBoost model. ANN remains the recall-oriented "
            "comparison model. Phase 1 and Phase 2 artifacts are unchanged."
        ),
        "",
        "## Actual model results carried forward",
        "",
        f"- XGBoost ROC-AUC: {primary_metrics['roc_auc']:.6f}",
        f"- XGBoost PR-AUC: {primary_metrics['pr_auc']:.6f}",
        f"- XGBoost recall at the original 0.50 threshold: {primary_metrics['recall']:.6f}",
        f"- ANN recall at 0.50: {primary_metrics['ann_recall']:.6f}",
        "",
        "## Global SHAP drivers",
        "",
        (
            "SHAP values explain changes in the XGBoost raw margin (log-odds). Positive SHAP values "
            "push predicted default risk upward; negative values push it downward."
        ),
        "",
        "| Rank | Feature | Mean absolute SHAP | Directional pattern |",
        "|---:|---|---:|---|",
        *[
            f"| {row['rank']} | {row['feature']} | {row['mean_absolute_shap']:.5f} | "
            f"{row['direction']} |"
            for _, row in top_features.iterrows()
        ],
        "",
        (
            "Directional labels summarize association in the SHAP sample and are not causal claims. "
            "Mixed effects should be read from the beeswarm plot."
        ),
        "",
        "## Probability calibration",
        "",
        (
            "Calibration mappings were fitted from three-fold out-of-fold predictions using only the "
            "Phase 1 training IDs. Validation labels were not used to fit either mapping."
        ),
        "",
        "| Method | Brier score | Log loss | 10-bin ECE | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row.method} | {row.brier_score:.6f} | {row.log_loss:.6f} | "
            f"{row.expected_calibration_error_10bin:.6f} | {row.roc_auc:.6f} | {row.pr_auc:.6f} |"
            for row in calibration.itertuples(index=False)
        ],
        "",
        (
            f"Selected for decision analysis: **{selected_calibration}**, based on the lowest validation "
            "Brier score. This is post-hoc calibration, not a new ranking model."
        ),
        "",
        "## Threshold analysis",
        "",
        (
            "Decision analysis uses the selected calibrated probabilities. At threshold 0.50: "
            f"TP={int(at_half.tp):,}, FP={int(at_half.fp):,}, TN={int(at_half.tn):,}, "
            f"FN={int(at_half.fn):,}, recall={at_half.recall:.4f}, "
            f"precision={at_half.precision:.4f}, specificity={at_half.specificity:.4f}, "
            f"F1={at_half.f1:.4f}."
        ),
        "",
        "The complete threshold grid is in `phase3_threshold_metrics.csv`.",
        "",
        "## Illustrative business cost sensitivity",
        "",
        (
            "**These are hypothetical cost units, not Home Credit or bank economics.** False-positive "
            "cost is fixed to one unit and false-negative cost is varied."
        ),
        "",
        "| Assumed FN:FP cost | Cost-minimizing threshold | TP | FP | TN | FN | Recall | Precision | Cost units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.fn_to_fp_cost_ratio} | {row.threshold:.5f} | {row.tp:,} | {row.fp:,} | "
            f"{row.tn:,} | {row.fn:,} | {row.recall:.4f} | {row.precision:.4f} | "
            f"{row.total_illustrative_cost_units:,} |"
            for row in cost_table.itertuples(index=False)
        ],
        "",
        (
            "There is no universal optimal threshold: it changes with the assumed relative cost of "
            "missing a default versus reviewing or declining a non-default applicant."
        ),
        "",
        "## Validation-derived risk bands",
        "",
        band_metadata["method"],
        "",
        "| Risk band | Applicants | Share | Observed defaults | Observed default rate | Probability range |",
        "|---|---:|---:|---:|---:|---|",
        *[
            f"| {index} | {int(row.applicant_count):,} | {row.applicant_share:.2%} | "
            f"{int(row.observed_defaults):,} | {row.observed_default_rate:.2%} | "
            f"{row.probability_min:.4f}–{row.probability_max:.4f} |"
            for index, row in bands.iterrows()
        ],
        "",
        f"Limitation: {band_metadata['limitation']}",
        "",
        "## Anonymized applicant case studies",
        "",
        (
            "The public case table contains aliases only. Positive SHAP values are risk drivers and "
            "negative values are protective drivers."
        ),
        "",
        "| Case | Band | Actual | Raw score | Calibrated probability | Main risk drivers | Main protective drivers |",
        "|---|---|---:|---:|---:|---|---|",
        *[
            f"| {row.case_alias} | {row.risk_band} | {row.actual_outcome} | "
            f"{row.xgboost_score:.4f} | {row.calibrated_probability:.4f} | "
            f"{row.main_risk_drivers} | {row.main_protective_drivers} |"
            for row in cases.itertuples(index=False)
        ],
        "",
        (
            "Applicant IDs are stored only in a Git-ignored internal artifact and are not exposed in "
            "portfolio reports."
        ),
    ]
    (REPORT_DIR / "phase3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    before_hashes = {str(path.relative_to(REPORT_DIR.parent)): sha256(path) for path in PRESERVED_ARTIFACTS}
    print("Loading Phase 2 data and artifacts", flush=True)
    X_train, y_train, X_valid, y_valid = load_phase2_data()
    preprocessor = joblib.load(MODEL_DIR / "phase2_preprocessor.joblib")
    X_valid_t = preprocessor.transform(X_valid).astype(np.float32)
    if not sparse.issparse(X_valid_t):
        X_valid_t = sparse.csr_matrix(X_valid_t)
    else:
        X_valid_t = X_valid_t.tocsr()
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "phase2_xgboost.json")
    raw_probabilities = model.predict_proba(X_valid_t)[:, 1]
    saved = pd.read_csv(PROCESSED_DIR / "phase2_validation_predictions.csv")
    max_prediction_difference = float(
        np.max(np.abs(raw_probabilities - saved["xgboost_probability"].to_numpy()))
    )
    if max_prediction_difference > 1e-6:
        raise ValueError(f"Loaded Phase 2 predictions differ by {max_prediction_difference}")

    transformed_names = preprocessor.get_feature_names_out()
    labels = human_feature_names(preprocessor, transformed_names)
    print("Computing global SHAP explanations", flush=True)
    explainer, importance, shap_additivity_error = analyze_shap(
        model, X_valid_t, raw_probabilities, labels, transformed_names
    )

    with (METRIC_DIR / "phase2_metrics.json").open(encoding="utf-8") as handle:
        phase2_metrics = json.load(handle)["metrics_at_threshold_0_5"]
    fixed_estimators = int(phase2_metrics["XGBoost"]["best_iteration"]) + 1
    calibrated_probabilities, calibration, selected_calibration = fit_calibrators(
        X_train,
        y_train,
        raw_probabilities,
        y_valid,
        fixed_estimators,
    )
    del X_train
    print("Evaluating thresholds, costs, and risk bands", flush=True)
    threshold_table = threshold_analysis(y_valid.to_numpy(), calibrated_probabilities)
    cost_table = cost_analysis(y_valid.to_numpy(), calibrated_probabilities, threshold_table)
    bands, band_metadata = derive_risk_bands(y_valid.to_numpy(), calibrated_probabilities)
    cases = create_case_studies(
        explainer,
        X_valid_t,
        y_valid,
        raw_probabilities,
        calibrated_probabilities,
        band_metadata,
        labels,
        transformed_names,
        X_valid,
    )
    decision_predictions = saved[[ID_COLUMN, TARGET]].copy()
    decision_predictions["xgboost_score"] = raw_probabilities
    decision_predictions["calibrated_probability"] = calibrated_probabilities
    decision_predictions.to_csv(
        PROCESSED_DIR / "phase3_validation_decision_probabilities.csv", index=False
    )

    primary = {
        **phase2_metrics["XGBoost"],
        "ann_recall": phase2_metrics["ANN"]["recall"],
    }
    write_report(
        importance,
        calibration,
        selected_calibration,
        threshold_table,
        cost_table,
        bands,
        band_metadata,
        cases,
        primary,
    )
    after_hashes = {str(path.relative_to(REPORT_DIR.parent)): sha256(path) for path in PRESERVED_ARTIFACTS}
    preservation = {
        "before_sha256": before_hashes,
        "after_sha256": after_hashes,
        "all_preserved": before_hashes == after_hashes,
        "max_recomputed_vs_saved_phase2_prediction_difference": max_prediction_difference,
    }
    if not preservation["all_preserved"]:
        raise ValueError("A Phase 1 or Phase 2 artifact changed during Phase 3")
    (REPORT_DIR / "phase3_preservation_checks.json").write_text(
        json.dumps(preservation, indent=2), encoding="utf-8"
    )
    phase3_metrics = {
        "primary_model": "Phase 2 XGBoost",
        "phase2_model_metrics_at_threshold_0_5": phase2_metrics["XGBoost"],
        "comparison_ann_recall_at_threshold_0_5": phase2_metrics["ANN"]["recall"],
        "shap_sample_size": SHAP_SAMPLE_SIZE,
        "shap_max_additivity_error_raw_margin": shap_additivity_error,
        "calibration_fit_source": "3-fold OOF predictions from Phase 1 training IDs only",
        "selected_calibration": selected_calibration,
        "calibration_results": calibration.to_dict(orient="records"),
        "risk_band_method": band_metadata,
        "hypothetical_cost_assumptions": {
            "false_positive_cost_units": 1,
            "false_negative_cost_units_tested": list(FN_FP_RATIOS),
            "is_real_home_credit_economics": False,
        },
        "cost_optima": cost_table.to_dict(orient="records"),
        "preservation_checks": preservation,
    }
    (METRIC_DIR / "phase3_metrics.json").write_text(
        json.dumps(phase3_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(phase3_metrics, indent=2))


if __name__ == "__main__":
    main()
