"""Retrain three models on application plus relational features using Phase 1 IDs."""

from __future__ import annotations

import gc
import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import sparse
from sklearn.linear_model import LogisticRegression
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
from src.train_baselines import build_preprocessor, evaluate

SOURCE_PREFIXES = {
    "Bureau + bureau balance": "BUREAU_",
    "Previous applications": "PREV_",
    "Installment payments": "INST_",
    "POS cash balance": "POS_",
    "Credit card balance": "CC_",
}


def set_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def make_xgboost(
    scale_pos_weight: float,
    *,
    n_estimators: int = 2_000,
    early_stopping_rounds: int = 75,
) -> XGBClassifier:
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
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_and_validate_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    application = pd.read_csv(RAW_DIR / "application_train.csv", low_memory=False)
    relational = pd.read_parquet(PROCESSED_DIR / "phase2_relational_features.parquet")
    train_manifest = pd.read_csv(PROCESSED_DIR / "train_split_ids.csv")
    valid_manifest = pd.read_csv(PROCESSED_DIR / "validation_split_ids.csv")

    application_ids = set(application[ID_COLUMN])
    train_ids = train_manifest[ID_COLUMN].tolist()
    valid_ids = valid_manifest[ID_COLUMN].tolist()
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    checks = {
        "application_rows_before_join": len(application),
        "relational_rows": len(relational),
        "relational_unique_ids": int(relational[ID_COLUMN].nunique()),
        "relational_duplicate_ids": int(relational[ID_COLUMN].duplicated().sum()),
        "target_absent_from_relational_features": TARGET not in relational.columns,
        "train_validation_overlap": len(train_set & valid_set),
        "split_union_matches_application_ids": train_set | valid_set == application_ids,
        "train_ids_match_phase1_count": len(train_ids) == 246_008,
        "validation_ids_match_phase1_count": len(valid_ids) == 61_503,
        "manifest_targets_match_application": True,
    }
    if checks["relational_duplicate_ids"] or not checks["target_absent_from_relational_features"]:
        raise ValueError(f"Relational grain/leakage failure: {checks}")
    if checks["train_validation_overlap"] or not checks["split_union_matches_application_ids"]:
        raise ValueError(f"Phase 1 split identity failure: {checks}")

    merged = application.merge(relational, on=ID_COLUMN, how="left", validate="one_to_one")
    checks["rows_after_join"] = len(merged)
    checks["join_preserved_application_rows"] = len(merged) == len(application)
    checks["relational_feature_count"] = relational.shape[1] - 1
    checks["application_feature_count"] = application.shape[1] - 2
    checks["enriched_raw_feature_count"] = merged.shape[1] - 2

    merged = merged.set_index(ID_COLUMN, verify_integrity=True)
    y = merged.pop(TARGET).astype(np.int8)
    relational_columns = [column for column in relational if column != ID_COLUMN]
    merged[relational_columns] = (
        merged[relational_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    checks["nonfinite_relational_values_after_cleaning"] = int(
        (~np.isfinite(merged[relational_columns].to_numpy(dtype=np.float64))).sum()
    )
    manifest_targets = pd.concat([train_manifest, valid_manifest]).set_index(ID_COLUMN)[TARGET]
    checks["manifest_targets_match_application"] = bool(
        y.sort_index().equals(manifest_targets.astype(np.int8).sort_index())
    )
    required_true = (
        "target_absent_from_relational_features",
        "split_union_matches_application_ids",
        "join_preserved_application_rows",
        "manifest_targets_match_application",
    )
    if not all(checks[name] for name in required_true):
        raise ValueError(f"Data quality/leakage check failed: {checks}")
    if checks["nonfinite_relational_values_after_cleaning"]:
        raise ValueError(f"Non-finite relational values remain: {checks}")

    X_train = merged.loc[train_ids].copy()
    X_valid = merged.loc[valid_ids].copy()
    y_train = y.loc[train_ids].copy()
    y_valid = y.loc[valid_ids].copy()
    checks["train_positive_count"] = int(y_train.sum())
    checks["validation_positive_count"] = int(y_valid.sum())
    checks["train_id_order_matches_phase1_manifest"] = X_train.index.tolist() == train_ids
    checks["validation_id_order_matches_phase1_manifest"] = X_valid.index.tolist() == valid_ids
    (REPORT_DIR / "phase2_leakage_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return X_train, y_train, X_valid, y_valid, checks


def plot_confusion_matrices(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4))
    for axis, (name, metrics) in zip(axes, results.items()):
        cm = metrics["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=axis)
        axis.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "phase2_confusion_matrices.png", dpi=160)
    plt.close(fig)


def run_ablation(
    X_train: sparse.csr_matrix,
    y_train: pd.Series,
    X_valid: sparse.csr_matrix,
    y_valid: pd.Series,
    transformed_names: np.ndarray,
    scale_pos_weight: float,
    phase1_xgb: dict,
    phase2_xgb: dict,
) -> pd.DataFrame:
    relational_prefixes = tuple(f"num__{prefix}" for prefix in SOURCE_PREFIXES.values())
    app_mask = np.array(
        [not name.startswith(relational_prefixes) for name in transformed_names], dtype=bool
    )
    records = [
        {
            "configuration": "Application only (Phase 1 recorded)",
            "added_source": "None",
            "transformed_feature_count": int(app_mask.sum()),
            "roc_auc": phase1_xgb["roc_auc"],
            "pr_auc": phase1_xgb["pr_auc"],
            "delta_roc_auc_vs_phase1": 0.0,
            "delta_pr_auc_vs_phase1": 0.0,
            "best_iteration": 961,
        }
    ]
    for source_name, prefix in SOURCE_PREFIXES.items():
        source_mask = np.array(
            [name.startswith(f"num__{prefix}") for name in transformed_names], dtype=bool
        )
        selected = app_mask | source_mask
        print(f"Ablation: application + {source_name} ({selected.sum()} features)", flush=True)
        model = make_xgboost(scale_pos_weight, n_estimators=1_200, early_stopping_rounds=50)
        model.fit(
            X_train[:, selected],
            y_train,
            eval_set=[(X_valid[:, selected], y_valid)],
            verbose=False,
        )
        probabilities = model.predict_proba(X_valid[:, selected])[:, 1]
        metrics = evaluate(y_valid, probabilities)
        records.append(
            {
                "configuration": f"Application + {source_name}",
                "added_source": source_name,
                "transformed_feature_count": int(selected.sum()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "delta_roc_auc_vs_phase1": metrics["roc_auc"] - phase1_xgb["roc_auc"],
                "delta_pr_auc_vs_phase1": metrics["pr_auc"] - phase1_xgb["pr_auc"],
                "best_iteration": int(model.best_iteration),
            }
        )
        del model, probabilities
        gc.collect()
    records.append(
        {
            "configuration": "Application + all relational sources (Phase 2)",
            "added_source": "All",
            "transformed_feature_count": len(transformed_names),
            "roc_auc": phase2_xgb["roc_auc"],
            "pr_auc": phase2_xgb["pr_auc"],
            "delta_roc_auc_vs_phase1": phase2_xgb["roc_auc"] - phase1_xgb["roc_auc"],
            "delta_pr_auc_vs_phase1": phase2_xgb["pr_auc"] - phase1_xgb["pr_auc"],
            "best_iteration": phase2_xgb["best_iteration"],
        }
    )
    return pd.DataFrame(records)


def main() -> None:
    ensure_output_dirs()
    set_seeds()
    X_train, y_train, X_valid, y_valid, checks = load_and_validate_data()
    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = build_preprocessor(numerical, categorical)
    print("Fitting train-only preprocessing", flush=True)
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_valid_t = preprocessor.transform(X_valid).astype(np.float32)
    if not sparse.issparse(X_train_t):
        X_train_t = sparse.csr_matrix(X_train_t)
        X_valid_t = sparse.csr_matrix(X_valid_t)
    else:
        X_train_t = X_train_t.tocsr()
        X_valid_t = X_valid_t.tocsr()
    transformed_names = preprocessor.get_feature_names_out()
    joblib.dump(preprocessor, MODEL_DIR / "phase2_preprocessor.joblib", compress=3)
    checks["phase1_transformed_feature_count"] = 241
    checks["phase2_transformed_feature_count"] = int(X_train_t.shape[1])
    del X_train, X_valid
    gc.collect()

    results: dict[str, dict] = {}
    print("Training Phase 2 Logistic Regression", flush=True)
    logistic = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    logistic.fit(X_train_t, y_train)
    logistic_prob = logistic.predict_proba(X_valid_t)[:, 1]
    results["Logistic Regression"] = evaluate(y_valid, logistic_prob)
    joblib.dump(logistic, MODEL_DIR / "phase2_logistic_regression.joblib", compress=3)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives
    print("Training Phase 2 XGBoost", flush=True)
    xgb = make_xgboost(scale_pos_weight)
    xgb.fit(X_train_t, y_train, eval_set=[(X_valid_t, y_valid)], verbose=False)
    xgb_prob = xgb.predict_proba(X_valid_t)[:, 1]
    results["XGBoost"] = evaluate(y_valid, xgb_prob)
    results["XGBoost"]["best_iteration"] = int(xgb.best_iteration)
    xgb.save_model(MODEL_DIR / "phase2_xgboost.json")

    with (METRIC_DIR / "baseline_metrics.json").open(encoding="utf-8") as handle:
        phase1_metrics = json.load(handle)["metrics_at_threshold_0_5"]
    ablation = run_ablation(
        X_train_t,
        y_train,
        X_valid_t,
        y_valid,
        transformed_names,
        scale_pos_weight,
        phase1_metrics["XGBoost"],
        {**results["XGBoost"], "best_iteration": int(xgb.best_iteration)},
    )
    ablation.to_csv(REPORT_DIR / "phase2_ablation.csv", index=False)

    print("Training Phase 2 ANN", flush=True)
    inputs = tf.keras.Input(shape=(X_train_t.shape[1],), sparse=True, name="features")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="default_probability")(x)
    ann = tf.keras.Model(inputs=inputs, outputs=outputs)
    ann.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_roc_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    history = ann.fit(
        X_train_t,
        y_train.to_numpy(),
        validation_data=(X_valid_t, y_valid.to_numpy()),
        epochs=40,
        batch_size=2_048,
        class_weight={0: 1.0, 1: scale_pos_weight},
        callbacks=callbacks,
        verbose=2,
    )
    ann_prob = ann.predict(X_valid_t, batch_size=4_096, verbose=0).ravel()
    results["ANN"] = evaluate(y_valid, ann_prob)
    results["ANN"]["epochs_trained"] = len(history.history["loss"])
    ann.save(MODEL_DIR / "phase2_ann.keras")

    predictions = pd.DataFrame(
        {
            ID_COLUMN: y_valid.index.to_numpy(),
            TARGET: y_valid.to_numpy(),
            "logistic_probability": logistic_prob,
            "xgboost_probability": xgb_prob,
            "ann_probability": ann_prob,
        }
    )
    predictions.to_csv(PROCESSED_DIR / "phase2_validation_predictions.csv", index=False)

    comparisons = []
    for model_name, phase2 in results.items():
        phase1 = phase1_metrics[model_name]
        row = {"model": model_name}
        for metric in ("roc_auc", "pr_auc", "recall", "precision", "f1"):
            row[f"phase1_{metric}"] = phase1[metric]
            row[f"phase2_{metric}"] = phase2[metric]
            row[f"delta_{metric}"] = phase2[metric] - phase1[metric]
        comparisons.append(row)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(REPORT_DIR / "phase2_vs_phase1.csv", index=False)
    pd.DataFrame(
        {
            name: {key: value for key, value in metrics.items() if not isinstance(value, dict)}
            for name, metrics in results.items()
        }
    ).T.to_csv(REPORT_DIR / "phase2_model_comparison.csv")
    plot_confusion_matrices(results)

    metadata = {
        "random_state": RANDOM_STATE,
        "split_source": "Phase 1 train_split_ids.csv and validation_split_ids.csv",
        "train_rows": len(y_train),
        "validation_rows": len(y_valid),
        "application_raw_feature_count": checks["application_feature_count"],
        "relational_feature_count": checks["relational_feature_count"],
        "enriched_raw_feature_count": checks["enriched_raw_feature_count"],
        "phase1_transformed_feature_count": 241,
        "phase2_transformed_feature_count": int(X_train_t.shape[1]),
        "class_weight_positive": scale_pos_weight,
        "metrics_at_threshold_0_5": results,
    }
    (METRIC_DIR / "phase2_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
