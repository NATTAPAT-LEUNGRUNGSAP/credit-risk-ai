"""Train leakage-safe Logistic Regression, XGBoost and Keras ANN baselines."""

from __future__ import annotations

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
    VALIDATION_SIZE,
    ensure_output_dirs,
)


def set_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numerical), ("cat", categorical_pipe, categorical)],
        sparse_threshold=1.0,
    )


def plot_confusion_matrices(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4))
    for axis, (name, metrics) in zip(axes, results.items()):
        cm = metrics["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axis)
        axis.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "baseline_confusion_matrices.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    set_seeds()
    frame = pd.read_csv(RAW_DIR / "application_train.csv", low_memory=False)
    y = frame.pop(TARGET).astype(np.int8)
    ids = frame.pop(ID_COLUMN)

    # Split before any learned preprocessing. IDs remain only in the split manifest.
    X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
        frame,
        y,
        ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    pd.DataFrame({ID_COLUMN: id_train, TARGET: y_train, "split": "train"}).to_csv(
        PROCESSED_DIR / "train_split_ids.csv", index=False
    )
    pd.DataFrame({ID_COLUMN: id_valid, TARGET: y_valid, "split": "validation"}).to_csv(
        PROCESSED_DIR / "validation_split_ids.csv", index=False
    )

    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = build_preprocessor(numerical, categorical)
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_valid_t = preprocessor.transform(X_valid).astype(np.float32)
    if not sparse.issparse(X_train_t):
        X_train_t = sparse.csr_matrix(X_train_t)
        X_valid_t = sparse.csr_matrix(X_valid_t)
    joblib.dump(preprocessor, MODEL_DIR / "baseline_preprocessor.joblib", compress=3)

    results: dict[str, dict] = {}
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
    joblib.dump(logistic, MODEL_DIR / "logistic_regression.joblib", compress=3)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    xgb = XGBClassifier(
        n_estimators=2_000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=negatives / positives,
        early_stopping_rounds=75,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_t, y_train, eval_set=[(X_valid_t, y_valid)], verbose=False)
    xgb_prob = xgb.predict_proba(X_valid_t)[:, 1]
    results["XGBoost"] = evaluate(y_valid, xgb_prob)
    xgb.save_model(MODEL_DIR / "xgboost_baseline.json")

    class_weight = {0: 1.0, 1: negatives / positives}
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
        metrics=[tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
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
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    ann_prob = ann.predict(X_valid_t, batch_size=4_096, verbose=0).ravel()
    results["ANN"] = evaluate(y_valid, ann_prob)
    results["ANN"]["epochs_trained"] = len(history.history["loss"])
    ann.save(MODEL_DIR / "ann_baseline.keras")

    predictions = pd.DataFrame(
        {
            ID_COLUMN: id_valid.to_numpy(),
            TARGET: y_valid.to_numpy(),
            "logistic_probability": logistic_prob,
            "xgboost_probability": xgb_prob,
            "ann_probability": ann_prob,
        }
    )
    predictions.to_csv(PROCESSED_DIR / "baseline_validation_predictions.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "transformed_features": int(X_train_t.shape[1]),
        "class_weight_positive": float(negatives / positives),
        "xgboost_best_iteration": int(xgb.best_iteration),
        "metrics_at_threshold_0_5": results,
    }
    (METRIC_DIR / "baseline_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            name: {k: v for k, v in values.items() if not isinstance(v, dict)}
            for name, values in results.items()
        }
    ).T.to_csv(REPORT_DIR / "baseline_model_comparison.csv")
    plot_confusion_matrices(results)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
