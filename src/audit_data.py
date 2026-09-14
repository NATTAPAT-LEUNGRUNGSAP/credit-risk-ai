"""Inventory all Home Credit CSVs and deeply audit application_train.csv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURE_DIR, ID_COLUMN, RAW_DIR, REPORT_DIR, TARGET, ensure_output_dirs


def count_csv_rows(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Count records without loading multi-GB tables into memory."""
    newlines = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            newlines += block.count(b"\n")
            last = block[-1:]
    physical_lines = newlines if last == b"\n" else newlines + 1
    return max(physical_lines - 1, 0)


def inventory_csvs() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    schema_records: list[dict[str, object]] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        try:
            sample = pd.read_csv(path, nrows=2_000, low_memory=False)
            encoding = "utf-8"
        except UnicodeDecodeError:
            sample = pd.read_csv(path, nrows=2_000, low_memory=False, encoding="cp1252")
            encoding = "cp1252"
        rows = count_csv_rows(path)
        records.append(
            {
                "file": path.name,
                "rows": rows,
                "columns": sample.shape[1],
                "size_mb": round(path.stat().st_size / 1024**2, 2),
                "encoding": encoding,
            }
        )
        for column, dtype in sample.dtypes.items():
            schema_records.append(
                {
                    "file": path.name,
                    "column": column,
                    "sample_inferred_dtype": str(dtype),
                }
            )
    inventory = pd.DataFrame(records)
    inventory.to_csv(REPORT_DIR / "csv_inventory.csv", index=False)
    pd.DataFrame(schema_records).to_csv(REPORT_DIR / "csv_schema.csv", index=False)
    return inventory


def audit_application_train() -> dict[str, object]:
    path = RAW_DIR / "application_train.csv"
    frame = pd.read_csv(path, low_memory=False)
    numerical = frame.drop(columns=[TARGET]).select_dtypes(include=np.number).columns.tolist()
    categorical = frame.select_dtypes(exclude=np.number).columns.tolist()
    missing = (
        frame.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: 100 * x["missing_count"] / len(frame))
        .sort_values("missing_pct", ascending=False)
    )
    missing.to_csv(REPORT_DIR / "application_train_missing_values.csv")

    target_counts = frame[TARGET].value_counts().sort_index()
    target_rates = frame[TARGET].value_counts(normalize=True).sort_index()
    duplicate_ids = int(frame[ID_COLUMN].duplicated().sum())
    duplicate_rows = int(frame.duplicated().sum())
    numeric_values = frame[numerical].to_numpy(copy=False)
    infinite_values = int(np.isinf(numeric_values).sum())
    constant_columns = [column for column in frame if frame[column].nunique(dropna=False) <= 1]

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x=TARGET, ax=axis, color="#2563EB")
    axis.set_title("Application train target distribution")
    for patch in axis.patches:
        axis.annotate(
            f"{int(patch.get_height()):,}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    top_missing = missing.head(25).sort_values("missing_pct")
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.barh(top_missing.index, top_missing["missing_pct"], color="#D97706")
    axis.set_xlabel("Missing values (%)")
    axis.set_title("Top 25 columns by missingness")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_missing_values.png", dpi=160)
    plt.close(fig)

    audit = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_rates": {str(k): float(v) for k, v in target_rates.items()},
        "numerical_features_excluding_target": len(numerical),
        "categorical_features": len(categorical),
        "columns_with_missing_values": int((missing["missing_count"] > 0).sum()),
        "columns_over_50pct_missing": int((missing["missing_pct"] > 50).sum()),
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "infinite_numeric_values": infinite_values,
        "constant_columns": constant_columns,
    }
    (REPORT_DIR / "application_train_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def write_report(inventory: pd.DataFrame, audit: dict[str, object]) -> None:
    target_counts = audit["target_counts"]
    target_rates = audit["target_rates"]
    lines = [
        "# Phase 1 Data Audit",
        "",
        "## CSV inventory",
        "",
        "| " + " | ".join(inventory.columns) + " |",
        "| " + " | ".join(["---"] * len(inventory.columns)) + " |",
        *[
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in inventory.itertuples(index=False, name=None)
        ],
        "",
        "## application_train.csv",
        "",
        f"- Shape: {audit['rows']:,} rows x {audit['columns']} columns",
        f"- Repaid (TARGET=0): {target_counts['0']:,} ({target_rates['0']:.2%})",
        f"- Default/payment difficulty (TARGET=1): {target_counts['1']:,} ({target_rates['1']:.2%})",
        f"- Numerical features (ID included, target excluded): {audit['numerical_features_excluding_target']}",
        f"- Categorical features: {audit['categorical_features']}",
        f"- Columns with missing values: {audit['columns_with_missing_values']}",
        f"- Columns over 50% missing: {audit['columns_over_50pct_missing']}",
        f"- Duplicate SK_ID_CURR: {audit['duplicate_ids']}",
        f"- Fully duplicate rows: {audit['duplicate_rows']}",
        f"- Infinite numeric values: {audit['infinite_numeric_values']}",
        f"- Constant columns: {audit['constant_columns']}",
        "",
        (
            "Detailed per-column schema and missingness are in `csv_schema.csv` and "
            "`application_train_missing_values.csv`."
        ),
    ]
    (REPORT_DIR / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    inventory = inventory_csvs()
    audit = audit_application_train()
    write_report(inventory, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
