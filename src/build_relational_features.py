"""Build memory-conscious, one-row-per-applicant relational features with DuckDB."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from src.config import ID_COLUMN, PROCESSED_DIR, RAW_DIR, REPORT_DIR, ensure_output_dirs

SOURCE_OUTPUTS = {
    "bureau": "phase2_bureau_features.parquet",
    "previous_application": "phase2_previous_features.parquet",
    "installments_payments": "phase2_installments_features.parquet",
    "POS_CASH_balance": "phase2_pos_features.parquet",
    "credit_card_balance": "phase2_credit_card_features.parquet",
}


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def csv(path: Path) -> str:
    return f"read_csv_auto('{sql_path(path)}', header=true, sample_size=200000)"


def copy_query(connection: duckdb.DuckDBPyConnection, query: str, output: Path) -> None:
    output_sql = sql_path(output)
    connection.execute(
        f"COPY ({query}) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def build_bureau(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    bureau = csv(RAW_DIR / "bureau.csv")
    balance = csv(RAW_DIR / "bureau_balance.csv")
    query = f"""
    WITH bb_by_loan AS (
        SELECT
            SK_ID_BUREAU,
            COUNT(*) AS BB_MONTH_COUNT,
            MIN(MONTHS_BALANCE) AS BB_OLDEST_MONTH,
            SUM(CASE WHEN STATUS = '0' THEN 1 ELSE 0 END) AS BB_CURRENT_MONTHS,
            SUM(CASE WHEN STATUS IN ('1', '2', '3', '4', '5') THEN 1 ELSE 0 END) AS BB_DELINQUENT_MONTHS,
            SUM(CASE WHEN STATUS = '5' THEN 1 ELSE 0 END) AS BB_SEVERE_MONTHS,
            SUM(CASE WHEN STATUS = 'C' THEN 1 ELSE 0 END) AS BB_CLOSED_MONTHS,
            SUM(CASE WHEN STATUS = 'X' THEN 1 ELSE 0 END) AS BB_UNKNOWN_MONTHS,
            MAX(TRY_CAST(STATUS AS INTEGER)) AS BB_MAX_DELINQUENCY_STATUS
        FROM {balance}
        GROUP BY SK_ID_BUREAU
    )
    SELECT
        b.SK_ID_CURR,
        COUNT(*) AS BUREAU_LOAN_COUNT,
        COUNT(DISTINCT b.CREDIT_TYPE) AS BUREAU_CREDIT_TYPE_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS BUREAU_ACTIVE_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS BUREAU_CLOSED_COUNT,
        SUM(CASE WHEN b.CREDIT_ACTIVE = 'Sold' THEN 1 ELSE 0 END) AS BUREAU_SOLD_COUNT,
        AVG(CASE WHEN b.CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END) AS BUREAU_ACTIVE_RATIO,
        AVG(CASE WHEN b.CREDIT_ACTIVE = 'Closed' THEN 1.0 ELSE 0.0 END) AS BUREAU_CLOSED_RATIO,
        MIN(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MIN,
        MAX(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MAX,
        AVG(b.DAYS_CREDIT) AS BUREAU_DAYS_CREDIT_MEAN,
        MAX(b.CREDIT_DAY_OVERDUE) AS BUREAU_DAYS_OVERDUE_MAX,
        AVG(b.CREDIT_DAY_OVERDUE) AS BUREAU_DAYS_OVERDUE_MEAN,
        AVG(CASE WHEN b.CREDIT_DAY_OVERDUE > 0 THEN 1.0 ELSE 0.0 END) AS BUREAU_OVERDUE_LOAN_RATIO,
        SUM(COALESCE(b.AMT_CREDIT_SUM, 0)) AS BUREAU_CREDIT_SUM,
        AVG(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_MEAN,
        MAX(b.AMT_CREDIT_SUM) AS BUREAU_CREDIT_MAX,
        SUM(COALESCE(b.AMT_CREDIT_SUM_DEBT, 0)) AS BUREAU_DEBT_SUM,
        AVG(b.AMT_CREDIT_SUM_DEBT) AS BUREAU_DEBT_MEAN,
        SUM(COALESCE(b.AMT_CREDIT_SUM_LIMIT, 0)) AS BUREAU_LIMIT_SUM,
        SUM(COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0)) AS BUREAU_OVERDUE_AMOUNT_SUM,
        MAX(COALESCE(b.AMT_CREDIT_SUM_OVERDUE, 0)) AS BUREAU_OVERDUE_AMOUNT_MAX,
        SUM(COALESCE(b.AMT_CREDIT_SUM_DEBT, 0)) /
            NULLIF(SUM(COALESCE(b.AMT_CREDIT_SUM, 0)), 0) AS BUREAU_DEBT_TO_CREDIT_RATIO,
        SUM(COALESCE(b.CNT_CREDIT_PROLONG, 0)) AS BUREAU_PROLONG_SUM,
        SUM(COALESCE(b.AMT_ANNUITY, 0)) AS BUREAU_ANNUITY_SUM,
        AVG(b.DAYS_CREDIT_UPDATE) AS BUREAU_DAYS_UPDATE_MEAN,
        SUM(COALESCE(bb.BB_MONTH_COUNT, 0)) AS BUREAU_BALANCE_MONTH_COUNT,
        MIN(bb.BB_OLDEST_MONTH) AS BUREAU_BALANCE_OLDEST_MONTH,
        SUM(COALESCE(bb.BB_DELINQUENT_MONTHS, 0)) AS BUREAU_BALANCE_DELINQUENT_MONTHS,
        SUM(COALESCE(bb.BB_SEVERE_MONTHS, 0)) AS BUREAU_BALANCE_SEVERE_MONTHS,
        SUM(COALESCE(bb.BB_CLOSED_MONTHS, 0)) AS BUREAU_BALANCE_CLOSED_MONTHS,
        SUM(COALESCE(bb.BB_UNKNOWN_MONTHS, 0)) AS BUREAU_BALANCE_UNKNOWN_MONTHS,
        MAX(bb.BB_MAX_DELINQUENCY_STATUS) AS BUREAU_BALANCE_MAX_DELINQUENCY,
        SUM(COALESCE(bb.BB_DELINQUENT_MONTHS, 0)) /
            NULLIF(SUM(COALESCE(bb.BB_MONTH_COUNT, 0)), 0) AS BUREAU_BALANCE_DELINQUENT_RATIO,
        SUM(COALESCE(bb.BB_SEVERE_MONTHS, 0)) /
            NULLIF(SUM(COALESCE(bb.BB_MONTH_COUNT, 0)), 0) AS BUREAU_BALANCE_SEVERE_RATIO
    FROM {bureau} b
    LEFT JOIN bb_by_loan bb USING (SK_ID_BUREAU)
    GROUP BY b.SK_ID_CURR
    """
    copy_query(connection, query, output)


def build_previous(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    previous = csv(RAW_DIR / "previous_application.csv")
    query = f"""
    SELECT
        SK_ID_CURR,
        COUNT(*) AS PREV_APPLICATION_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS PREV_CONTRACT_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_APPROVED_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS PREV_REFUSED_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS PREV_CANCELED_COUNT,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END) AS PREV_APPROVED_RATIO,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END) AS PREV_REFUSED_RATIO,
        AVG(CASE WHEN NAME_CLIENT_TYPE = 'Repeater' THEN 1.0 ELSE 0.0 END) AS PREV_REPEATER_RATIO,
        AVG(AMT_APPLICATION) AS PREV_APPLICATION_AMOUNT_MEAN,
        SUM(COALESCE(AMT_APPLICATION, 0)) AS PREV_APPLICATION_AMOUNT_SUM,
        AVG(AMT_CREDIT) AS PREV_CREDIT_AMOUNT_MEAN,
        SUM(COALESCE(AMT_CREDIT, 0)) AS PREV_CREDIT_AMOUNT_SUM,
        AVG(AMT_ANNUITY) AS PREV_ANNUITY_MEAN,
        SUM(COALESCE(AMT_ANNUITY, 0)) AS PREV_ANNUITY_SUM,
        AVG(AMT_DOWN_PAYMENT) AS PREV_DOWN_PAYMENT_MEAN,
        AVG(RATE_DOWN_PAYMENT) AS PREV_DOWN_PAYMENT_RATE_MEAN,
        AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0)) AS PREV_CREDIT_TO_APPLICATION_MEAN,
        SUM(COALESCE(AMT_CREDIT, 0)) /
            NULLIF(SUM(COALESCE(AMT_APPLICATION, 0)), 0) AS PREV_CREDIT_TO_APPLICATION_TOTAL,
        AVG(CNT_PAYMENT) AS PREV_TERM_MEAN,
        MAX(CNT_PAYMENT) AS PREV_TERM_MAX,
        MIN(DAYS_DECISION) AS PREV_DAYS_DECISION_MIN,
        MAX(DAYS_DECISION) AS PREV_DAYS_DECISION_MAX,
        AVG(DAYS_DECISION) AS PREV_DAYS_DECISION_MEAN,
        AVG(CASE WHEN NFLAG_INSURED_ON_APPROVAL = 1 THEN 1.0 ELSE 0.0 END) AS PREV_INSURED_RATIO
    FROM {previous}
    GROUP BY SK_ID_CURR
    """
    copy_query(connection, query, output)


def build_installments(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    installments = csv(RAW_DIR / "installments_payments.csv")
    query = f"""
    SELECT
        SK_ID_CURR,
        COUNT(*) AS INST_PAYMENT_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS INST_CONTRACT_COUNT,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS INST_LATE_COUNT,
        AVG(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END) AS INST_LATE_RATIO,
        AVG(GREATEST(COALESCE(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0), 0)) AS INST_DPD_MEAN,
        MAX(GREATEST(COALESCE(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0), 0)) AS INST_DPD_MAX,
        SUM(GREATEST(COALESCE(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0), 0)) AS INST_DPD_SUM,
        SUM(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END) AS INST_UNDERPAY_COUNT,
        AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) AS INST_UNDERPAY_RATIO,
        SUM(GREATEST(COALESCE(AMT_INSTALMENT - AMT_PAYMENT, 0), 0)) AS INST_UNDERPAY_AMOUNT_SUM,
        MAX(GREATEST(COALESCE(AMT_INSTALMENT - AMT_PAYMENT, 0), 0)) AS INST_UNDERPAY_AMOUNT_MAX,
        SUM(COALESCE(AMT_PAYMENT, 0)) AS INST_PAYMENT_AMOUNT_SUM,
        SUM(COALESCE(AMT_INSTALMENT, 0)) AS INST_DUE_AMOUNT_SUM,
        SUM(COALESCE(AMT_PAYMENT, 0)) /
            NULLIF(SUM(COALESCE(AMT_INSTALMENT, 0)), 0) AS INST_PAYMENT_TO_DUE_RATIO,
        AVG(AMT_PAYMENT) AS INST_PAYMENT_AMOUNT_MEAN,
        AVG(AMT_INSTALMENT) AS INST_DUE_AMOUNT_MEAN
    FROM {installments}
    GROUP BY SK_ID_CURR
    """
    copy_query(connection, query, output)


def build_pos(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    pos = csv(RAW_DIR / "POS_CASH_balance.csv")
    query = f"""
    SELECT
        SK_ID_CURR,
        COUNT(*) AS POS_RECORD_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS POS_CONTRACT_COUNT,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1.0 ELSE 0.0 END) AS POS_ACTIVE_RATIO,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1.0 ELSE 0.0 END) AS POS_COMPLETED_RATIO,
        AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END) AS POS_DELINQUENT_RATIO,
        AVG(SK_DPD) AS POS_DPD_MEAN,
        MAX(SK_DPD) AS POS_DPD_MAX,
        AVG(SK_DPD_DEF) AS POS_DPD_DEF_MEAN,
        MAX(SK_DPD_DEF) AS POS_DPD_DEF_MAX,
        AVG(CNT_INSTALMENT_FUTURE) AS POS_FUTURE_INSTALMENTS_MEAN,
        MAX(CNT_INSTALMENT_FUTURE) AS POS_FUTURE_INSTALMENTS_MAX,
        MIN(MONTHS_BALANCE) AS POS_OLDEST_MONTH,
        MAX(MONTHS_BALANCE) AS POS_LATEST_MONTH
    FROM {pos}
    GROUP BY SK_ID_CURR
    """
    copy_query(connection, query, output)


def build_credit_card(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    card = csv(RAW_DIR / "credit_card_balance.csv")
    query = f"""
    SELECT
        SK_ID_CURR,
        COUNT(*) AS CC_RECORD_COUNT,
        COUNT(DISTINCT SK_ID_PREV) AS CC_CONTRACT_COUNT,
        AVG(AMT_BALANCE) AS CC_BALANCE_MEAN,
        MAX(AMT_BALANCE) AS CC_BALANCE_MAX,
        ARG_MAX(AMT_BALANCE, MONTHS_BALANCE) AS CC_BALANCE_LATEST,
        AVG(AMT_CREDIT_LIMIT_ACTUAL) AS CC_LIMIT_MEAN,
        AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILIZATION_MEAN,
        MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS CC_UTILIZATION_MAX,
        SUM(COALESCE(AMT_BALANCE, 0)) /
            NULLIF(SUM(COALESCE(AMT_CREDIT_LIMIT_ACTUAL, 0)), 0) AS CC_UTILIZATION_TOTAL,
        SUM(COALESCE(AMT_DRAWINGS_CURRENT, 0)) AS CC_DRAWINGS_SUM,
        SUM(COALESCE(AMT_DRAWINGS_ATM_CURRENT, 0)) AS CC_ATM_DRAWINGS_SUM,
        SUM(COALESCE(AMT_PAYMENT_CURRENT, 0)) AS CC_PAYMENT_CURRENT_SUM,
        SUM(COALESCE(AMT_PAYMENT_TOTAL_CURRENT, 0)) AS CC_PAYMENT_TOTAL_SUM,
        SUM(COALESCE(AMT_INST_MIN_REGULARITY, 0)) AS CC_MIN_PAYMENT_SUM,
        SUM(COALESCE(AMT_PAYMENT_TOTAL_CURRENT, 0)) /
            NULLIF(SUM(COALESCE(AMT_INST_MIN_REGULARITY, 0)), 0) AS CC_PAYMENT_TO_MIN_RATIO,
        SUM(GREATEST(COALESCE(AMT_INST_MIN_REGULARITY - AMT_PAYMENT_CURRENT, 0), 0))
            AS CC_MIN_PAYMENT_SHORTFALL_SUM,
        AVG(CASE WHEN AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY THEN 1.0 ELSE 0.0 END)
            AS CC_MIN_PAYMENT_SHORTFALL_RATIO,
        SUM(COALESCE(CNT_DRAWINGS_CURRENT, 0)) AS CC_DRAWING_COUNT_SUM,
        AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END) AS CC_DELINQUENT_RATIO,
        AVG(SK_DPD) AS CC_DPD_MEAN,
        MAX(SK_DPD) AS CC_DPD_MAX,
        AVG(SK_DPD_DEF) AS CC_DPD_DEF_MEAN,
        MAX(SK_DPD_DEF) AS CC_DPD_DEF_MAX,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1.0 ELSE 0.0 END) AS CC_ACTIVE_RATIO
    FROM {card}
    GROUP BY SK_ID_CURR
    """
    copy_query(connection, query, output)


def build_joined_features(connection: duckdb.DuckDBPyConnection, output: Path) -> None:
    application = csv(RAW_DIR / "application_train.csv")
    paths = {name: sql_path(PROCESSED_DIR / file) for name, file in SOURCE_OUTPUTS.items()}
    query = f"""
    SELECT
        app.SK_ID_CURR,
        CASE WHEN b.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS BUREAU_HAS_HISTORY,
        b.* EXCLUDE (SK_ID_CURR),
        CASE WHEN p.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS PREV_HAS_HISTORY,
        p.* EXCLUDE (SK_ID_CURR),
        CASE WHEN i.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS INST_HAS_HISTORY,
        i.* EXCLUDE (SK_ID_CURR),
        CASE WHEN pos.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS POS_HAS_HISTORY,
        pos.* EXCLUDE (SK_ID_CURR),
        CASE WHEN cc.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS CC_HAS_HISTORY,
        cc.* EXCLUDE (SK_ID_CURR)
    FROM (SELECT SK_ID_CURR FROM {application}) app
    LEFT JOIN read_parquet('{paths['bureau']}') b USING (SK_ID_CURR)
    LEFT JOIN read_parquet('{paths['previous_application']}') p USING (SK_ID_CURR)
    LEFT JOIN read_parquet('{paths['installments_payments']}') i USING (SK_ID_CURR)
    LEFT JOIN read_parquet('{paths['POS_CASH_balance']}') pos USING (SK_ID_CURR)
    LEFT JOIN read_parquet('{paths['credit_card_balance']}') cc USING (SK_ID_CURR)
    """
    copy_query(connection, query, output)


def quality_checks(output: Path) -> dict[str, object]:
    frame = pd.read_parquet(output)
    source_counts = {}
    for prefix in ("BUREAU", "PREV", "INST", "POS", "CC"):
        source_counts[prefix] = int(frame[f"{prefix}_HAS_HISTORY"].sum())
    feature_columns = [column for column in frame if column != ID_COLUMN]
    checks = {
        "rows": len(frame),
        "columns_including_id": int(frame.shape[1]),
        "relational_feature_count": len(feature_columns),
        "unique_applicant_ids": int(frame[ID_COLUMN].nunique()),
        "duplicate_applicant_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "target_present": "TARGET" in frame.columns,
        "duplicate_feature_names": len(feature_columns) - len(set(feature_columns)),
        "source_applicants_with_history": source_counts,
        "null_counts_by_source": {
            prefix: int(frame.filter(regex=f"^{prefix}_").isna().sum().sum())
            for prefix in ("BUREAU", "PREV", "INST", "POS", "CC")
        },
    }
    if checks["duplicate_applicant_ids"] != 0 or checks["target_present"]:
        raise ValueError(f"Relational leakage/grain check failed: {checks}")
    (REPORT_DIR / "phase2_feature_quality.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature": feature_columns,
            "source": [column.split("_", 1)[0] for column in feature_columns],
            "dtype": [str(frame[column].dtype) for column in feature_columns],
            "missing_count": [int(frame[column].isna().sum()) for column in feature_columns],
        }
    ).to_csv(REPORT_DIR / "phase2_feature_dictionary.csv", index=False)
    return checks


def main() -> None:
    ensure_output_dirs()
    temp_dir = PROCESSED_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '4GB'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    builders = {
        "bureau": build_bureau,
        "previous_application": build_previous,
        "installments_payments": build_installments,
        "POS_CASH_balance": build_pos,
        "credit_card_balance": build_credit_card,
    }
    try:
        for source, builder in builders.items():
            output = PROCESSED_DIR / SOURCE_OUTPUTS[source]
            print(f"Building {source} -> {output.name}", flush=True)
            builder(connection, output)
        joined = PROCESSED_DIR / "phase2_relational_features.parquet"
        print(f"Joining one-row-per-applicant features -> {joined.name}", flush=True)
        build_joined_features(connection, joined)
    finally:
        connection.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
    checks = quality_checks(joined)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
