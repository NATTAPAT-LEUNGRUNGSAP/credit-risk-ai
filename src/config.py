from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def ensure_output_dirs() -> None:
    for path in (PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR, METRIC_DIR):
        path.mkdir(parents=True, exist_ok=True)

