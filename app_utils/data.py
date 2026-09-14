from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "app" / "demo_artifacts"


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((DEMO_DIR / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DEMO_DIR / name)


def demo_figure(name: str) -> Path:
    return DEMO_DIR / "figures" / name

