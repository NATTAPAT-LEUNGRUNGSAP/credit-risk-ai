from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"data/raw", "data/processed", "models"}
FORBIDDEN_NAMES = {".env", "secrets.toml"}
FORBIDDEN_SUFFIXES = {".ckpt", ".h5", ".joblib", ".keras", ".onnx", ".pb", ".pkl"}
SECRET_PATTERNS = [
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
]


def release_candidates() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
    )
    return [PROJECT_ROOT / item.decode() for item in output.split(b"\0") if item]


def test_release_candidates_exclude_private_and_large_artifacts() -> None:
    candidates = release_candidates()
    assert candidates

    for path in candidates:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        assert not any(relative == part or relative.startswith(f"{part}/") for part in FORBIDDEN_PARTS)
        assert path.name not in FORBIDDEN_NAMES
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES
        assert path.stat().st_size < 5_000_000


def test_release_text_contains_no_credentials_or_local_user_paths() -> None:
    for path in release_candidates():
        if path.suffix.lower() not in {".csv", ".json", ".md", ".py", ".toml", ".txt"}:
            continue
        content = path.read_bytes()
        assert b"C:\\Users\\" not in content
        assert all(pattern.search(content) is None for pattern in SECRET_PATTERNS)


def test_cloud_default_dependencies_are_minimal_and_analysis_is_optional() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    default_names = {dependency.split("=")[0] for dependency in config["project"]["dependencies"]}
    analysis = " ".join(config["project"]["optional-dependencies"]["analysis"])

    assert default_names == {"altair", "pandas", "streamlit"}
    assert "tensorflow" in analysis
    assert "xgboost" in analysis
    assert "shap" in analysis


def test_readme_has_release_sections_and_verified_metrics() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for section in [
        "# Credit Risk AI — Explainable Credit Default Risk Decision Support",
        "## Architecture and workflow",
        "## Methodology / Leakage prevention",
        "## Privacy / Public demo",
        "## Limitations",
        "## Repository structure",
    ]:
        assert section in readme
    for value in ["0.7866", "0.2882", "0.3070", "0.7247", "307,511", "8.07%"]:
        assert value in readme
