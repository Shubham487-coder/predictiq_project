"""
Task 1: load_and_enrich
Calls build_features() from feature_engineering.py, optionally filters
to a simulation cutoff datetime (as_of), and saves to features_latest.csv.

Airflow integration:
  - as_of is read from dag_run.conf["as_of"] when run via DAG
  - PREDICTIQ_PROCESSED_DIR Variable overrides the output directory
  - Returns the output CSV path via XCom (automatic PythonOperator return)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.feature_engineering import build_features

_FALLBACK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _processed_dir() -> Path:
    try:
        from airflow.models import Variable
        p = Variable.get("PREDICTIQ_PROCESSED_DIR", default_var=None)
        if p:
            return Path(p)
    except Exception:
        pass
    return _FALLBACK_PROCESSED_DIR


def run(as_of: str | None = None, **context):
    """
    as_of: ISO datetime string e.g. '2015-08-01 06:00:00'
           If None, uses all available data.
           When called from Airflow, read from dag_run.conf.
    """
    # ── Read as_of from Airflow dag_run conf if available ────────────────────
    if as_of is None and context.get("dag_run"):
        as_of = context["dag_run"].conf.get("as_of")

    processed_dir = _processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load_and_enrich] as_of={as_of or 'ALL DATA'}")
    print(f"[load_and_enrich] Output dir: {processed_dir}")
    cache_path = processed_dir / "features_full_cache.csv"
    if cache_path.exists():
        print("[load_and_enrich] Loading from cache...")
        features = pd.read_csv(cache_path, parse_dates=["datetime"])
    else:
        print("[load_and_enrich] Building features from raw CSVs...")
        features, nan_counts, nan_rows, rows_after = build_features()
        features.to_csv(cache_path, index=False)
        print(f"[load_and_enrich] Cache saved: {len(features)} rows")

    # ── Filter to simulation cutoff if provided ───────────────────────────────
    if as_of:
        cutoff = pd.Timestamp(as_of)
        before = len(features)
        features = features[features["datetime"] <= cutoff].reset_index(drop=True)
        print(f"[load_and_enrich] Filtered {before} → {len(features)} rows (cutoff: {cutoff})")

    out_path = processed_dir / "features_latest.csv"
    features.to_csv(out_path, index=False)
    print(f"[load_and_enrich] Done. Saved {len(features)} rows to {out_path}")
    return str(out_path)   # ← XCom: downstream tasks pull this path


if __name__ == "__main__":
    run(as_of="2015-08-01 06:00:00")