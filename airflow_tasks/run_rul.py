"""
Task 5: run_rul
Reads rul_input.json, calls POST /rul/batch in chunks of BATCH_SIZE,
saves results to data/processed/rul_results.json.

Airflow integration:
  - XCom: pulls rul_input.json path from Task 3 (build_rul_input)
  - Airflow Connection 'fastapi_predictiq' provides host/port (falls back to localhost:8000)
  - Airflow Variable 'PREDICTIQ_BATCH_SIZE' controls batch size (default 100)
  - Returns rul_results.json path via XCom
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import httpx

PROJECT_ROOT            = Path(__file__).resolve().parents[1]
_FALLBACK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
_FALLBACK_API_BASE      = "http://localhost:8000/api/v1"
_FALLBACK_BATCH_SIZE    = 100
TIMEOUT                 = 120


def _get_config(context: dict) -> tuple[str, int, Path, Path]:
    """Returns (api_base, batch_size, input_path, output_path)."""
    api_base   = _FALLBACK_API_BASE
    batch_size = _FALLBACK_BATCH_SIZE
    input_path = _FALLBACK_PROCESSED_DIR / "rul_input.json"
    out_path   = _FALLBACK_PROCESSED_DIR / "rul_results.json"

    try:
        from airflow.hooks.base import BaseHook
        from airflow.models import Variable

        try:
            conn = BaseHook.get_connection("fastapi_predictiq")
            scheme = conn.schema or "http"
            host   = conn.host   or "localhost"
            port   = conn.port   or 8000
            api_base = f"{scheme}://{host}:{port}/api/v1"
        except Exception:
            pass

        try:
            batch_size = int(Variable.get("PREDICTIQ_BATCH_SIZE", default_var=str(_FALLBACK_BATCH_SIZE)))
        except Exception:
            pass

        ti = context.get("ti")
        if ti:
            xcom_path = ti.xcom_pull(task_ids="build_rul_input")
            if xcom_path:
                input_path = Path(xcom_path)
                out_path   = input_path.parent / "rul_results.json"

    except Exception:
        pass

    return api_base, batch_size, input_path, out_path


def run(**context):
    api_base, batch_size, input_path, out_path = _get_config(context)

    print(f"[run_rul] Reading {input_path}...")
    with open(input_path) as f:
        records = json.load(f)
    print(f"[run_rul] Loaded {len(records)} sequences")

    # Rename 'sequence' → 'sensor_sequence' (what the API expects)
    api_records = [
        {
            "machine_id":      r["machine_id"],
            "component":       r["component"],
            "sensor_sequence": r["sequence"],
        }
        for r in records
    ]

    num_batches = math.ceil(len(api_records) / batch_size)
    print(f"[run_rul] Calling POST {api_base}/rul/batch "
          f"in {num_batches} batches of {batch_size}...")

    all_results = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for i in range(num_batches):
            chunk = api_records[i * batch_size : (i + 1) * batch_size]
            print(f"  [run_rul] Batch {i + 1}/{num_batches} — {len(chunk)} sequences...")
            resp = client.post(f"{api_base}/rul/batch", json={"requests": chunk})
            resp.raise_for_status()
            response_data = resp.json()
            if response_data.get("status") != "success":
                raise RuntimeError(
                    f"RUL API error on batch {i + 1}: {response_data.get('error')}"
                )
            batch_results = response_data["data"]
            all_results.extend(batch_results)
            print(f"  [run_rul] Batch {i + 1} done — got {len(batch_results)} results")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f)

    print(f"[run_rul] Saved {len(all_results)} RUL results to {out_path}")
    return str(out_path)   # ← XCom: Task 6 pulls this path


if __name__ == "__main__":
    run()