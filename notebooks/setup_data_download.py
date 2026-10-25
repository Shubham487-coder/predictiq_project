from pathlib import Path
import sys

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

DATASET_FILES = {
    "PdM_telemetry.csv": "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv",
    "PdM_errors.csv": "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv",
    "PdM_maint.csv": "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv",
    "PdM_failures.csv": "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv",
    "PdM_machines.csv": "https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv",
}


def download_file(filename: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / filename

    print(f"Downloading {filename} from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def print_csv_summary(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))


def main() -> int:
    try:
        downloaded_paths = [
            download_file(filename, url)
            for filename, url in DATASET_FILES.items()
        ]

        print("\nCSV summaries:")
        for csv_path in downloaded_paths:
            print_csv_summary(csv_path)

        print(f"\nDownloaded {len(downloaded_paths)} CSV file(s) to {RAW_DIR}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
