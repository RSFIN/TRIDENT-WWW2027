from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cicc_baseline.statistics import mcnemar_exact  # noqa: E402


def load_predictions(path: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "test":
                continue
            rows[str(row["sample_id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact paired McNemar test for two prediction JSONL files.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    first = load_predictions(args.a)
    second = load_predictions(args.b)
    if set(first) != set(second):
        raise SystemExit("Prediction files do not contain the same in-scope test sample IDs.")
    ids = sorted(first)
    truth = [str(first[key]["true_label"]) for key in ids]
    prediction_a = [str(first[key]["top1_prediction"]) for key in ids]
    prediction_b = [str(second[key]["top1_prediction"]) for key in ids]
    print(json.dumps(mcnemar_exact(truth, prediction_a, prediction_b), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

