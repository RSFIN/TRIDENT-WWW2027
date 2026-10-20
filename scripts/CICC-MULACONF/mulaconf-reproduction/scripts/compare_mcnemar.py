from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mulaconf_baseline.statistics import mcnemar_exact  # noqa: E402


def load_correctness(path: str) -> dict[str, bool]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return {
        row["sample_id"]: bool(row["unified_main_correct"])
        for row in rows
        if not row.get("is_oos", False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact paired McNemar test using unified-v1 main correctness")
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    first, second = load_correctness(args.first), load_correctness(args.second)
    if set(first) != set(second):
        raise ValueError("prediction files do not contain identical in-scope sample IDs")
    ids = sorted(first)
    print(json.dumps(mcnemar_exact([first[i] for i in ids], [second[i] for i in ids]), indent=2))


if __name__ == "__main__":
    main()
