from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cicc_baseline.experiment import run_experiment  # noqa: E402
from cicc_baseline.reporting import print_console_report  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the training-free CICC baseline.")
    parser.add_argument("--config", required=True, help="Path to the JSON experiment config.")
    parser.add_argument("--dataset", required=True, choices=("clinc150",))
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end validation subset.")
    args = parser.parse_args()
    output_dir = run_experiment(args.config, project_root=PROJECT_ROOT, smoke=args.smoke)
    print_console_report(output_dir)


if __name__ == "__main__":
    main()

