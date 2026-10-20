from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mulaconf_baseline.experiment import run_experiment  # noqa: E402
from mulaconf_baseline.reporting import print_console_report  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run powerset/tree multi-label conformal baselines.")
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument(
        "--dataset",
        choices=("mixatis", "mixsnips", "clinc150"),
        required=True,
        help="Run exactly one dataset",
    )
    parser.add_argument("--smoke", action="store_true", help="Run one seed on small subsets")
    args = parser.parse_args()
    dataset_filter = args.dataset
    output_dir = run_experiment(
        args.config,
        project_root=PROJECT_ROOT,
        smoke=args.smoke,
        dataset_filter=dataset_filter,
    )
    summary_filename = f"summary_{dataset_filter}.json"
    print_console_report(output_dir, summary_filename)


if __name__ == "__main__":
    main()
