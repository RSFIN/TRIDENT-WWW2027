from __future__ import annotations

import json

from mulaconf_baseline.reporting import format_console_report
from mulaconf_baseline.unified_v1 import TABLE_COLUMNS


def test_console_report_is_one_dataset_table(tmp_path) -> None:
    aggregate = {
        f"unified_v1.{key}": {"mean": 0.5, "std": 0.1}
        for _, key in TABLE_COLUMNS
    }
    summary = {
        "seeds": [42, 43, 44],
        "results": {"clinc150": {"powerset": aggregate, "tree_fixed_alpha": aggregate}},
    }
    (tmp_path / "summary_clinc150.json").write_text(json.dumps(summary), encoding="utf-8")
    report = format_console_report(tmp_path, "summary_clinc150.json")
    assert "CLINC150｜统一口径 v1.0" in report
    assert report.count("| powerset |") == 1
    assert report.count("| tree_fixed_alpha |") == 1
    assert "Seed 42" not in report
