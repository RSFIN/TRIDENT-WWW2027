from __future__ import annotations

import json

from cicc_baseline.reporting import format_console_report
from cicc_baseline.unified_v1 import TABLE_COLUMNS


def test_console_report_is_one_unified_table(tmp_path) -> None:
    aggregate = {
        f"unified_v1.{key}": {"mean": 0.5, "std": 0.1}
        for _, key in TABLE_COLUMNS
    }
    (tmp_path / "summary.json").write_text(
        json.dumps({"seeds": [42, 43, 44], "repeat_summary": aggregate}), encoding="utf-8"
    )
    report = format_console_report(tmp_path)
    assert "CLINC150｜统一口径 v1.0" in report
    assert "Macro-F1" in report and "OOS拒识率" in report
    assert report.count("| CICC |") == 1
    assert "Seed 42" not in report
