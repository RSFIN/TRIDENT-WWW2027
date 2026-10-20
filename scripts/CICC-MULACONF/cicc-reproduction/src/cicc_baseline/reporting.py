"""Console table for the WWW 2027 unified metric protocol v1.0."""

from __future__ import annotations

import json
from pathlib import Path

from .unified_v1 import TABLE_COLUMNS


def _cell(row: dict[str, float], repeats: int) -> str:
    return f"{row['mean']:.4f} ± {row['std']:.4f}" if repeats > 1 else f"{row['mean']:.4f}"


def format_console_report(output_dir: str | Path) -> str:
    root = Path(output_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["repeat_summary"]
    headers = ["方法", *(label for label, _ in TABLE_COLUMNS)]
    values = ["CICC", *(
        _cell(aggregate[f"unified_v1.{key}"], len(summary["seeds"]))
        for _, key in TABLE_COLUMNS
    )]
    separator = ["---", *("---:" for _ in TABLE_COLUMNS)]
    return "\n".join([
        f"CLINC150｜统一口径 v1.0｜α=0.1｜{len(summary['seeds'])} 次重复汇总（拒识计入 ID 分母）",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
        "| " + " | ".join(values) + " |",
        f"结果目录：{root.resolve()}",
    ])


def print_console_report(output_dir: str | Path) -> None:
    print(format_console_report(output_dir))
