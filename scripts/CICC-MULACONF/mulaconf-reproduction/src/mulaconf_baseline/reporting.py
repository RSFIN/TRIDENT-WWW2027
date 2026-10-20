"""Console tables for the WWW 2027 unified metric protocol v1.0."""

from __future__ import annotations

import json
from pathlib import Path

from .unified_v1 import TABLE_COLUMNS


def _cell(row: dict[str, float], repeats: int) -> str:
    return f"{row['mean']:.4f} ± {row['std']:.4f}" if repeats > 1 else f"{row['mean']:.4f}"


def format_console_report(output_dir: str | Path, summary_filename: str = "summary.json") -> str:
    root = Path(output_dir)
    summary = json.loads((root / summary_filename).read_text(encoding="utf-8"))
    if len(summary["results"]) != 1:
        raise ValueError("统一口径输出要求每条命令只运行一个数据集")
    dataset, methods = next(iter(summary["results"].items()))
    headers = ["方法", *(label for label, _ in TABLE_COLUMNS)]
    separator = ["---", *("---:" for _ in TABLE_COLUMNS)]
    lines = [
        f"{dataset.upper()}｜统一口径 v1.0｜α=0.1｜{len(summary['seeds'])} 次重复汇总（拒识计入 ID 分母）",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for method, aggregate in methods.items():
        # MixATIS/MixSNIPS have no OOS annotation, so unified_v1.oos_reject_rate
        # is never computed there. An undefined metric must render as "n/a" rather
        # than crash the whole console table (the JSON summary is unaffected).
        values = [method, *(
            _cell(aggregate[f"unified_v1.{key}"], len(summary["seeds"]))
            if f"unified_v1.{key}" in aggregate
            else "n/a"
            for _, key in TABLE_COLUMNS
        )]
        lines.append("| " + " | ".join(values) + " |")
    lines.append(f"结果目录：{root.resolve()}")
    return "\n".join(lines)


def print_console_report(output_dir: str | Path, summary_filename: str = "summary.json") -> None:
    print(format_console_report(output_dir, summary_filename))
