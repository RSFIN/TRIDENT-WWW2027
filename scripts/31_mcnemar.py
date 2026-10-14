#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_mcnemar.py — McNemar significance test between method pairs.

Usage:
  python scripts/31_mcnemar.py --dataset clinc150 --pred-dir data/predictions/clinc150

Inputs:
  - test.jsonl with fields: main_intent (gold), oos (bool), llm_main (pred)
  - Optional: other methods' per-query predictions loaded via load_*() hooks.

Output:
  - results/mcnemar_<dataset>.md : pairwise table with *** p<0.001 / ** p<0.01 / * p<0.05
  - stdout summary
"""
import argparse, json, math, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------- statistics ----------
def mcnemar(b: int, c: int) -> float:
    """McNemar chi-square with continuity correction. Returns p-value."""
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # p = P(chi2_1 > chi2) ; use survival of chi-square_1
    # chi2_1 survival = erfc(sqrt(chi2/2))
    return math.erfc(math.sqrt(chi2 / 2.0))


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ---------- loaders ----------
def load_ours(test_path: Path):
    """Ours: per-query main-intent correctness.
    After conformal calibration, rejected queries (empty set) are counted as wrong.
    We approximate: use llm_main vs main_intent for in-domain non-rejected queries.
    Rejected queries are loaded from a separate reject mask if available.
    """
    correct = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = r.get("main_intent")
            pred = r.get("llm_main")
            # If rejected (empty set after conformal), pred=None => wrong
            if r.get("rejected", False):
                correct.append(False)
            else:
                correct.append(pred == gold)
    return correct


def load_baseline_preds(pred_path: Path, gold_key="main_intent", pred_key="llm_main"):
    """Generic loader for a baseline's per-query predictions."""
    correct = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = r.get(gold_key)
            pred = r.get(pred_key)
            correct.append(pred == gold)
    return correct


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="extra baseline jsonl files: name=path pairs")
    args = ap.parse_args()

    test_path = Path(args.pred_dir) / "test.jsonl"
    if not test_path.exists():
        print(f"[ERR] {test_path} not found"); sys.exit(1)

    methods = {"Ours": load_ours(test_path)}

    # Parse extra baselines: --baselines CICC=path1.jsonl MULACONF=path2.jsonl
    for spec in args.baselines:
        name, path = spec.split("=", 1)
        methods[name] = load_baseline_preds(Path(path))

    names = list(methods.keys())
    n = len(methods[names[0]])
    for k, v in methods.items():
        assert len(v) == n, f"{k} has {len(v)} samples, expected {n}"

    # Pairwise
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            a_arr, b_arr = methods[A], methods[B]
            b = sum(1 for x, y in zip(a_arr, b_arr) if x and not y)
            c = sum(1 for x, y in zip(a_arr, b_arr) if not x and y)
            p = mcnemar(b, c)
            rows.append((A, B, b, c, p, stars(p)))

    # Print
    print(f"\n=== McNemar on {args.dataset} (N={n}) ===\n")
    print(f"{'A':<20} {'B':<20} {'A&B~B':>6} {'A~B&B':>6} {'p':>10} {'sig':>5}")
    print("-" * 75)
    for A, B, b, c, p, s in rows:
        print(f"{A:<20} {B:<20} {b:>6} {c:>6} {p:>10.4f} {s:>5}")

    # Save markdown
    out = ROOT / "results" / f"mcnemar_{args.dataset}.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# McNemar Test — {args.dataset} (N={n})\n\n")
        f.write("| A | B | A right B wrong | A wrong B right | p-value | sig |\n")
        f.write("|---|---|---|---|---|---|\n")
        for A, B, b, c, p, s in rows:
            f.write(f"| {A} | {B} | {b} | {c} | {p:.4f} | {s} |\n")
        f.write("\n\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001\n")
    print(f"\n[OK] saved -> {out}")


if __name__ == "__main__":
    main()
