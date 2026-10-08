"""Watchdog：等待 MixSNIPS test 完成 → 自动运行 04 + 04b 评估 → 汇总结果落盘。

轮询 test.jsonl 行数，达到 2199 或文件 5 分钟不再增长时视为完成；
随后运行 scripts/04_calibrate_and_eval.py 和 scripts/04b_eval_extra.py，
把输出追加写入 results/mixsnips_final_summary.txt 后退出。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TEST = ROOT / "data" / "predictions" / "mixsnips" / "test.jsonl"
TARGET = 2199
SUMMARY = ROOT / "results" / "mixsnips_final_summary.txt"

def count_lines(p):
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    print(f"[watchdog] 等待 MixSNIPS test 完成 (target={TARGET}) ...", flush=True)
    stable_since = None
    last_n = -1
    while True:
        n = count_lines(TEST)
        now = time.time()
        if n >= TARGET:
            print(f"[watchdog] test 完成: {n} 条", flush=True)
            break
        if n == last_n:
            if stable_since is None:
                stable_since = now
            elif now - stable_since > 300:
                print(f"[watchdog] 文件已 {int(now-stable_since)}s 无增长 (n={n})，判定完成", flush=True)
                break
        else:
            stable_since = None
            last_n = n
            print(f"[watchdog] 进度: {n}/{TARGET} ({n/TARGET*100:.1f}%)", flush=True)
        time.sleep(60)

    results = []
    for script, label in [
        ("scripts/04_calibrate_and_eval.py", "04 L1-L5"),
        ("scripts/04b_eval_extra.py", "04b ECE/误拒率"),
    ]:
        print(f"[watchdog] 运行 {label} ...", flush=True)
        r = subprocess.run([str(PY), script, "--dataset", "mixsnips"],
                           capture_output=True, text=True, encoding="utf-8",
                           cwd=str(ROOT), timeout=1800)
        results.append(f"===== {label} (exit={r.returncode}) =====\n{r.stdout}\n{r.stderr[-2000:] if r.stderr else ''}")
        print(f"[watchdog] {label} exit={r.returncode}", flush=True)

    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "\n".join(results))
    print(f"[watchdog] 汇总已写入 {SUMMARY}", flush=True)

if __name__ == "__main__":
    main()
