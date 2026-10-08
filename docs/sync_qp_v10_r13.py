# -*- coding: utf-8 -*-
"""Supplement QP v1.0 multi-intent rows (T2b) + R13 full set metrics into Feishu doc.
- block_replace T2b html5-block (doxcnlPyj3GoqoYd7yaYoNG1vqe) -> docs/t2b_table.html
- block_replace R13 html5-block (doxcnwmphG3Wm7PD4yNJKrgZOhJ) -> docs/r13_table.html
- str_replace R13 paragraph (drop 'single-label only' wording)
- str_replace T2b conclusion (add QP v1.0 multi-intent comparison)
"""
import subprocess
import sys

DOC = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
T2B_BLOCK = "doxcnlPyj3GoqoYd7yaYoNG1vqe"
R13_BLOCK = "doxcnwmphG3Wm7PD4yNJKrgZOhJ"
T2B_HTML = r"@E:\PY-Project\WWW2027\code\intent-conformal\docs\t2b_table.html"
R13_HTML = r"@E:\PY-Project\WWW2027\code\intent-conformal\docs\r13_table.html"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print("CMD:", " ".join(cmd)[:200])
    print("RC:", r.returncode)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:1000])
    if r.returncode != 0:
        sys.exit(r.returncode)


# 1) replace T2b html5-block (now 20 rows incl QP v1.0 x6)
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "block_replace", "--block-id", T2B_BLOCK,
    "--content", f'<html5-block path="{T2B_HTML}"></html5-block>', "--as", "user"
])

# 2) replace R13 html5-block (full set metrics)
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "block_replace", "--block-id", R13_BLOCK,
    "--content", f'<html5-block path="{R13_HTML}"></html5-block>', "--as", "user"
])

# 3) R13 paragraph: drop stale 'single-label only' wording
old_r13 = "R13（SetFit 少量微调 + MC-Dropout 不确定性路由，LLM fallback）以 6.3%–15.9% 的 LLM 调用率实现系统级降本，但仅输出单标签：多意图集合指标 N/A、校准无保证（ECE 为代理口径），与 Ours 的免训练 + 集合级共形 + 三态定位不同，作为系统级效率对照（误拒/OOS/澄清均为代理口径，故不进主表）。"
new_r13 = "R13（SetFit 少量微调 + MC-Dropout 不确定性路由，LLM fallback，v5）以 6.3%–15.9% 的 LLM 调用率实现系统级降本；集合指标基于 SetFit softmax 概率质量阈值生成（CLINC/MixATIS/MixSNIPS 阈值 0.50/0.67/0.69）。核心对照：集合质量（IoU 0.190/0.302/0.475）全面低于 Ours（0.856/0.387/0.757），校准无保证（ECE 0.374–0.695 vs Ours 0.005/0.219/0.045），CLINC150 OOS 拒识仅 5.7%（Ours 100%）——省调用但无覆盖率保证与集合能力，与 Ours 的免训练 + 集合级共形 + 三态定位不同；误拒/OOS/澄清均为代理口径，故仅入效率副表。"
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "str_replace", "--pattern", old_r13,
    "--content", new_r13, "--as", "user"
])

# 4) T2b conclusion: insert QP v1.0 multi-intent comparison before R11 sentence
old_seg = "新基线定性：R11 TECP-style（qwen3-8b/14b）为 argmax 单点输出（Hit@1=Hit@3、无澄清），MixSNIPS IoU 0.536/0.529 低于 Ours 0.757，MixATIS 0.368/0.384 与 Ours 0.387 相当但 ECE 0.471–0.566 远劣于 Ours 0.219；"
new_seg = "新基线定性：R12-QP v1.0（统一口径重算，3-seed 均值）在 MixATIS/MixSNIPS 上即使 10-shot 集合质量仍低于 Ours（IoU 0.299/0.460 vs 0.387/0.757），且集合偏大（avg 4.90/4.25，真实意图≤3）、从不拒识（误拒 0）、澄清为统计性触发（前两 beam 概率差<0.10 即澄清，MixATIS 澄清率 97%）、ECE 随 shot 恶化至 0.154/0.321——生成式固定候选集无校准拒识机制；R11 TECP-style（qwen3-8b/14b）为 argmax 单点输出（Hit@1=Hit@3、无澄清），MixSNIPS IoU 0.536/0.529 低于 Ours 0.757，MixATIS 0.368/0.384 与 Ours 0.387 相当但 ECE 0.471–0.566 远劣于 Ours 0.219；"
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "str_replace", "--pattern", old_seg,
    "--content", new_seg, "--as", "user"
])

print("ALL OK")
