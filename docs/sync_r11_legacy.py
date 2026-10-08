# -*- coding: utf-8 -*-
"""Sync R11 legacy rerun numbers (qwen3-14b CLINC150) into Feishu doc.
- block_replace T2a html5-block (doxcnNjORQmh1FpBXebG6EsBPEM) -> docs/t2a_table.html
- block_replace T2b html5-block (doxcnFpBylKjsxj4ppLfdyDGCFb) -> docs/t2b_table.html
- str_replace T2a conclusion sentence (legacy wording)
"""
import subprocess
import sys

DOC = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
T2A_BLOCK = "doxcnNjORQmh1FpBXebG6EsBPEM"
T2B_BLOCK = "doxcnFpBylKjsxj4ppLfdyDGCFb"
T2A_HTML = r"@E:\PY-Project\WWW2027\code\intent-conformal\docs\t2a_table.html"
T2B_HTML = r"@E:\PY-Project\WWW2027\code\intent-conformal\docs\t2b_table.html"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print("CMD:", " ".join(cmd))
    print("RC:", r.returncode)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:2000])
    if r.returncode != 0:
        sys.exit(r.returncode)


# 1) replace T2a html5-block
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "block_replace", "--block-id", T2A_BLOCK,
    "--content", f'<html5-block path="{T2A_HTML}"></html5-block>', "--as", "user"
])

# 2) replace T2b html5-block
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "block_replace", "--block-id", T2B_BLOCK,
    "--content", f'<html5-block path="{T2B_HTML}"></html5-block>', "--as", "user"
])

# 3) str_replace T2a conclusion sentence
old_sentence = "R11 TECP-style（qwen3-8b/14b）F1 0.938/0.955 高于 Ours，但其 OOS 拒识率仅 0.4%–6.3%（Ours 为 100%），14b 依赖强制候选分类策略且 ECE 0.038–0.055 劣于 Ours 0.005"
new_sentence = "R11 TECP-style（qwen3-8b/14b，legacy 原提示）F1 0.938/0.948 高于 Ours，但其 OOS 拒识率仅 0.4%–8.3%（Ours 为 100%），且 ECE 0.045–0.055 劣于 Ours 0.005"
run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "str_replace", "--pattern", old_sentence,
    "--content", new_sentence, "--as", "user"
])

print("ALL OK")
