# -*- coding: utf-8 -*-
"""Rebuild the sentence chunk that was stripped when '<0.10' was parsed as a tag."""
import subprocess
import sys

DOC = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

old = "前两 beam 概率差LLM 少样本基线（qwen3-14b 10-shot）IoU 0.872/0.952 反超 Ours"
new = "前两 beam 概率差小于 0.10 即澄清，MixATIS 澄清率 97%）；R11 TECP-style（qwen3-8b/14b）为 argmax 单点输出（Hit@1=Hit@3、无澄清），MixSNIPS IoU 0.536/0.529 低于 Ours 0.757，MixATIS 0.368/0.384 与 Ours 0.387 相当但 ECE 0.471–0.566 远劣于 Ours 0.219；LLM 少样本基线（qwen3-14b 10-shot）IoU 0.872/0.952 反超 Ours"

r = subprocess.run([
    "lark-cli", "docs", "+update", "--doc", DOC,
    "--command", "str_replace", "--pattern", old,
    "--content", new, "--as", "user"
], capture_output=True, text=True, encoding="utf-8")
print("RC:", r.returncode)
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[:1000])
sys.exit(r.returncode)
