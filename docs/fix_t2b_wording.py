# -*- coding: utf-8 -*-
"""Fix the '<0.10' tag-stripped wording in T2b conclusion."""
import subprocess
import sys

DOC = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

old = "澄清为统计性触发（前两 beam 概率差 即澄清，MixATIS 澄清率 97%）"
new = "澄清为统计性触发（前两 beam 概率差小于 0.10 即澄清，MixATIS 澄清率 97%）"

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
