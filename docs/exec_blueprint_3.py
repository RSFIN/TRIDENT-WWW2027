import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
anchor = "doxcnH6mQIVMgpAtPSiHHugpnob"

if __name__ == "__main__":
    # step1: insert new html5-block after preamble
    content = '<html5-block path="@E:/PY-Project/WWW2027/code/intent-conformal/docs/figure_blueprint.html"></html5-block>'
    r1 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                         "--block-id", anchor, "--content", content, "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print("STEP1:", r1.stdout[:1000]); print(r1.stderr[:400])
    if r1.returncode != 0:
        sys.exit(r1.returncode)
    # step2: replace preamble text (same anchor id, unchanged by step1)
    new_preamble = ("<p>论文配图＝11 张 Figure＋5 张 Table，蓝图总览见下图；每张图与第 5 节实验一一对应，"
                    "绘图安排在 W3–W4，先用占位数据跑通版式，再以最终实验数字替换。"
                    "F11（P0-1 等风险权衡曲线）由 scripts/26_risk_clarify.py 产出，纯本地零 API，"
                    "可在投稿前直接渲染。</p>")
    r2 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
                         "--block-id", anchor, "--content", new_preamble, "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print("STEP2:", r2.stdout[:1000]); print(r2.stderr[:400])
    sys.exit(r2.returncode)
