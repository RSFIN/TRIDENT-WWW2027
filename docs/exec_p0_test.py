import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    anchor = "doxcnfBVvgjjP9KvQJsomus6b0f"
    content = "<h3>P0 补充实验：等风险对比与澄清公平化（2026-10-08）</h3><p>为回应两大审稿风险（Ours 的 Macro-F1/IoU 并非各表最高），补充两组纯本地零 API 实验（scripts/26_risk_clarify.py，可复现）。</p>"
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", anchor, "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:3000]); print(r.stderr[:1500])
    sys.exit(r.returncode)
