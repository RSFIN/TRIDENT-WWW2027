import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

new_text = ("<p>以下为已完成的基线实测数据，统一口径 v1.0：α=0.1，split conformal。Ours 为本方法 qwen3.7-plus "
"免训练版（seed42/seed123 两次独立运行均值，各指标跨 seed 差异 ≤0.016 IoU、≤0.008 ECE，见统计检验章节）。"
"复现基线：R5=CICC [31]、R12=MULACONF [33]、R12-QP=QP-OneModel [12]（按统一口径 v1.0 重算）、R10=UniCR [34]、"
"R11=TECP-style（qwen3-8b/14b，single run）、R13=SetFit+MC Route（代理口径，效率副表）、LLM 少样本基线"
"（qwen3-14b/DeepSeek-R1-distill，10-shot 最强变体入主表、全组合见附录）。另附 P0 等风险对比与澄清公平化"
"补充实验（2026-10-08，纯本地零 API，可复现）。</p>")

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
                        "--block-id", "doxcn0M9wLkJImn1Ha2tZ2O3Vqg",
                        "--content", new_text, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:1500]); print(r.stderr[:800])
    sys.exit(r.returncode)
