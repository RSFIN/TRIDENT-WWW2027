import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

conclusion = ("<p><b>T2a 结论</b>：Ours 在统一口径下实现 100% OOS 拒识（1.000）与最优校准（ECE 0.005），"
"avg set=0.87（含拒识 ∅）贴合名义 α=0.1 的 coverage。与共形家族对比：CICC/MULACONF 为达 coverage "
"输出 78–135 个意图的巨大集合（avg set 4.5–135.1），Ours 以紧致集合达标。与生成式单点基线对比："
"R12-QP 按统一口径 v1.0 重算后 Macro-F1 仅 0.137–0.203（早期口径 0.503–0.761 不再采用），且固定 5 候选导致 "
"Cov(J≥0.5)=0；R11 TECP-style（qwen3-8b/14b）F1 0.938/0.955 高于 Ours，但其 OOS 拒识率仅 0.4%–6.3%"
"（Ours 为 100%），14b 依赖强制候选分类策略且 ECE 0.038–0.055 劣于 Ours 0.005；LLM 少样本基线"
"（qwen3-14b 10-shot F1 0.956）以 OOS 漏检 16.7%（DeepSeek 漏检 4.5%）和无三态/无校准机制为代价。"
"风险权衡视角（P0-1 补充实验）：CLINC150 上 OOS 与域内完全可分，将 OOS gate 从 oos_fpr=0.05 收紧至 "
"0.0005 后，Ours F1 0.899→0.929、误拒率 0.139→0.097，OOS 拒识保持 100%——与最强少样本的 F1 差距"
"（~2.7pt）源于保守的 FPR95 拒绝阈值而非方法缺陷，且 ECE 优 6–8 倍。与 learned fusion 对比：R10（UniCR）"
"仅 +0.6pt F1、−1.5pt 误拒微弱领先，但需在 calib 上拟合四维特征头，多意图上集合能力退化（见 T2b）；"
"Ours 零训练 rank 归一化融合达到相当 coverage（0.856 vs 0.867）与校准（ECE 0.005 vs 0.004）——"
"免训练不是性能代价，而是与学参融合等价的能力。</p>")

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
                        "--block-id", "doxcnf3YlqEVwGQkeeEMSuPcVwb",
                        "--content", conclusion, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:2500])
    print(r.stderr[:1500])
    sys.exit(r.returncode)
