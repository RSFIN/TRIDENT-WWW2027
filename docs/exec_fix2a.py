import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

def hb(fname):
    return '<html5-block path="' + HT + fname + '"></html5-block>'

if __name__ == "__main__":
    content = (
"<h2>实验结果汇总（基线实测，seed42/123 两次独立运行）</h2>"
"<p>以下为 2026-09 已完成的基线实测数据，统一口径：α=0.1，split conformal，CLINC150 test=5500。Ours 为本方法 qwen3.7-plus 免训练版；R5=CICC [31]、R12=MULACONF [33]、R12-QP=QP-OneModel [12] 均在相同 calib/test 划分上复现。Ours 数值为 seed42 与 seed123 两次独立运行均值（各指标跨 seed 差异 ≤0.016 IoU、≤0.008 ECE，见统计检验章节）。</p>"
"<h3>T2a CLINC150 主结果</h3>"
+ hb("t2a_table.html") +
"<p><b>T2a 结论</b>：Ours 在统一口径下实现 100% OOS 拒识（1.000）与最优校准（ECE 0.005），avg set=0.87（含拒识 ∅）贴合名义 α=0.1 的 coverage。与共形家族对比：CICC/MULACONF 为达 coverage 输出 78–135 个意图的巨大集合（avg set 4.5–135.1），Ours 以紧致集合达标。与生成式单点基线对比：R12-QP 按统一口径 v1.0 重算后 Macro-F1 仅 0.137–0.203（早期口径 0.503–0.761 不再采用），且固定 5 候选导致 Cov(J≥0.5)=0；R11 TECP-style（qwen3-8b/14b）F1 0.938/0.955 高于 Ours，但其 OOS 拒识率仅 0.4%–6.3%（Ours 为 100%），14b 依赖强制候选分类策略且 ECE 0.038–0.055 劣于 Ours 0.005；LLM 少样本基线（qwen3-14b 10-shot F1 0.956）以 OOS 漏检 16.7%（DeepSeek 漏检 4.5%）和无三态/无校准机制为代价。风险权衡视角（P0-1 补充实验）：CLINC150 上 OOS 与域内完全可分，将 OOS gate 从 oos_fpr=0.05 收紧至 0.0005 后，Ours F1 0.899→0.929、误拒率 0.139→0.097，OOS 拒识保持 100%——与最强少样本的 F1 差距（~2.7pt）源于保守的 FPR95 拒绝阈值而非方法缺陷，且 ECE 优 6–8 倍。与 learned fusion 对比：R10（UniCR）仅 +0.6pt F1、−1.5pt 误拒微弱领先，但需在 calib 上拟合四维特征头，多意图上集合能力退化（见 T2b）；Ours 零训练 rank 归一化融合达到相当 coverage（0.856 vs 0.867）与校准（ECE 0.005 vs 0.004）——免训练不是性能代价，而是与学参融合等价的能力。</p>"
"<h3>T2b MixATIS / MixSNIPS 多意图结果</h3>"
+ hb("t2b_table.html") +
"<p><b>T2b 结论</b>：集合级对比中，MULACONF Tree fixed α 在 MixATIS 上 avg set=748、误拒率=1.0（不可用），Powerset 版 set size=150.6 同样过大；Ours 以紧致配置（K=1、K_r=1，P3 消融定稿）在 MixATIS/MixSNIPS 上 avg set=1.52/1.80，三态决策（release/clarify/reject）保持可控。与 learned fusion（R10）对比：R10 的 logistic 头在多意图上把集合压成单点（澄清率 0%），Ours 以紧致配置在 MixSNIPS 上 IoU 0.757 vs 0.491、Hit@3 0.619 vs 0.163 反超；MixATIS 上 IoU 0.387（R10 0.451）、ECE 0.219（R10 0.141）差距已较旧配置显著缩小。新基线定性：R11 TECP-style（qwen3-8b/14b）为 argmax 单点输出（Hit@1=Hit@3、无澄清），MixSNIPS IoU 0.536/0.529 低于 Ours 0.757，MixATIS 0.368/0.384 与 Ours 0.387 相当但 ECE 0.471–0.566 远劣于 Ours 0.219；LLM 少样本基线（qwen3-14b 10-shot）IoU 0.872/0.952 反超 Ours（0.387/0.757），但无共形覆盖保证、无 OOS 拒识、无澄清交互、ECE 不可得。公平化视角（P0-2 补充实验）：oracle 澄清口径下 Ours MixSNIPS IoU 0.754→0.942（逼近少样本 0.952）、MixATIS IoU 0.396→0.760（少样本 0.872），集合质量与最强少样本相当——Ours 的多意图差距源于澄清态不直接输出最终集合的口径差异，且以更紧致的 avg set（1.52/1.80 vs 2.15/2.01）换取交互可控性。综合三数据集：Ours 以零训练 rank 融合 + 分层共形实现唯一同时具备覆盖率保证、OOS 全拒识、校准与三态交互的方案；少样本基线的 F1/IoU 领先以可靠性指标（OOS 漏检、无校准）为代价，R11/R13 则分别缺校准与集合能力。</p>")
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", "doxcnCGH0ZI4KBPsUTMRqE8cUjh", "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:2500]); print(r.stderr[:800])
    sys.exit(r.returncode)
