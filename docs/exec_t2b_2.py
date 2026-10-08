import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

conclusion = ("<p><b>T2b 结论</b>：集合级对比中，MULACONF Tree fixed α 在 MixATIS 上 avg set=748、误拒率=1.0"
"（不可用），Powerset 版 set size=150.6 同样过大；Ours 以紧致配置（K=1、K_r=1，P3 消融定稿）在 "
"MixATIS/MixSNIPS 上 avg set=1.52/1.80，三态决策（release/clarify/reject）保持可控。与 learned fusion（R10）"
"对比：R10 的 logistic 头在多意图上把集合压成单点（澄清率 0%），Ours 以紧致配置在 MixSNIPS 上 IoU 0.757 vs "
"0.491、Hit@3 0.619 vs 0.163 反超；MixATIS 上 IoU 0.387（R10 0.451）、ECE 0.219（R10 0.141）差距已较旧配置"
"显著缩小。新基线定性：R11 TECP-style（qwen3-8b/14b）为 argmax 单点输出（Hit@1=Hit@3、无澄清），MixSNIPS "
"IoU 0.536/0.529 低于 Ours 0.757，MixATIS 0.368/0.384 与 Ours 0.387 相当但 ECE 0.471–0.566 远劣于 Ours 0.219；"
"LLM 少样本基线（qwen3-14b 10-shot）IoU 0.872/0.952 反超 Ours（0.387/0.757），但无共形覆盖保证、无 OOS 拒识、"
"无澄清交互、ECE 不可得。公平化视角（P0-2 补充实验）：oracle 澄清口径下 Ours MixSNIPS IoU 0.754→0.942"
"（逼近少样本 0.952）、MixATIS IoU 0.396→0.760（少样本 0.872），集合质量与最强少样本相当——Ours 的多意图差距"
"源于澄清态不直接输出最终集合的口径差异，且以更紧致的 avg set（1.52/1.80 vs 2.15/2.01）换取交互可控性。"
"综合三数据集：Ours 以零训练 rank 融合 + 分层共形实现唯一同时具备覆盖率保证、OOS 全拒识、校准与三态交互的"
"方案；少样本基线的 F1/IoU 领先以可靠性指标（OOS 漏检、无校准）为代价，R11/R13 则分别缺校准与集合能力。</p>")

p0_section = ("<h3>P0 补充实验：等风险对比与澄清公平化（2026-10-08）</h3>"
"<p>为回应两大审稿风险（Ours 的 Macro-F1/IoU 并非各表最高），补充两组纯本地零 API 实验"
"（scripts/26_risk_clarify.py，可复现）。</p>"
"<p><b>P0-1 等风险对比（表 1）</b>：在 CLINC150 上收紧 OOS gate（oos_fpr 0.05→0.005→0.0005）。"
"OOS 与域内完全可分，任意 FPR 下 OOS 拒识率均为 1.000；收紧阈值仅提升域内指标：主配置 F1 0.899/误拒 0.139"
"→ 0.0005 档 F1 0.929/误拒 0.097（2-seed 均值），IoU 0.856→0.897，ECE 全程 ≤0.008 保持名义 α 覆盖。</p>"
'<html5-block path="' + HT + 'p0_1_table.html"></html5-block>'
"<p><b>P0-2 澄清公平化（表 2）</b>：将澄清态按 oracle 真值集合口径重算集合指标（一致性上界）。"
"MixSNIPS IoU 0.754→0.942（逼近少样本最强 0.952）、MixATIS IoU 0.396→0.760（少样本 0.872）；CLINC150 澄清率=0，"
"两口径一致（检查通过）。</p>"
'<html5-block path="' + HT + 'p0_2_table.html"></html5-block>'
"<p><b>对审稿风险的两点回应</b>：① 单意图 Macro-F1——Ours 的差距是保守拒绝阈值（FPR95）的有意选择，"
"收紧后与最强少样本（漏检 4.5%–16.7%）的差距收敛至 ~2.7pt，且 ECE 优 6–8 倍；② 多意图 IoU——差距主要来自"
"澄清态口径，oracle 下 MixSNIPS 追平、MixATIS 大幅收窄。两点均不改变论文核心主张：Ours 是唯一同时具备"
"覆盖率保证、OOS 全拒识、可校准置信度与三态交互的免训练方案。</p>"
"<h3>R13 系统级基线（效率副表，不入主表）</h3>"
"<p>R13（SetFit 少量微调 + MC-Dropout 不确定性路由，LLM fallback）以 6.3%–15.9% 的 LLM 调用率实现系统级降本，"
"但仅输出单标签：多意图集合指标 N/A、校准无保证（ECE 为代理口径），与 Ours 的免训练 + 集合级共形 + 三态"
"定位不同，作为系统级效率对照（详细数字见下表，误拒/OOS/澄清均为代理口径，故不进主表）。</p>"
'<html5-block path="' + HT + 'r13_table.html"></html5-block>')

if __name__ == "__main__":
    # 1) replace T2b conclusion
    r1 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
                         "--block-id", "doxcnxoGYn4hrP8K8iCzz9k0hTf",
                         "--content", conclusion, "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r1.stdout[:2500]); print(r1.stderr[:1500])
    if '"ok": false' in r1.stdout:
        sys.exit(1)
    # 2) insert P0 section after the (now replaced) T2b conclusion - fetch fresh anchor
    r2 = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                         "--keyword", "T2b 结论", "--detail", "with-ids", "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    import re
    m = re.search(r'<p id="(doxcn[^"]+)"[^>]*><b>T2b 结论', r2.stdout)
    if not m:
        print("anchor not found"); print(r2.stdout[:2000]); sys.exit(1)
    anchor = m.group(1)
    r3 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                         "--block-id", anchor, "--content", p0_section, "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r3.stdout[:2500]); print(r3.stderr[:1500])
    sys.exit(r3.returncode)
