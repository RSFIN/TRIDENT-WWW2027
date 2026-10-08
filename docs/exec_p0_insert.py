import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

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
    # fetch to locate T2b conclusion paragraph id
    r2 = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                         "--keyword", "T2b 结论", "--detail", "with-ids", "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    text = r2.stdout
    # try multiple patterns
    m = re.search(r'<p id="(doxcn[^"]+)"[^>]*><b>T2b 结论', text)
    if not m:
        m = re.search(r'"(doxcn[^"]+)"[^>]*><b>T2b 结论', text)
    if not m:
        # print surroundings for debug
        idx = text.find("T2b 结论")
        print("debug idx:", idx)
        print(text[max(0, idx-400):idx+100])
        sys.exit(1)
    anchor = m.group(1)
    print("anchor =", anchor)
    r3 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                         "--block-id", anchor, "--content", p0_section, "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r3.stdout[:3000]); print(r3.stderr[:1500])
    sys.exit(r3.returncode)
