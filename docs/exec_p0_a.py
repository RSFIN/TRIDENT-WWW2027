import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

def fetch_kw(kw):
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                        "--keyword", kw, "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return r.stdout

def insert_after(anchor, content):
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", anchor, "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:1500]); print(r.stderr[:800])
    return r.stdout

if __name__ == "__main__":
    # locate the P0 heading paragraph inserted at rev147
    text = fetch_kw("P0 补充实验")
    m = re.search(r'<h3 id="(doxcn[^"]+)"[^>]*>P0 补充实验', text)
    if not m:
        print("P0 h3 not found"); print(text[:1500]); sys.exit(1)
    h3_id = m.group(1)
    # find the intro paragraph right after h3
    m2 = re.search(r'<p id="(doxcn[^"]+)"[^>]*>为回应两大审稿风险', text)
    if not m2:
        print("intro p not found"); print(text[:2000]); sys.exit(1)
    p_id = m2.group(1)
    print("h3:", h3_id, " p:", p_id)

    partA = ("<p><b>P0-1 等风险对比（表 1）</b>：在 CLINC150 上收紧 OOS gate（oos_fpr 0.05→0.005→0.0005）。"
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
"覆盖率保证、OOS 全拒识、可校准置信度与三态交互的免训练方案。</p>")
    insert_after(p_id, partA)
    print("PART A DONE")
