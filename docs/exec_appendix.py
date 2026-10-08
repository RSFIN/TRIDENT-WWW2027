import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
anchor = "doxcnyjIX5J5vlALJG9VA76VOSe"

content = ('<h1>附录 A：LLM 少样本基线 9 组合全表（统一口径 v1.0）</h1>'
           '<p>LLM 少样本基线的完整 3×3 组合（qwen3-8b / qwen3-14b / DeepSeek-R1-Distill × 3/5/10-shot，'
           'TF-IDF IR 选例、闭集候选标签）。统一口径 v1.0：失败请求按 reject=∅ 计入完整分母；'
           '单次全量运行无均值±std；多意图 ECE/OOS 拒识率为 N/A，澄清率全 N/A。'
           '主表（T2a/T2b）仅列各数据集最强变体，此处给出全部组合供附录与复现核对。</p>'
           '<html5-block path="@E:/PY-Project/WWW2027/code/intent-conformal/docs/appendix_llm_fewshot.html"></html5-block>')

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", anchor, "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout[:1500]); print(r.stderr[:800])
    sys.exit(r.returncode)
