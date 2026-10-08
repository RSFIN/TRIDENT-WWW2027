import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                        "--keyword", "附录 A：LLM 少样本基线|参考来源", "--context-before", "1",
                        "--context-after", "2", "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout[:3500])
    sys.exit(0)
