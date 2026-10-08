import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                        "--keyword", "以下为 2026-09 已完成的基线实测数据", "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    text = r.stdout
    m = re.search(r'<p id="(doxcn[^"]+)"', text)
    print("candidate id:", m.group(1) if m else "NOT FOUND")
    idx = text.find("以下为 2026-09")
    print(text[max(0, idx-150):idx+120] if idx >= 0 else text[:600])
    sys.exit(0 if m else 1)
