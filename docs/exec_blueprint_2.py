import subprocess, sys, re, json

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

def fetch_anchor():
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                        "--keyword", "论文配图＝10 张 Figure|论文配图＝11 张 Figure", "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    txt = r.stdout
    m = re.search(r'<p id="(doxcn[^"]+)"', txt)
    print("anchor id:", m.group(1) if m else "NOT FOUND")
    return m.group(1) if m else None

if __name__ == "__main__":
    aid = fetch_anchor()
    if not aid:
        sys.exit(1)
    content = '<html5-block path="@E:/PY-Project/WWW2027/code/intent-conformal/docs/figure_blueprint.html"></html5-block>'
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", aid, "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout[:1200]); print(r.stderr[:600])
    sys.exit(r.returncode)
