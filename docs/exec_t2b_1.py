import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

def run(args, desc, must_ok=True):
    print("### " + desc)
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    out = r.stdout[:2500]
    print(out)
    if r.stderr.strip():
        print("STDERR: " + r.stderr[:1500])
    if must_ok and ('"ok": false' in out or '"result": "failed"' in out):
        print(">>> FAILED, abort")
        sys.exit(1)
    return out

if __name__ == "__main__":
    # 1) delete old T2b table
    run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
         "--block-id", "doxcnOnG4swXsobIIyD38v80k3d", "--as", "user"], "del old T2b table")
    # 2) insert new T2b html table after T2b heading
    run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
         "--block-id", "doxcnOIzX8jRpe1lqD6iDQjWlJb",
         "--content", '<html5-block path="' + HT + 't2b_table.html"></html5-block>', "--as", "user"],
        "insert T2b html table")
    print("T2B TABLE DONE")
