import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

def run(args, desc):
    print("### " + desc)
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    out = r.stdout[:2500]
    print(out)
    if r.stderr.strip():
        print("STDERR: " + r.stderr[:1500])
    if '"ok": false' in out or '"result": "failed"' in out:
        print(">>> FAILED, abort sequence")
        sys.exit(1)
    return r

def hb5(fname):
    return '<html5-block path="' + HT + fname + '"></html5-block>'

if __name__ == "__main__":
    # 1) delete test html5-block
    run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
         "--block-id", "doxcnewqn4yZqjGrwyFf3lYbT75", "--as", "user"], "del test html5-block")
    # 2) delete old T2a table
    run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
         "--block-id", "doxcn0KDp3OHfmSLmdV4eipcsCe", "--as", "user"], "del old T2a table")
    # 3) insert new T2a html table after heading
    run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
         "--block-id", "doxcnLaX0JT5snjksADVeCpOMYd",
         "--content", hb5("t2a_table.html"), "--as", "user"], "insert T2a html table")
    print("T2A SEQUENCE DONE")
