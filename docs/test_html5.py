import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    # insert html5-block test after T2a heading (paragraph anchor proven working)
    content = '<html5-block path="@E:/PY-Project/WWW2027/code/intent-conformal/docs/test_widget.html"></html5-block>'
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
                        "--block-id", "doxcnLaX0JT5snjksADVeCpOMYd",
                        "--content", content, "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:3000])
    print(r.stderr[:2000])
    sys.exit(r.returncode)
