import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
HT = r"@E:/PY-Project/WWW2027/code/intent-conformal/docs/"

def hb(fname):
    return '<html5-block path="' + HT + fname + '"></html5-block>'

if __name__ == "__main__":
    # STEP 1: delete the 10 blocks wrongly nested inside the ablation table td
    r1 = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
                         "--start-block-id", "doxcn8PZxUQPboc6uevfypNtnff",
                         "--end-block-id", "doxcnVF8bbUNgKCUa2JtYa9qpSc", "--as", "user"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print("### STEP1 del nested blocks")
    print(r1.stdout[:1500]); print(r1.stderr[:800])
    if '"ok": false' in r1.stdout:
        sys.exit(1)
