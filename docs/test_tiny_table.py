import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

def run_update(block_id, content):
    args = ["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
            "--block-id", block_id, "--content", content, "--as", "user"]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("=== STDOUT ===")
    print(r.stdout[:3000])
    print("=== STDERR ===")
    print(r.stderr[:2000])
    return r.returncode

if __name__ == "__main__":
    # minimal table test on T2a block
    tiny = "<table><thead><tr><th><p>方法</p></th><th><p>F1</p></th></tr></thead><tbody><tr><td><p>Ours</p></td><td><p>0.899</p></td></tr></tbody></table>"
    rc = run_update("doxcn0KDp3OHfmSLmdV4eipcsCe", tiny)
    sys.exit(rc)
