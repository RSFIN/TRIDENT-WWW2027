import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

def run(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("=== STDOUT ===")
    print(r.stdout[:3000])
    print("=== STDERR ===")
    print(r.stderr[:2000])
    return r.returncode

if __name__ == "__main__":
    # try deleting the old T2a table block
    rc = run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
              "--block-id", "doxcn0KDp3OHfmSLmdV4eipcsCe", "--as", "user"])
    sys.exit(rc)
