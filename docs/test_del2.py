import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    args = ["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
            "--block-id", "doxcn0KDp3OHfmSLmdV4eipcsCe", "--as", "user"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        print("=== STDOUT ===")
        print(r.stdout[:3000])
        print("=== STDERR ===")
        print(r.stderr[:2000])
        sys.exit(r.returncode)
    except subprocess.TimeoutExpired:
        print("TIMEOUT after 90s")
        sys.exit(2)
