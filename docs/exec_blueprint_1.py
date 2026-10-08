import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_delete",
                        "--block-id", "doxcnbRJgTW2cW6DSwMbItCgJIh", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:1200]); print(r.stderr[:600])
    sys.exit(r.returncode)
