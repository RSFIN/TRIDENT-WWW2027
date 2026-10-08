import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+history-revert", "--doc", doc,
                        "--history-version-id", "134144", "--wait-timeout-ms", "30000"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:3000])
    print(r.stderr[:2000])
    sys.exit(r.returncode)
