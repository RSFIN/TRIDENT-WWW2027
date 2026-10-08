import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
section = "doxcncdVCnZz0qrycyfxp15lbVg"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", section, "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print("RC:", r.returncode)
    print(r.stdout[:3000])
    print("STDERR:", r.stderr[:800])
    sys.exit(0)
