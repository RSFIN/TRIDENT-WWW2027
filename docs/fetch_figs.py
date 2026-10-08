import subprocess, sys, json

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
section = "doxcncdVCnZz0qrycyfxp15lbVg"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", section, "--detail", "full", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout)
    sys.exit(r.returncode)
