import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", "doxcne6ZvPjV82ZjjcV3BZZRfJg", "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    text = r.stdout
    with open("docs/section_ablation.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:3500])
    print(r.stderr[:500])
    sys.exit(r.returncode)
