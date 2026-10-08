import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "range",
                        "--start-block-id", "doxcnCGH0ZI4KBPsUTMRqE8cUjh",
                        "--end-block-id", "doxcnCGH0ZI4KBPsUTMRqE8cUjh",
                        "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    text = r.stdout
    with open("docs/ablation_table_dump.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("len:", len(text))
    print(text[:3000])
    print(r.stderr[:800])
    sys.exit(r.returncode)
