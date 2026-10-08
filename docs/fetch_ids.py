import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "keyword",
                        "--keyword", "T2a CLINC150|T2a 结论|T2b MixATIS|T2b 结论",
                        "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:6000])
    print(r.stderr[:2000])
    sys.exit(r.returncode)
