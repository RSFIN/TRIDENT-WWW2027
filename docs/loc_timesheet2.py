import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
sec = "doxcnSVGtvkgADi7ICoRggFESjc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", sec, "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout[:4000])
    sys.exit(0)
