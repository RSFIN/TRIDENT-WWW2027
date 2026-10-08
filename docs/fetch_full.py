import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--detail", "simple", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    text = r.stdout
    # save full to file for inspection
    with open("docs/full_dump.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("len:", len(text))
    print(text[:4000])
    print(r.stderr[:800])
    sys.exit(r.returncode)
