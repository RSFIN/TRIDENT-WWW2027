import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
sec = "doxcnv4qIiafsZLENVxcbLTwbBe"  # 子任务 R11 heading

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", sec, "--detail", "full", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    print(r.stdout[:6000])
    sys.exit(0)
