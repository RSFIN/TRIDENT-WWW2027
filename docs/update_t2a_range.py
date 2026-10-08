import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
content_path = r"E:\PY-Project\WWW2027\code\intent-conformal\docs\t2a_table.xml"

if __name__ == "__main__":
    args = ["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_replace",
            "--start-block-id", "doxcn0KDp3OHfmSLmdV4eipcsCe",
            "--end-block-id", "doxcn0KDp3OHfmSLmdV4eipcsCe",
            "--content", "@" + content_path, "--as", "user"]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print("=== STDOUT ===")
    print(r.stdout[:3000])
    print("=== STDERR ===")
    print(r.stderr[:2000])
    sys.exit(r.returncode)
