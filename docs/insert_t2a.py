import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
content_path = r"E:\PY-Project\WWW2027\code\intent-conformal\docs\t2a_table.xml"

if __name__ == "__main__":
    args = ["lark-cli", "docs", "+update", "--doc", doc, "--command", "block_insert_after",
            "--block-id", "doxcnLaX0JT5snjksADVeCpOMYd",
            "--content", "@" + content_path, "--as", "user"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        print("=== STDOUT ===")
        print(r.stdout[:4000])
        print("=== STDERR ===")
        print(r.stderr[:2000])
        sys.exit(r.returncode)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        sys.exit(2)
