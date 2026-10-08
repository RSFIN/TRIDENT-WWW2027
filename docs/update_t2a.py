import subprocess, sys, os

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
content_path = r"E:\PY-Project\WWW2027\code\intent-conformal\docs\t2a_table.xml"

def run_update(cmd, block_id, content_arg):
    args = ["lark-cli", "docs", "+update", "--doc", doc, "--command", cmd,
            "--block-id", block_id, "--content", content_arg, "--as", "user"]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("=== STDOUT ===")
    print(r.stdout[:3000])
    print("=== STDERR ===")
    print(r.stderr[:2000])
    return r.returncode

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "t2a"
    if mode == "t2a":
        rc = run_update("block_replace", "doxcn0KDp3OHfmSLmdV4eipcsCe", "@" + content_path)
        sys.exit(rc)
