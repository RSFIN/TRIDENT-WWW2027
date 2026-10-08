import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

def run(desc, args):
    print("### " + desc)
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        print(r.stdout[:2000])
        print(r.stderr[:1000])
        return r.returncode
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return 2

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "p"
    if mode == "p":
        run("insert simple paragraph", ["lark-cli", "docs", "+update", "--doc", doc,
            "--command", "block_insert_after", "--block-id", "doxcnLaX0JT5snjksADVeCpOMYd",
            "--content", "<p>PLACEHOLDER_TEST_P</p>", "--as", "user"])
    elif mode == "tbl":
        tiny = "<table><thead><tr><th><p>A</p></th><th><p>B</p></th></tr></thead><tbody><tr><td><p>1</p></td><td><p>2</p></td></tr></tbody></table>"
        run("insert tiny 2x2 table", ["lark-cli", "docs", "+update", "--doc", doc,
            "--command", "block_insert_after", "--block-id", "doxcnLaX0JT5snjksADVeCpOMYd",
            "--content", tiny, "--as", "user"])
