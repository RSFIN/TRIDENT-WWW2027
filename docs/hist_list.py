import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

def run(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    print(r.stdout[:4000])
    print(r.stderr[:2000])
    return r.returncode

if __name__ == "__main__":
    # list history to find revision 135 record
    sys.exit(run(["lark-cli", "docs", "+history-list", "--doc", doc, "--page-size", "20"]))
