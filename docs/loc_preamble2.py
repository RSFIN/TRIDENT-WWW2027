import subprocess, sys, re

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
section = "doxcncdVCnZz0qrycyfxp15lbVg"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", section, "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    txt = r.stdout
    # 找前导段（含 蓝图总览见下图）
    for m in re.finditer(r'<p id="(doxcn[^"]+)"[^>]*>([^<]{0,60})', txt):
        print(m.group(1), "|", m.group(2)[:50])
    sys.exit(0)
