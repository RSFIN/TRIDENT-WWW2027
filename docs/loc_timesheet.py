import subprocess, sys, re, json

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"
sec = "doxcnSVGtvkgADi7ICoRggFESjc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--scope", "section",
                        "--start-block-id", sec, "--detail", "with-ids", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    txt = r.stdout
    ids = re.findall(r'<(?:h[1-9]|p|table|html5-block|ul|ol|callout|blockquote|grid)[^>]* id="(doxcn[^"]+)"', txt)
    print("blocks in 写作时间表 section (top-level):")
    for i in ids:
        print(i)
    print("--- last id:", ids[-1] if ids else "NONE")
    sys.exit(0)
