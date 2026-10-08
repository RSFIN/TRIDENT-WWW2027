import subprocess, sys

doc = "WtUqdMM1IoQJTGxQIwwcJiM9ndc"

if __name__ == "__main__":
    r = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", doc, "--detail", "simple", "--as", "user"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    text = r.stdout
    with open("docs/full_dump_v2.txt", "w", encoding="utf-8") as f:
        f.write(text)
    # verify top-level structure around result chapter
    i = text.find("实验结果汇总")
    j = text.find("P3 消融执行结果")
    print("===== 结构检查 =====")
    print("实验结果汇总 idx:", i, " P3 idx:", j)
    # check no table wrapper around result chapter: look backward from 实验结果汇总 for </table>
    seg = text[:i]
    last_tbl_close = seg.rfind("</table>")
    last_tbl_open = seg.rfind("<table")
    print("last table close before 实验结果汇总:", last_tbl_close, " open:", last_tbl_open)
    print("fragment between them:", seg[last_tbl_close:last_tbl_close+80] if last_tbl_close >= 0 else "n/a")
    # P3 heading position relative to result chapter end
    print(text[i-150:i+120])
    print("---- around P3 ----")
    print(text[j-300:j+150])
    sys.exit(0)
