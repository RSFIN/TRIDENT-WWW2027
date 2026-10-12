import re
t = open('docs/_f8.txt', encoding='utf-8').read()
for m in re.finditer(r'<img[^>]*>', t):
    tag = m.group()
    bid = re.search(r'id=\\"([^\\]+)\\"', tag)
    cap = re.search(r'caption=\\"([^\\]{0,40})', tag)
    if bid and cap:
        print(bid.group(1), '||', cap.group(1))
