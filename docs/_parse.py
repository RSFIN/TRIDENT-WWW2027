import re
t = open('docs/_f10.txt', encoding='utf-8').read()
for m in re.finditer(r'<img[^>]*>', t):
    tag = m.group()
    bid = re.search(r'id=\\"([^\\]+)\\"', tag)
    cap = re.search(r'caption=\\"([^\\]{0,40})', tag)
    if bid and cap and 'Fig 3' in cap.group(1):
        print(bid.group(1))
