import json, re
d = json.load(open(r'E:\PY-Project\WWW2027\code\intent-conformal\docs\_f13.txt', encoding='utf-8-sig'))
c = d['data']['document']['content']
for m in re.finditer(r'<img[^>]*>', c):
    if 'Fig 3' in m.group():
        print(re.search(r'id="([^"]+)"', m.group()).group(1))
