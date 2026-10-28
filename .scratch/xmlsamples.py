import re
from lxml import etree
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W="{%s}"%NS["w"]
tree=etree.parse("/tmp/sample/x/word/document.xml")
body=tree.getroot().find("w:body",NS)
els=list(body)
def show(i, maxc=2200):
    s=etree.tostring(els[i]).decode()
    s=re.sub(r' xmlns:\w+="[^"]+"','',s)
    print("---- [%d] ----"%i); print(s[:maxc]); print()
for i in [0,1,2,4,5,6,13,19,29,30,32,36,37,125,126]:
    show(i)
