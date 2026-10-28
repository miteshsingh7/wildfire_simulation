import re
from lxml import etree
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
tree=etree.parse("/tmp/sample/x/word/document.xml")
body=tree.getroot().find("w:body",NS)
els=list(body)
def clean(e):
    s=etree.tostring(e).decode()
    return re.sub(r' xmlns:\w+="[^"]+"','',s)
def tbl_rows(i, rows, maxc=3500):
    t=els[i]
    print("---- TABLE [%d] ----"%i)
    print(clean(t.find("w:tblPr",NS)))
    print(clean(t.find("w:tblGrid",NS)))
    trs=t.findall("w:tr",NS)
    for r in rows:
        print("-- row", r); print(clean(trs[r])[:maxc]); print()
tbl_rows(26,[0,1],4500)
tbl_rows(33,[0],3000)
tbl_rows(141,[0,1,10],2600)
# header
h=etree.parse("/tmp/sample/x/word/header1.xml")
s=re.sub(r' xmlns:\w+="[^"]+"','',etree.tostring(h).decode())
print("---- HEADER ----"); print(s)
# sectPr + hyperlinks rels
print(open("/tmp/sample/x/word/_rels/document.xml.rels").read())
