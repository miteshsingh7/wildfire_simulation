from lxml import etree
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W="{%s}"%NS["w"]
tree=etree.parse("/tmp/sample/x/word/document.xml")
body=tree.getroot().find("w:body",NS)
for i,el in enumerate(body):
    if el.tag==W+"p":
        t="".join(x.text or "" for x in el.iter(W+"t"))
        if t.strip(): print("[%d] %s\n"%(i,t))
    elif el.tag==W+"tbl":
        for ri,tr in enumerate(el.findall("w:tr",NS)):
            cells=[]
            for tc in tr.findall("w:tc",NS):
                cells.append(" / ".join("".join(x.text or "" for x in p.iter(W+"t")) for p in tc.findall("w:p",NS)))
            print("  [%d.r%d] %s"%(i,ri," || ".join(cells)))
        print()
