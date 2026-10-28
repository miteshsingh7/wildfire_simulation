import sys
from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
W = "{%s}" % NS["w"]


def text_of(el):
    out = []
    for t in el.iter():
        if t.tag == W + "t":
            out.append(t.text or "")
        elif t.tag == W + "tab":
            out.append("\t")
        elif t.tag == W + "br":
            out.append("<BR:%s>" % t.get(W + "type", "line"))
    return "".join(out)


def rpr_str(r):
    rpr = r.find("w:rPr", NS)
    d = []
    if rpr is not None:
        for c in rpr:
            tag = c.tag.replace(W, "")
            val = c.get(W + "val")
            d.append(tag + ("=" + val if val else ""))
    return ",".join(d)


def run_fmt(p):
    fm = []
    for r in p.findall("w:r", NS):
        t = text_of(r)
        if t.strip():
            fm.append((t[:30], rpr_str(r)))
    return fm


def para_info(p):
    ppr = p.find("w:pPr", NS)
    d = {}
    if ppr is not None:
        for c in ppr:
            tag = c.tag.replace(W, "")
            if tag == "rPr":
                continue
            attrs = {k.replace(W, ""): v for k, v in c.attrib.items()}
            sub = [x.tag.replace(W, "") + str({k.replace(W, ""): v for k, v in x.attrib.items()}) for x in c]
            d[tag] = (attrs, sub)
    return d


def img_info(p):
    res = []
    for dr in p.iter(W + "drawing"):
        ext = dr.find(".//wp:extent", NS)
        blip = dr.find(".//a:blip", NS)
        docpr = dr.find(".//wp:docPr", NS)
        res.append((ext.get("cx"), ext.get("cy"), blip.get("{%s}embed" % NS["r"]), docpr.get("name") if docpr is not None else None))
    return res


def dump_table(tbl, indent=""):
    tblpr = tbl.find("w:tblPr", NS)
    print(indent + "TABLE tblPr:", etree.tostring(tblpr).decode().split(">", 1)[1][:700] if tblpr is not None else None)
    grid = tbl.find("w:tblGrid", NS)
    print(indent + "  grid:", [g.get(W + "w") for g in grid] if grid is not None else None)
    for ri, tr in enumerate(tbl.findall("w:tr", NS)):
        trpr = tr.find("w:trPr", NS)
        cells = []
        for tc in tr.findall("w:tc", NS):
            tcpr = tc.find("w:tcPr", NS)
            shd = tcpr.find("w:shd", NS) if tcpr is not None else None
            fill = shd.get(W + "fill") if shd is not None else None
            tcw = tcpr.find("w:tcW", NS) if tcpr is not None else None
            w = tcw.get(W + "w") if tcw is not None else None
            gs = tcpr.find("w:gridSpan", NS) if tcpr is not None else None
            imgs = img_info(tc)
            cells.append("[%s|w=%s|span=%s]%s %s" % (fill, w, gs.get(W + "val") if gs is not None else 1, " IMG" + str(imgs) if imgs else "", text_of(tc)[:80].replace("\n", " / ")))
        extra = ""
        if trpr is not None and ri < 2:
            extra = " trPr=" + etree.tostring(trpr).decode().split(">", 1)[1][:150]
        print(indent + "  row%d%s:" % (ri, extra), " || ".join(cells))


def main():
    path = sys.argv[1]
    tree = etree.parse(path)
    body = tree.getroot().find("w:body", NS)
    i = 0
    for el in body:
        tag = el.tag.replace(W, "")
        if tag == "p":
            ppr = para_info(el)
            style = ppr.get("pStyle", ({}, []))[0].get("val") if "pStyle" in ppr else "-"
            imgs = img_info(el)
            t = text_of(el)
            print("%3d P style=%s %s%s | %r" % (i, style, {k: v for k, v in ppr.items() if k != "pStyle"}, (" IMG=" + str(imgs)) if imgs else "", t[:120]))
            if "--fmt" in sys.argv:
                print("      fmt:", run_fmt(el)[:4])
        elif tag == "tbl":
            print("%3d " % i, end="")
            dump_table(el)
        elif tag == "sectPr":
            print("%3d SECTPR: %s" % (i, etree.tostring(el).decode().split(">", 1)[1][:900]))
        else:
            print("%3d %s" % (i, tag))
        i += 1


main()
