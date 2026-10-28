"""Compare formatting fingerprints of the generated document against the sample document."""

import sys
import zipfile
from collections import Counter

from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}
W = "{%s}" % NS["w"]


def attrs(el):
    return tuple(sorted((k.replace(W, ""), v) for k, v in el.attrib.items())) if el is not None else None


def run_sig(r):
    rpr = r.find("w:rPr", NS)
    if rpr is None:
        return None
    parts = []
    for c in rpr:
        parts.append((c.tag.replace(W, ""), attrs(c)))
    return tuple(parts)


def para_sig(p):
    ppr = p.find("w:pPr", NS)
    sig = []
    if ppr is not None:
        for c in ppr:
            if c.tag == W + "rPr":
                continue
            sig.append((c.tag.replace(W, ""), attrs(c), tuple((x.tag.replace(W, ""), attrs(x)) for x in c)))
    kind = "IMG" if p.find(".//w:drawing", NS) is not None else "TXT"
    runs = []
    for child in p:
        if child.tag == W + "r":
            runs.append(run_sig(child))
        elif child.tag == W + "hyperlink":
            for r in child.findall("w:r", NS):
                runs.append(("LINK", run_sig(r)))
    # unique run signatures in order of first appearance
    seen, uniq = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return (kind, tuple(sig), tuple(uniq) if kind == "TXT" else ())


def tbl_sig(t):
    tblpr = t.find("w:tblPr", NS)
    grid = [g.get(W + "w") for g in t.find("w:tblGrid", NS)]
    rows = t.findall("w:tr", NS)
    first_cell = rows[0].find("w:tc", NS)
    tcpr = first_cell.find("w:tcPr", NS)
    shd = tcpr.find("w:shd", NS)
    mar = tcpr.find("w:tcMar", NS)
    hdr = rows[0].find("w:trPr/w:tblHeader", NS) is not None
    p = first_cell.find("w:p", NS)
    return (
        etree.tostring(tblpr).decode().split(">", 1)[1],
        len(grid),
        attrs(shd),
        tuple(attrs(c) for c in mar),
        hdr,
        para_sig(p),
    )


def load(path):
    z = zipfile.ZipFile(path)
    root = etree.fromstring(z.read("word/document.xml"))
    body = root.find("w:body", NS)
    paras, tbls = Counter(), Counter()
    for el in body:
        if el.tag == W + "p":
            paras[para_sig(el)] += 1
        elif el.tag == W + "tbl":
            tbls[tbl_sig(el)] += 1
            # body rows too
            for r in el.findall("w:tr", NS)[1:2]:
                tc = r.find("w:tc", NS)
                tbls[("ROW1", attrs(tc.find("w:tcPr/w:shd", NS)), para_sig(tc.find("w:p", NS)))] += 1
    sect = body.find("w:sectPr", NS)
    return paras, tbls, etree.tostring(sect).decode().split(">", 1)[1] if sect is not None else None, z


sample = sys.argv[1]
mine = sys.argv[2]
sp, st, ss, sz = load(sample)
mp, mt, ms, mz = load(mine)

print("== paragraph signatures: sample", len(sp), "| mine", len(mp))
only_mine = [s for s in mp if s not in sp]
print("signatures in mine NOT in sample:", len(only_mine))
for s in only_mine:
    print("  ", mp[s], "x", s)
only_sample = [s for s in sp if s not in mp]
print("signatures in sample NOT used in mine:", len(only_sample))
for s in only_sample:
    print("  ", sp[s], "x", str(s)[:1800])

print("\n== table signatures: sample", len(st), "| mine", len(mt))
only_mine_t = [s for s in mt if s not in st]
print("table signatures in mine NOT in sample:", len(only_mine_t))
for s in only_mine_t:
    print("  ", str(s)[:300])

print("\nsectPr identical:", ss == ms)

# package-level parity of non-body parts
for part in ["word/styles.xml", "word/numbering.xml", "word/settings.xml", "word/header1.xml",
             "word/_rels/header1.xml.rels", "word/fontTable.xml", "[Content_Types].xml"]:
    print(f"{part}: identical = {sz.read(part) == mz.read(part)}")
for logo in ["word/media/58309ee97eee869f9f05bdc53b047a687d142f7b.png", "word/media/9d2522f1d247c37d592a1cd06ff3f72c814c2c4c.png"]:
    print(f"{logo.split('/')[-1][:8]} logo identical = {sz.read(logo) == mz.read(logo)}")
