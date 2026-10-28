"""XML builders that reproduce the sample document's exact paragraph, table and image patterns."""

from xml.sax.saxutils import escape

FONT = "Cambria"
MONO = "Consolas"
INK = "1c1c1b"
BROWN = "574239"
LINK = "0563C1"

EMU_PER_INCH = 914400


def _fonts(name):
    return f'<w:rFonts w:ascii="{name}" w:cs="{name}" w:eastAsia="{name}" w:hAnsi="{name}"/>'


def run(text, *, bold=False, italic=False, size=22, color=INK, font=FONT, underline=False):
    rpr = _fonts(font)
    if bold:
        rpr += "<w:b/><w:bCs/>"
    if italic:
        rpr += "<w:i/><w:iCs/>"
    rpr += f'<w:color w:val="{color}"/>'
    if underline:
        rpr += '<w:u w:val="single"/>'
    rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def hyperlink(rid, text, size=22):
    inner = run(text, size=size, color=LINK, underline=True)
    return f'<w:hyperlink r:id="{rid}" w:history="1">{inner}</w:hyperlink>'


def _para(ppr, inner):
    return f"<w:p><w:pPr>{ppr}</w:pPr>{inner}</w:p>"


def title(text):
    return _para(
        '<w:spacing w:after="320" w:before="240"/><w:jc w:val="center"/>',
        run(text, bold=True, size=40),
    )


def label(text):
    return _para('<w:spacing w:after="160"/>', run(text, bold=True, size=28))


def body(*parts):
    """Body paragraph. Each part is either a str (plain run), ('b', str) bold run, or raw XML starting with '<'."""
    inner = ""
    for p in parts:
        if isinstance(p, tuple):
            inner += run(p[1], bold=True)
        elif p.startswith("<"):
            inner += p
        else:
            inner += run(p)
    return _para('<w:spacing w:after="160" w:line="276"/>', inner)


def note(lead, text):
    """Grey 9.5 pt note: bold lead-in followed by italic text (the sample's governance / timing notes)."""
    return _para(
        '<w:spacing w:after="160" w:line="276"/>',
        run(lead, bold=True, size=19, color="555555") + run(text, italic=True, size=19, color="555555"),
    )


def h1(text, page_break=False):
    brk = "<w:pageBreakBefore/>" if page_break else ""
    return _para(
        f'<w:pStyle w:val="Heading1"/><w:keepNext/>{brk}<w:spacing w:after="120" w:before="320"/>',
        run(text, bold=True, size=30),
    )


def h3(text):
    return _para(
        '<w:pStyle w:val="Heading3"/><w:keepNext/><w:spacing w:after="80" w:before="240"/>',
        run(text, bold=True, size=26, color=BROWN),
    )


def bullet(*parts):
    inner = ""
    for p in parts:
        if isinstance(p, tuple):
            inner += run(p[1], bold=True)
        elif p.startswith("<"):
            inner += p
        else:
            inner += run(p)
    return _para(
        '<w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
        '<w:spacing w:after="90" w:line="270"/>',
        inner,
    )


def source_label(text):
    return _para('<w:spacing w:after="80" w:before="160"/>', run(text))


def caption(text):
    return _para(
        '<w:spacing w:after="240" w:before="80"/><w:jc w:val="center"/>',
        run(text, italic=True, size=19, color=BROWN),
    )


def image(rid, name, descr, doc_id, width_in, height_in):
    cx, cy = int(width_in * EMU_PER_INCH), int(height_in * EMU_PER_INCH)
    d = escape(descr, {'"': "&quot;"})
    drawing = (
        "<w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent t="0" r="0" b="0" l="0"/>'
        f'<wp:docPr id="{doc_id}" name="{name}" descr="{d}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{doc_id}" name="{name}" descr="{d}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>"
    )
    return _para(
        '<w:keepNext/><w:spacing w:after="0" w:before="120"/><w:jc w:val="center"/>',
        f"<w:r>{drawing}</w:r>",
    )


_BORDERS = (
    '<w:tblBorders><w:top w:val="single" w:color="auto" w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/>'
    '<w:bottom w:val="single" w:color="auto" w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/>'
    '<w:insideH w:val="single" w:color="auto" w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/></w:tblBorders>'
)


def _tbl(widths, rows_xml):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    return f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="9360"/>{_BORDERS}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{rows_xml}</w:tbl>'


def _tc(width, fill, mar, inner):
    top, side = mar
    return (
        f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{width}"/><w:shd w:fill="{fill}" w:color="auto" w:val="clear"/>'
        f'<w:tcMar><w:top w:type="dxa" w:w="{top}"/><w:left w:type="dxa" w:w="{side}"/>'
        f'<w:bottom w:type="dxa" w:w="{top}"/><w:right w:type="dxa" w:w="{side}"/></w:tcMar></w:tcPr>{inner}</w:tc>'
    )


def data_table(widths, header, rows):
    """Header row (edebe6, bold 10pt) plus body rows (fbf9f4, 9.5pt). A cell may be a str or a list of str lines."""
    hdr = "".join(
        _tc(w, "edebe6", (100, 150), f"<w:p>{run(t, bold=True, size=20)}</w:p>") for w, t in zip(widths, header)
    )
    out = f"<w:tr><w:trPr><w:tblHeader/></w:trPr>{hdr}</w:tr>"
    for row in rows:
        cells = ""
        for w, cell in zip(widths, row):
            lines = cell if isinstance(cell, list) else [cell]
            paras = ""
            for i, line in enumerate(lines):
                after = "0" if i == len(lines) - 1 else "60"
                paras += f'<w:p><w:pPr><w:spacing w:after="{after}"/></w:pPr>{run(line, size=19)}</w:p>'
            cells += _tc(w, "fbf9f4", (80, 120), paras)
        out += f"<w:tr>{cells}</w:tr>"
    return _tbl(widths, out)


def code_table(lines):
    paras = ""
    for line in lines:
        paras += (
            '<w:p><w:pPr><w:spacing w:after="0" w:before="0" w:line="252"/></w:pPr>'
            f"{run(line, size=17, font=MONO)}</w:p>"
        )
    cell = _tc(9360, "f5f3ee", (100, 160), paras)
    return _tbl([9360], f"<w:tr>{cell}</w:tr>")
