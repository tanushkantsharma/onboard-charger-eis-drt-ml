"""Turn pandoc's default reference.docx into an MDPI-like style sheet:
Palatino Linotype 10 pt justified body, 18 pt bold title, 10 pt bold section headings,
9 pt captions/tables, A4 page. Custom paragraph styles used by build_manuscript.py are added."""
import re, zipfile, shutil, sys
from pathlib import Path

HERE = Path(__file__).parent
SRC, DST = HERE / "ref_default.docx", HERE / "mdpi_reference.docx"
FONT = "Palatino Linotype"


def rpr(sz, bold=False, italic=False, color="000000"):
    return (f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>'
            + ("<w:b/><w:bCs/>" if bold else "") + ("<w:i/><w:iCs/>" if italic else "")
            + f'<w:color w:val="{color}"/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>')


def ppr(jc="both", before=0, after=0, line=260, ind_first=0, keep_next=False):
    return (f'<w:pPr>' + ("<w:keepNext/>" if keep_next else "")
            + f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
            + (f'<w:ind w:firstLine="{ind_first}"/>' if ind_first else "")
            + f'<w:jc w:val="{jc}"/></w:pPr>')


def style(sid, name, p, r, based="Normal"):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/><w:qFormat/>{p}{r}</w:style>')


def set_style(xml, sid, p, r):
    pat = re.compile(rf'(<w:style [^>]*w:styleId="{sid}"[^>]*>)(.*?)(</w:style>)', re.S)
    m = pat.search(xml)
    if not m:
        return xml
    inner = re.sub(r"<w:pPr>.*?</w:pPr>|<w:pPr/>", "", m.group(2), flags=re.S)
    inner = re.sub(r"<w:rPr>.*?</w:rPr>|<w:rPr/>", "", inner, flags=re.S)
    return xml[:m.start()] + m.group(1) + inner + p + r + m.group(3) + xml[m.end():]


def main():
    tmp = HERE / "_refdoc_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(SRC).extractall(tmp)
    st = (tmp / "word/styles.xml").read_text()
    body = (ppr("both", 0, 0, 260), rpr(20))
    for sid in ["Normal", "BodyText", "FirstParagraph", "Compact"]:
        st = set_style(st, sid, *body)
    st = set_style(st, "Title", ppr("left", 0, 240, 240), rpr(36, bold=True))
    st = set_style(st, "Heading1", ppr("left", 240, 120, 260, keep_next=True), rpr(20, bold=True))
    st = set_style(st, "Heading2", ppr("left", 120, 60, 260, keep_next=True), rpr(20, italic=True))
    st = set_style(st, "Heading3", ppr("left", 120, 60, 260, keep_next=True), rpr(20))
    st = set_style(st, "Caption", ppr("both", 120, 240, 228), rpr(18))
    st = set_style(st, "ImageCaption", ppr("both", 120, 240, 228), rpr(18))
    st = set_style(st, "TableCaption", ppr("both", 240, 60, 228, keep_next=True), rpr(18))
    st = set_style(st, "Bibliography", ppr("left", 0, 0, 228), rpr(18))
    st = set_style(st, "Figure", ppr("center", 120, 0, 240, keep_next=True), rpr(20))
    st = set_style(st, "CaptionedFigure", ppr("center", 120, 0, 240, keep_next=True), rpr(20))
    st = set_style(st, "Hyperlink", "", rpr(18, color="1F4E79"))
    extra = "".join([
        style("ArticleType", "Article Type", ppr("left", 0, 120, 240), rpr(20, italic=True)),
        style("Authors", "Authors", ppr("left", 0, 120, 260), rpr(20, bold=True)),
        style("Affiliations", "Affiliations", ppr("left", 0, 0, 228), rpr(16)),
        style("AbstractMDPI", "Abstract MDPI", ppr("both", 240, 120, 260), rpr(18)),
        style("BackMatter", "Back Matter", ppr("both", 60, 60, 228), rpr(18)),
        style("TableText", "Table Text", ppr("center", 0, 0, 228), rpr(16)),
        style("RefList", "Reference List", ppr("left", 0, 40, 228), rpr(18)),
    ])
    st = st.replace("</w:styles>", extra + "</w:styles>")
    # table body font 8 pt via table style
    st = re.sub(r'(<w:style w:type="table" w:default="1" w:styleId="Table">.*?)(</w:style>)',
                lambda m: m.group(1).replace("<w:rPr>", "<w:rPr><w:sz w:val=\"16\"/><w:szCs w:val=\"16\"/>", 1) + m.group(2)
                if "<w:rPr>" in m.group(1) else m.group(1) + rpr(16) + m.group(2), st, flags=re.S)
    (tmp / "word/styles.xml").write_text(st)
    doc = (tmp / "word/document.xml").read_text()
    sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1417" w:right="1531" w:bottom="1077" '
            'w:left="1531" w:header="1020" w:footer="850" w:gutter="0"/><w:lnNumType w:countBy="1" w:restart="continuous"/>'
            '</w:sectPr>')   # A4, MDPI-like margins, continuous line numbers (requested by most reviewers)
    doc = re.sub(r"<w:sectPr.*?</w:sectPr>|<w:sectPr/>", "", doc, flags=re.S)
    doc = doc.replace("</w:body>", sect + "</w:body>")
    (tmp / "word/document.xml").write_text(doc)
    if DST.exists():
        DST.unlink()
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tmp.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(tmp))
    shutil.rmtree(tmp)
    print("wrote", DST)


if __name__ == "__main__":
    main()
