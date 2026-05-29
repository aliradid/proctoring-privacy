"""Build the manuscript as a Word document conforming to the Informatica
(Slovenia) Informatica2026.dotx template.

The Informatica template defines a fixed set of named paragraph styles
(I_Title, I_Authors, I_Abstract, I_Keywords, I_SectionTitle,
I_SubSectionTitle, I_Text, I_FigureCaption, I_TableCaption, I_References,
I_Bibliography). We open the .dotx as a document, clear it, and emit the
content of `manuscript/manuscript.md` using those styles directly so the
result respects the journal's typography out of the box.

Run: `python3 code/build_informatica.py`
Output: `manuscript/Article_Informatica.docx`
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "manuscript" / "Informatica2026.dotx"
MS_PATH = ROOT / "manuscript" / "manuscript.md"
OUT = ROOT / "manuscript" / "Article_Informatica.docx"
FIG_DIR = ROOT / "figures"

# Same figure insertion plan as before, but with Informatica-style captions
# Single-column width (≈ 3.15"). Wide content that does not fit will be wrapped
# in a one-column section break (see `_open_one_column` / `_open_two_columns`).
FIG_WIDTH_INCH = 3.15
WIDE_FIG_WIDTH_INCH = 6.4
# Tables that have these many or more columns are emitted in a one-column
# section so they can use the full page width.
WIDE_TABLE_COL_THRESHOLD = 4


def _new_section_break(doc, num_cols: int):
    """Append a continuous section break that switches the column count.

    Word renders this as the document continuing on the same page but with a
    new column setting.
    """
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    sect_pr = OxmlElement("w:sectPr")
    # continuous section break
    type_el = OxmlElement("w:type")
    type_el.set(qn("w:val"), "continuous")
    sect_pr.append(type_el)
    pgsz = OxmlElement("w:pgSz")
    pgsz.set(qn("w:w"), "11906")
    pgsz.set(qn("w:h"), "16838")
    sect_pr.append(pgsz)
    pgmar = OxmlElement("w:pgMar")
    pgmar.set(qn("w:top"), "1644")
    pgmar.set(qn("w:right"), "1077")
    pgmar.set(qn("w:bottom"), "1134")
    pgmar.set(qn("w:left"), "1304")
    pgmar.set(qn("w:header"), "936")
    pgmar.set(qn("w:footer"), "720")
    sect_pr.append(pgmar)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), "284")
    sect_pr.append(cols)
    pPr.append(sect_pr)


# Figures are placed by explicit [[FIG:key]] markers in the markdown, which is
# robust to prose edits. Each marker is replaced (in document order) by the
# image + an auto-numbered caption. Captions are keyed here.
FIGURE_CAPTIONS = {
    "architecture": "End-to-end privacy-preserving proctoring pipeline.",
    "si_timeline": "Simulated 60-second session illustrating the trajectory of the visual, acoustic and behavioural risk components and the resulting Suspicion Index.",
    "visual_pr_curves": "Precision-recall curves of YOLOv8-s on the COCO val2017 proctoring subset (217 images, 1,164 annotations).",
    "visual_tp_fp": "Per-class true-positive, false-positive and false-negative counts after greedy IoU 0.5 matching.",
    "acoustic_roc_pr": "ROC and PR curves for the Whisper-Base content-free acoustic module on the LibriSpeech-derived benchmark.",
    "acoustic_per_class": "Per-class distribution of the secondary-speaker probability on the 400-clip LibriSpeech benchmark.",
    "behaviour_distributions": "Per-scenario distribution of the geometric descriptors (yaw amplitude and lip variance) computed from MediaPipe FaceMesh landmarks of real Pexels face anchors.",
    "fusion_comparison": "Cross-validated F1 of the five fusion strategies on real-modality features (error bars = standard deviation over 5 folds).",
    "fusion_roc": "Pooled 5-fold cross-validation ROC curves of all fusion strategies on real-modality features.",
    "fusion_confusion": "Confusion matrix of the random-forest fusion model under 5-fold cross-validation on the 300 real-feature sessions.",
    "fusion_ablation": "Ablation of the random-forest fusion model: leave-one-modality-out (upper) and single-modality (lower).",
    "latency": "Per-module inference latency on the Apple M4 Max host (Mac Studio, 14-core CPU, 32-core integrated GPU, 36 GB unified memory) with both MPS GPU acceleration and CPU-only configurations.",
    "robustness": "Robustness study: class-presence recall versus gamma and scale for the visual subsystem, and F1 plus score distributions versus SNR for the acoustic subsystem under real LibriSpeech noise injection.",
    "privacy": "Adversarial privacy verification on 30 real LibriSpeech speakers: re-identification AUC achieved by a raw-audio attacker (proxy of conventional proctoring) versus a metadata-only attacker (the proposed system).",
}
# Figures that must span the full page width (rendered in a 1-column section).
WIDE_FIGURE_KEYS = {
    "architecture", "si_timeline", "acoustic_roc_pr", "fusion_roc",
    "latency", "robustness",
}
FIG_MARKER_RE = re.compile(r"^\s*\[\[FIG:([a-z_]+)\]\]\s*$")


def _apply_style(p, style_name: str, doc):
    """Apply a named style; fall back to Normal if the style is missing."""
    try:
        p.style = doc.styles[style_name]
    except KeyError:
        p.style = doc.styles["Normal"]


def _add_run(p, text: str, bold: bool = False, italic: bool = False):
    r = p.add_run(text)
    if bold:
        r.font.bold = True
    if italic:
        r.font.italic = True
    return r


def _add_text_paragraph(doc, text: str, style: str = "I_Text"):
    p = doc.add_paragraph()
    _apply_style(p, style, doc)
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            _add_run(p, part[2:-2], bold=True)
        elif part.startswith("*") and part.endswith("*"):
            _add_run(p, part[1:-1], italic=True)
        else:
            _add_run(p, part)
    return p


def _add_figure(doc, key: str, fig_no: int):
    fig_name = f"fig_{key}.png"
    fig_path = FIG_DIR / fig_name
    caption_text = FIGURE_CAPTIONS.get(key, "")
    if not fig_path.exists():
        print(f"  WARNING missing figure {fig_name}")
        return
    wide = key in WIDE_FIGURE_KEYS
    if wide:
        # Close the surrounding 2-column body section.
        _new_section_break(doc, 2)
    p = doc.add_paragraph()
    _apply_style(p, "I_Figure", doc)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    width = Inches(WIDE_FIG_WIDTH_INCH if wide else FIG_WIDTH_INCH)
    p.add_run().add_picture(str(fig_path), width=width)
    cap = doc.add_paragraph()
    _apply_style(cap, "I_FigureCaption", doc)
    _add_run(cap, f"Figure {fig_no}: ", bold=True)
    _add_run(cap, caption_text)
    if wide:
        # Close the wide 1-column section that contained this figure.
        _new_section_break(doc, 1)


def _add_table_md(doc, header, rows, table_no: int, title: str):
    wide = len(header) >= WIDE_TABLE_COL_THRESHOLD
    if wide:
        _new_section_break(doc, 2)
    # Caption above
    cap = doc.add_paragraph()
    _apply_style(cap, "I_TableCaption", doc)
    _add_run(cap, f"Table {table_no}: ", bold=True)
    _add_run(cap, title)
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    try:
        table.style = doc.styles["Table Grid"]
    except KeyError:
        # Apply manual borders if Table Grid isn't shipped with the template
        from docx.oxml import OxmlElement as _OE
        tbl_pr = table._tbl.tblPr
        borders = _OE("w:tblBorders")
        for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = _OE(f"w:{tag}")
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "4")
            b.set(qn("w:color"), "000000")
            borders.append(b)
        tbl_pr.append(borders)
    for j, h in enumerate(header):
        c = table.rows[0].cells[j]
        c.text = ""
        para = c.paragraphs[0]
        _apply_style(para, "I_Table", doc)
        _add_run(para, h.strip(), bold=True)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = table.rows[i].cells[j]
            c.text = ""
            para = c.paragraphs[0]
            _apply_style(para, "I_Table", doc)
            text = val.strip()
            is_bold = text.startswith("**") and text.endswith("**")
            if is_bold:
                text = text[2:-2]
            _add_run(para, text, bold=is_bold)
    if wide:
        _new_section_break(doc, 1)


def _parse_table(lines, idx):
    header_line = lines[idx]
    sep = lines[idx + 1]
    if not re.match(r"\s*\|.*\|\s*$", sep):
        return None
    header = [c.strip() for c in header_line.strip().strip("|").split("|")]
    rows = []
    i = idx + 2
    while i < len(lines) and re.match(r"\s*\|.*\|\s*$", lines[i]):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return header, rows, i


def render():
    work_path = OUT.with_suffix(".docx")
    # 1) Convert .dotx -> .docx by patching the content-type only.
    with zipfile.ZipFile(TEMPLATE, "r") as zin:
        data = {name: zin.read(name) for name in zin.namelist()}
    ct = data["[Content_Types].xml"].decode()
    ct = ct.replace(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
    )
    data["[Content_Types].xml"] = ct.encode()
    tmp = ROOT / "manuscript" / "_informatica_seed.docx"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, payload in data.items():
            zout.writestr(name, payload)

    # 2) Open in python-docx and remove every pre-existing body element so we
    #    keep the styles, page setup and section properties but start with a
    #    clean canvas.
    doc = Document(tmp)
    body = doc.element.body
    # Preserve the sectPr at the very end of the body
    sect_pr = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is sect_pr:
            continue
        body.remove(child)

    md = MS_PATH.read_text().splitlines()
    fig_no = 1
    table_no = 1
    i = 0
    in_authors = False
    title_done = False
    references_started = False
    body_two_columns_started = False

    while i < len(md):
        line = md[i]
        s = line.rstrip()

        # H1 — title
        if s.startswith("# "):
            p = doc.add_paragraph()
            _apply_style(p, "I_Title", doc)
            _add_run(p, s[2:].strip(), bold=True)
            title_done = True
            i += 1
            continue

        # Author block
        if s.startswith("**Radid Ali**"):
            p = doc.add_paragraph()
            _apply_style(p, "I_Authors", doc)
            _add_run(p, "Radid Ali, Ghazouani Mohamed and El habib Benlahmar")
            aff = doc.add_paragraph()
            _apply_style(aff, "I_Text", doc)
            _add_run(aff, "Department of Mathematics and Computer Science, Hassan II University, Faculty of Sciences Ben M'sik, Casablanca, Morocco", italic=True)
            mail = doc.add_paragraph()
            _apply_style(mail, "I_Text", doc)
            _add_run(mail, "E-mail: radidalix@outlook.fr, ghazouani.fsbm@gmail.com, h.benlahmer@gmail.com", italic=True)
            # skip until the `## Abstract` heading
            j = i + 1
            while j < len(md) and not md[j].strip().startswith("## Abstract"):
                j += 1
            i = j
            continue

        # Abstract handling: H2 "## Abstract" then a body paragraph; we render
        # the body as an I_Abstract paragraph (no separate heading line).
        if s.startswith("## Abstract"):
            # Consume the abstract body (the next non-empty line) as I_Abstract
            j = i + 1
            while j < len(md) and md[j].strip() == "":
                j += 1
            abstract_lines = []
            while j < len(md) and md[j].strip() and not md[j].strip().startswith(("##", "**Keywords**")):
                abstract_lines.append(md[j].strip())
                j += 1
            abstract_text = " ".join(abstract_lines)
            p = doc.add_paragraph()
            _apply_style(p, "I_Abstract", doc)
            _add_run(p, "Abstract. ", bold=True)
            _add_run(p, abstract_text)
            i = j
            continue
        if s.strip().startswith("**Keywords**"):
            p = doc.add_paragraph()
            _apply_style(p, "I_Keywords", doc)
            _add_run(p, "Keywords: ", bold=True)
            _add_run(p, s.strip()[len("**Keywords**"):].lstrip(" :—-"))
            # Slovenian abstract (Povzetek). The journal asks for both
            # languages; the translation below was prepared with
            # machine-translation assistance and should be reviewed by a
            # native Slovenian speaker before final acceptance.
            ps = doc.add_paragraph()
            _apply_style(ps, "I_Abstract_SI", doc)
            _add_run(ps, "Povzetek. ", bold=True)
            _add_run(ps, (
                "Hitra uporaba spletnih platform za ocenjevanje je pospešila uvajanje "
                "proctoring sistemov, ki temeljijo na umetni inteligenci. Večina komercialnih "
                "rešitev se zanaša na vsiljivo biometrično obdelavo — neprekinjeno prepoznavanje "
                "obrazov, izločanje glasovnih odtisov in prepis govora — kar je v nasprotju s "
                "strogimi predpisi o varstvu podatkov, kot sta evropska Splošna uredba o varstvu "
                "podatkov (GDPR) in maroška uredba CNDP, zakon 09-08, ki tovrstne podatke "
                "uvrščata med občutljive osebne podatke. V članku predstavljamo večmodalno "
                "ogrodje za nadzor izpitov, ki ohranja zasebnost in ne shranjuje biometričnih "
                "predlog, surovega videa ali jezikovne vsebine. Sistem povezuje YOLOv8-s za "
                "kontekstualno zaznavanje predmetov, MediaPipe FaceMesh za neidentificirajoče "
                "vedenjske namige (odklon pogleda, drža glave, gibanje ustnic) in Whisper-Base "
                "kodirnik v načinu brez prepisa govora. Vsi izhodi modulov so pretvorjeni v "
                "standardiziran tok metapodatkov in obdelani z modelom za izračun indeksa suma, "
                "katerega strategija je izbrana empirično med petimi različicami. Na realni "
                "podmnožici slik COCO val2017 (217 slik, 1.164 oznak) YOLOv8-s doseže makro F1 "
                "0,606 in povprečni AP@0,5 0,519. Pri 400 zvočnih posnetkih, sestavljenih iz "
                "korpusa LibriSpeech, kodirnik Whisper-Base v brezvsebinskem načinu doseže ROC "
                "AUC 0,774 za prisotnost dodatnega govorca (F1 = 0,78) in AUC 0,935 za "
                "zaznavanje šepetanja. Pri 5-kratnem prečnem preverjanju na 300 sejah s "
                "preverjenimi modalnostnimi izhodi se naključni gozd uvrsti kot najboljša "
                "strategija s F1 = 0,905 ± 0,041 (95-odstotni interval [0,876, 0,931]) in je "
                "statistično pomembno boljši od osnovnega pravila tehtane vsote (McNemar χ² = "
                "7,20, p = 7,3 × 10⁻³). Na napravi Apple M4 Max (Mac Studio) cevovod deluje pri "
                "29,5 sličicah na sekundo s pospeševanjem MPS in 30,0 sličicah na sekundo v "
                "načinu samo CPU. Sistem porabi največ 1.075 MB pomnilnika. Pri preverjanju "
                "zasebnosti z 30 resničnimi govorci napadalec s surovim zvokom doseže AUC 0,994 "
                "za ponovno identifikacijo, medtem ko enak napadalec, omejen na tok "
                "metapodatkov, doseže le 0,478 (pod naključno mejo). To kvantitativno potrjuje "
                "lastnost ogrodja, da ne shranjuje biometričnih podatkov, in ohrani skladnost z "
                "9. členom GDPR ter z zakonom 09-08 maroške CNDP."
            ))
            # End of title block: insert a continuous section break that makes
            # everything *before* it appear in one column. The final sectPr
            # at the end of the document keeps the body in two columns.
            _new_section_break(doc, 1)
            body_two_columns_started = True
            i += 1
            continue

        # Section / subsection headings
        if s.startswith("## ") and "References" not in s and "Acknowledgements" not in s and "Data and Code" not in s:
            heading = s[3:].strip()
            # The Informatica I_SectionTitle / I_SubSectionTitle styles
            # auto-number sections, so we strip the leading "N. " or "N.M "
            # so we don't render "7 7. Discussion".
            heading = re.sub(r"^\d+(?:\.\d+)?\.?\s+", "", heading)
            p = doc.add_paragraph()
            _apply_style(p, "I_SectionTitle", doc)
            _add_run(p, heading, bold=True)
            i += 1
            continue
        if s.startswith("## References"):
            p = doc.add_paragraph()
            # I_References is the template's UN-numbered back-matter heading
            # style; using I_SectionTitle here would auto-number it ("12").
            _apply_style(p, "I_References", doc)
            _add_run(p, "References", bold=True)
            references_started = True
            i += 1
            continue
        if s.startswith("## Acknowledgements") or s.startswith("## Data and Code Availability"):
            p = doc.add_paragraph()
            _apply_style(p, "I_References", doc)
            _add_run(p, s[3:].strip(), bold=True)
            i += 1
            continue
        if s.startswith("### "):
            heading = s[4:].strip()
            heading = re.sub(r"^\d+(?:\.\d+)?\.?\s+", "", heading)
            p = doc.add_paragraph()
            _apply_style(p, "I_SubSectionTitle", doc)
            _add_run(p, heading, bold=True)
            i += 1
            continue

        # Explicit figure marker: [[FIG:key]]
        fig_marker = FIG_MARKER_RE.match(line)
        if fig_marker:
            _add_figure(doc, fig_marker.group(1), fig_no)
            fig_no += 1
            i += 1
            continue

        # Tables
        if re.match(r"\s*\|.*\|\s*$", line):
            parsed = _parse_table(md, i)
            if parsed:
                header, rows, next_i = parsed
                # Look back for a "Table N. ..." caption line above
                title = ""
                for back in range(i - 1, max(0, i - 4), -1):
                    if md[back].strip().startswith("**Table"):
                        title = re.sub(r"^\*\*Table \d+\.\s*", "", md[back].strip()).rstrip("*").strip()
                        break
                _add_table_md(doc, header, rows, table_no, title)
                table_no += 1
                i = next_i
                continue

        # Equation blocks (single-line)
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            p = doc.add_paragraph()
            _apply_style(p, "I_Text", doc)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_run(p, s[2:-2].strip(), italic=True)
            i += 1
            continue
        if s.startswith("$$"):
            j = i + 1
            buf = []
            while j < len(md) and not md[j].strip().startswith("$$"):
                buf.append(md[j])
                j += 1
            p = doc.add_paragraph()
            _apply_style(p, "I_Text", doc)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_run(p, " ".join(buf).strip(), italic=True)
            i = j + 1
            continue

        # Bulleted / numbered lists
        if s.lstrip().startswith("- "):
            p = doc.add_paragraph()
            _apply_style(p, "I_Text", doc)
            text = "• " + s.lstrip()[2:]
            for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
                if not part:
                    continue
                if part.startswith("**") and part.endswith("**"):
                    _add_run(p, part[2:-2], bold=True)
                elif part.startswith("*") and part.endswith("*"):
                    _add_run(p, part[1:-1], italic=True)
                else:
                    _add_run(p, part)
            i += 1
            continue
        m_num = re.match(r"^(\d+)\.\s+(.*)", s.lstrip())
        if m_num:
            p = doc.add_paragraph()
            _apply_style(p, "I_Text", doc)
            _add_run(p, f"{m_num.group(1)}. ", bold=True)
            for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", m_num.group(2)):
                if not part:
                    continue
                if part.startswith("**") and part.endswith("**"):
                    _add_run(p, part[2:-2], bold=True)
                elif part.startswith("*") and part.endswith("*"):
                    _add_run(p, part[1:-1], italic=True)
                else:
                    _add_run(p, part)
            i += 1
            continue

        # References (start with [N]). We keep the literal [N] brackets so they
        # match the in-text [N] citations, and render each entry in the body
        # text style (I_References is a heading style, not an entry style).
        if references_started and re.match(r"^\[\d+\]", s.strip()):
            p = doc.add_paragraph()
            _apply_style(p, "I_Text", doc)
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", s.strip()):
                if not part:
                    continue
                if part.startswith("**") and part.endswith("**"):
                    _add_run(p, part[2:-2], bold=True)
                elif part.startswith("*") and part.endswith("*"):
                    _add_run(p, part[1:-1], italic=True)
                else:
                    _add_run(p, part)
            i += 1
            continue

        # Skip blank lines + table-caption-only lines (already handled)
        if s.strip() == "":
            i += 1
            continue
        if s.strip().startswith("**Table"):
            i += 1
            continue

        # Collapse soft-wrapped paragraph
        para_lines = [s]
        j = i + 1
        while j < len(md):
            nxt = md[j]
            if not nxt.strip() or nxt.lstrip().startswith(("#", "|", "- ", "$$", "```")) or FIG_MARKER_RE.match(nxt) or re.match(r"^\d+\.\s", nxt.strip()) or (references_started and re.match(r"^\[\d+\]", nxt.strip())):
                break
            para_lines.append(nxt)
            j += 1
        para_text = " ".join(p.strip() for p in para_lines)
        _add_text_paragraph(doc, para_text, style="I_Text")
        i = j

    doc.save(work_path)
    print(f"Wrote {work_path}")
    print(f"  ({fig_no - 1} figures, {table_no - 1} tables)")


if __name__ == "__main__":
    render()
