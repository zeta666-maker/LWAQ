"""Convert markdown reports and patent drafts into .docx.

This is the single converter for the project. It supersedes the earlier text-only
version; it additionally supports

  $$ ... $$        display math  -> Word native equation (OMML, editable)
  $ ... $          inline math   -> Word native equation, inline
  ![alt](path)     images        -> embedded picture, scaled to the text width
  <standalone $$>  equation body on the following line(s), optional \\tag{n} number

Everything else (headings, tables, lists, block quotes, code blocks, rules, bold,
italic, inline code) is handled as before.

Usage:
  # the patent draft (equations + figures)
  python tools/md_to_docx.py \
      --src docs_patent/航空燃油齿轮泵故障检测算法专利.md \
      --out docs_patent/航空燃油齿轮泵故障检测算法专利.docx

  # the report set (no equations, no figures)
  python tools/md_to_docx.py --batch reports

  # a single report
  python tools/md_to_docx.py --src REPORT_FOR_TEACHER.md --out docs_word/REPORT_FOR_TEACHER.docx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

CJK_FONT = "Microsoft YaHei"
LATIN_FONT = "Times New Roman"

# Reports handed to a supervisor, converted by --batch reports.
REPORT_FILES = [
    "REPORT_FOR_TEACHER.md",
    "CKAA_TRAINABLE_PARAMS.md",
    "REPLY_TO_TEACHER.md",
    "AQCL_2BIT_ANALYSIS.md",
    "PAPER_COMPARISON.md",
    "FINAL_RESULTS_BOTH_PAPERS.md",
]

INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*|\$[^$\n]+\$)")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
DISPLAY_MATH_RE = re.compile(r"^\s*\$\$(.+?)\$\$\s*$")


# --------------------------------------------------------------------------- fonts
def set_font(run, name: str = CJK_FONT, *, latin: str = LATIN_FONT) -> None:
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)


# ------------------------------------------------------------------------ equations
def latex_to_omml(latex: str):
    """Return an OMML element for the LaTeX, or None when conversion is unavailable."""
    latex = latex.strip()
    if not latex:
        return None
    try:
        from latex2mathml.converter import convert as latex_to_mathml
        from mathml2omml import convert as mathml_to_omml
    except ImportError:
        return None
    try:
        omml_string = mathml_to_omml(latex_to_mathml(latex))
    except Exception:  # noqa: BLE001 - fall back to plain text
        return None
    if not omml_string:
        return None
    try:
        from lxml import etree

        wrapped = (
            '<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"{omml_string}</root>"
        )
        root = etree.fromstring(
            wrapped.encode("utf-8"),
            parser=etree.XMLParser(recover=True, resolve_entities=False),
        )
    except Exception:  # noqa: BLE001
        return None
    for child in root:
        return child
    return None


def add_display_equation(document: Document, latex: str) -> bool:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    element = latex_to_omml(latex)
    if element is None:
        run = paragraph.add_run(latex)
        run.font.name = "Cambria Math"
        run.font.size = Pt(10.5)
        return False
    paragraph._p.append(element)
    return True


def add_inline_math(paragraph, latex: str) -> bool:
    element = latex_to_omml(latex)
    if element is None:
        run = paragraph.add_run(latex)
        run.font.name = "Cambria Math"
        run.font.size = Pt(10.5)
        return False
    paragraph._p.append(element)
    return True


# ----------------------------------------------------------------------------- text
def add_runs(paragraph, text: str, *, size: float | None = None) -> None:
    for token in INLINE_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            set_font(run)
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt((size or 10.5) - 0.5)
            run.font.color.rgb = RGBColor(0xA0, 0x30, 0x30)
        elif token.startswith("$") and token.endswith("$") and len(token) > 2:
            add_inline_math(paragraph, token[1:-1])
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            run = paragraph.add_run(token[1:-1])
            run.italic = True
            set_font(run)
        else:
            run = paragraph.add_run(token)
            set_font(run)
        if size:
            paragraph.runs[-1].font.size = Pt(size)


def add_image(document: Document, path: Path, *, max_width_cm: float = 15.5) -> bool:
    if not path.exists():
        paragraph = document.add_paragraph()
        add_runs(paragraph, f"[missing image: {path.name}]")
        return False
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Cm(max_width_cm))
    return True


def add_table(document: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    width = max(len(r) for r in rows)
    table = document.add_table(rows=0, cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for col_index in range(width):
            text = row[col_index] if col_index < len(row) else ""
            cell = cells[col_index]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(1)
            paragraph.paragraph_format.space_after = Pt(1)
            add_runs(paragraph, text, size=9)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                if row_index == 0:
                    run.bold = True
    document.add_paragraph()


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def add_horizontal_rule(document: Document) -> None:
    paragraph = document.add_paragraph()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    pbdr.append(bottom)
    paragraph._p.get_or_add_pPr().append(pbdr)


# ------------------------------------------------------------------------- converter
def convert(md_path: Path, docx_path: Path) -> dict:
    document = Document()
    normal = document.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.font.name = LATIN_FONT
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), CJK_FONT)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    stats = {"equations": 0, "equation_fallbacks": 0, "images": 0,
             "missing_images": 0, "tables": 0}

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(12)
            run = paragraph.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            continue

        # standalone image, optional caption from the alt text
        image_match = IMAGE_RE.fullmatch(stripped)
        if image_match:
            caption, target = image_match.group(1), image_match.group(2)
            ok = add_image(document, (md_path.parent / target).resolve())
            stats["images" if ok else "missing_images"] += 1
            if caption:
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(paragraph, caption, size=9.5)
                for run in paragraph.runs:
                    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            index += 1
            continue

        # display equation opened by a standalone "$$" line
        if stripped == "$$":
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                body.append(lines[index])
                index += 1
            index += 1
            # A source line ending in a comma continues the formula on the next line;
            # keep those as separate aligned rows so terms do not run together.
            parts = [part.strip() for part in body if part.strip()]
            latex = " ".join(parts)
            tag_match = re.search(r"\\tag\{([^}]*)\}", latex)
            tag = tag_match.group(1) if tag_match else ""
            latex = re.sub(r"\\tag\{[^}]*\}", "", latex).strip()
            if len(parts) > 1:
                rows = " \\\\ ".join(part.rstrip(",").strip() for part in parts)
                latex = f"\\begin{{aligned}} {rows} \\end{{aligned}}"
            ok = add_display_equation(document, latex)
            if tag:
                number = document.add_paragraph()
                number.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                add_runs(number, f"({tag})", size=10)
            stats["equations" if ok else "equation_fallbacks"] += 1
            continue

        # display equation entirely on one line
        math_match = DISPLAY_MATH_RE.match(stripped)
        if math_match:
            ok = add_display_equation(document, math_match.group(1))
            stats["equations" if ok else "equation_fallbacks"] += 1
            index += 1
            continue

        # table
        if stripped.startswith("|") and index + 1 < len(lines) and TABLE_SEP_RE.match(
            lines[index + 1]
        ):
            rows = [split_row(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_row(lines[index]))
                index += 1
            add_table(document, rows)
            stats["tables"] += 1
            continue

        # heading
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = document.add_heading("", level=min(level, 4))
            add_runs(heading, stripped[level:].strip())
            for run in heading.runs:
                run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)
                set_font(run)
            index += 1
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}", stripped):
            add_horizontal_rule(document)
            index += 1
            continue

        # block quote
        if stripped.startswith(">"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(18)
            add_runs(paragraph, stripped.lstrip("> ").strip())
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            index += 1
            continue

        # lists
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        numbered = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if bullet and not stripped.startswith("!["):
            paragraph = document.add_paragraph(style="List Bullet")
            add_runs(paragraph, bullet.group(1))
            index += 1
            continue
        if numbered:
            paragraph = document.add_paragraph(style="List Number")
            add_runs(paragraph, numbered.group(2))
            index += 1
            continue

        if not stripped:
            index += 1
            continue

        paragraph = document.add_paragraph()
        add_runs(paragraph, stripped)
        index += 1

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(docx_path))
    stats["bytes"] = docx_path.stat().st_size
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--batch", choices=("reports",), default=None,
                        help="convert the standard report set into docs_word/")
    parser.add_argument("--outdir", default="docs_word")
    args = parser.parse_args()

    jobs: list[tuple[Path, Path]] = []
    if args.batch == "reports":
        outdir = ROOT / args.outdir
        jobs = [(ROOT / name, outdir / (Path(name).stem + ".docx")) for name in REPORT_FILES]
    elif args.src:
        source = Path(args.src)
        target = Path(args.out) if args.out else source.with_suffix(".docx")
        jobs = [(source, target)]
    else:
        parser.error("pass --src or --batch reports")

    for source, target in jobs:
        if not source.exists():
            print(f"  MISSING {source.name}")
            continue
        stats = convert(source, target)
        suffix = ""
        if stats["equations"] or stats["equation_fallbacks"]:
            suffix += f"  eq={stats['equations']}"
            if stats["equation_fallbacks"]:
                suffix += f"(fallback {stats['equation_fallbacks']})"
        if stats["images"] or stats["missing_images"]:
            suffix += f"  img={stats['images']}"
            if stats["missing_images"]:
                suffix += f"(missing {stats['missing_images']})"
        print(f"  {source.name:46s} -> {target.name:46s} "
              f"{stats['bytes']:>9,} B  tables={stats['tables']}{suffix}")


if __name__ == "__main__":
    main()
