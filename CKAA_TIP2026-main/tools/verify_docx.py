"""Verify the generated .docx files actually contain the expected content.

Checks that headings, tables and figures survived the markdown conversion, so we do
not hand over a document that silently lost its tables.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]

# document -> figures that MUST appear in the text
EXPECTED = {
    "REPORT_FOR_TEACHER.docx": ["95.99", "97.92", "2.37", "73.86", "72.82", "69.41", "46.26",
                                 "98.6", "94.0", "62.6", "79.8", "2.35%", "82.28"],
    "CKAA_TRAINABLE_PARAMS.md": ["2,148,044", "89,327,144", "2.35%", "85,208,064", "346.9"],
    "REPLY_TO_TEACHER.docx": ["95.99", "46.26", "62.6", "79.8", "2.35%"],
    "AQCL_2BIT_ANALYSIS.docx": ["46.26", "62.6", "79.8", "11.12", "3.406"],
}


def main() -> None:
    outdir = ROOT / "docs_word"
    for path in sorted(outdir.glob("*.docx")):
        document = Document(str(path))
        paragraphs = [p.text for p in document.paragraphs]
        body = "\n".join(paragraphs)
        for table in document.tables:
            for row in table.rows:
                body += "\n" + " | ".join(c.text for c in row.cells)

        headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
        print(f"{path.name}")
        print(f"    paragraphs={len(paragraphs):4d}  tables={len(document.tables):3d}  "
              f"headings={len(headings):3d}  chars={len(body):6,d}")

        key = "CKAA_TRAINABLE_PARAMS.md" if "TRAINABLE" in path.name else path.name
        wanted = EXPECTED.get(path.name) or EXPECTED.get(key, [])
        missing = [w for w in wanted if w not in body]
        if missing:
            print(f"    MISSING FIGURES: {missing}")
        elif wanted:
            print(f"    all {len(wanted)} key figures present")

    print()
    print("note: table cell text is included in the search above, so a table that was")
    print("      converted correctly will satisfy its figures.")


if __name__ == "__main__":
    main()
