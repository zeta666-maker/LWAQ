"""Extract the full text and structure of a .docx into a readable UTF-8 file.

Used to read the patent template (PGMTKD(1).docx) so the new patent draft can follow
its exact section structure, heading style and claim layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: extract_docx.py <input.docx> [output.txt]")
        return
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix(".extracted.txt")

    document = Document(str(source))
    lines: list[str] = []

    for block in document.element.body.iter():
        tag = block.tag.split("}")[-1]
        if tag == "p":
            text = "".join(node.text or "" for node in block.iter() if node.tag.endswith("}t"))
            style = ""
            for node in block.iter():
                if node.tag.endswith("}pStyle"):
                    style = node.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val") or ""
            if text.strip():
                lines.append(f"[{style}] {text}" if style else text)
        elif tag == "tbl":
            rows = []
            for tr in block.iter():
                if tr.tag.endswith("}tr"):
                    cells = []
                    for tc in tr.iter():
                        if tc.tag.endswith("}tc"):
                            cell_text = "".join(
                                n.text or "" for n in tc.iter() if n.tag.endswith("}t")
                            )
                            cells.append(cell_text.strip())
                    if cells:
                        rows.append(" | ".join(cells))
            if rows:
                lines.append("[TABLE]")
                lines.extend(f"  {r}" for r in rows)
                lines.append("[/TABLE]")

    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target}  ({len(lines)} blocks, {target.stat().st_size:,} bytes)")
    print()
    print("=== first 120 lines ===")
    for line in lines[:120]:
        print(line)


if __name__ == "__main__":
    main()
