"""Render the patent .docx to PDF through Word, and report layout statistics.

Uses Word's COM interface because there is no LibreOffice/pandoc on this machine.
Doing it from Python avoids the PowerShell 5.1 ANSI .ps1 encoding trap that mangled
the Chinese filename.
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WD_STAT_PAGES = 2
WD_STAT_WORDS = 0
WD_EXPORT_PDF = 17


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", default=str(
        ROOT / "docs_patent" / "航空燃油齿轮泵故障检测算法专利.docx"))
    parser.add_argument("--pdf", default="")
    args = parser.parse_args()

    source = Path(args.docx).resolve()
    if not source.exists():
        print(f"missing: {source}")
        return
    target = Path(args.pdf).resolve() if args.pdf else source.with_suffix(".pdf")

    import win32com.client  # noqa: PLC0415

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        document = word.Documents.Open(str(source), False, True)
        document.ExportAsFixedFormat(str(target), WD_EXPORT_PDF)
        print(f"pages          : {document.ComputeStatistics(WD_STAT_PAGES)}")
        print(f"words          : {document.ComputeStatistics(WD_STAT_WORDS)}")
        print(f"inline shapes  : {document.InlineShapes.Count}")
        print(f"equations      : {document.OMaths.Count}")
        print(f"tables         : {document.Tables.Count}")
        document.Close(False)
    finally:
        word.Quit()
    print(f"pdf            : {target}  ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
