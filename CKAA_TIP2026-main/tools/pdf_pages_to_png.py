"""Render selected PDF pages to PNG so the layout can be inspected visually."""

from __future__ import annotations

import argparse
import os.path as osp
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(
        ROOT / "docs_patent" / "航空燃油齿轮泵故障检测算法专利.pdf"))
    parser.add_argument("--pages", default="1,5,14,16")
    parser.add_argument("--outdir", default=str(ROOT / "docs_patent" / "preview"))
    parser.add_argument("--zoom", type=float, default=1.6)
    args = parser.parse_args()

    source = Path(args.pdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(str(source))
    print(f"pdf pages: {document.page_count}")
    for token in args.pages.split(","):
        number = int(token.strip()) - 1
        if not 0 <= number < document.page_count:
            print(f"  page {number + 1}: out of range")
            continue
        page = document.load_page(number)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(args.zoom, args.zoom))
        target = outdir / f"page{number + 1:02d}.png"
        pixmap.save(str(target))
        print(f"  page {number + 1:3d} -> {target.name}  "
              f"({pixmap.width}x{pixmap.height}, {target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
