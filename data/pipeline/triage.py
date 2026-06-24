"""Triage: classify each source file by how it must be converted.

Run from the repo root:

    python -m data.pipeline.triage
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "raw"
SUPPORTED = {".pdf", ".txt", ".md"}
TEXT_THRESHOLD = 80  # avg chars/page below this => treat a PDF as scanned


def classify(path: Path) -> dict[str, str]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return {"type": "text", "action": "clean -> .md"}
    if ext == ".pdf":
        doc = fitz.open(path)
        avg = sum(len(p.get_text()) for p in doc) // max(len(doc), 1)
        doc.close()
        if avg < TEXT_THRESHOLD:
            return {"type": "scanned-pdf", "action": "Typhoon OCR -> .md"}
        return {"type": "text-pdf", "action": "extract -> .md"}
    return {"type": "unsupported", "action": "skip"}


def iter_sources(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED and path.name != ".gitkeep":
            yield path


def main() -> None:
    rows = [(str(p), *classify(p).values()) for p in iter_sources(INPUT_DIR)]
    if not rows:
        print(f"No source files found under {INPUT_DIR}/")
        return

    width = max(len(r[0]) for r in rows)
    print(f"{'FILE'.ljust(width)}  {'TYPE'.ljust(12)}  ACTION")
    print("-" * (width + 32))
    for file, typ, action in rows:
        print(f"{file.ljust(width)}  {typ.ljust(12)}  {action}")

    need_ocr = sum(1 for r in rows if r[1] == "scanned-pdf")
    print(f"\nTotal: {len(rows)} files | need OCR (Typhoon): {need_ocr}")


if __name__ == "__main__":
    main()
