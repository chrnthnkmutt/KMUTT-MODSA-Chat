"""Normalize every source file into clean Markdown under processed/.

    text PDF    -> extract text (PyMuPDF) + clean
    scanned PDF -> Typhoon OCR (needs TYPHOON_OCR_API_KEY in .env) + clean
    .txt / .md  -> clean

Output mirrors the input folders: data/raw/fees/x.pdf -> data/processed/fees/x.md

Run from the repo root:

    python -m data.pipeline.normalize
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
import pymupdf as fitz

from data.pipeline.clean import clean_thai_text

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

INPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
SUPPORTED = {".pdf", ".txt", ".md"}
TEXT_THRESHOLD = 80
OCR_DELAY_SEC = 3.5  # stay under Typhoon's 20 req/min limit


def is_scanned_pdf(path: Path) -> bool:
    doc = fitz.open(path)
    avg = sum(len(p.get_text()) for p in doc) // max(len(doc), 1)
    doc.close()
    return avg < TEXT_THRESHOLD


def extract_text_pdf(path: Path) -> str:
    doc = fitz.open(path)
    parts = [f"<!-- page: {i} -->\n{page.get_text()}" for i, page in enumerate(doc, start=1)]
    doc.close()
    return clean_thai_text("\n\n".join(parts))


def ocr_pdf_typhoon(path: Path) -> str:
    if not os.getenv("TYPHOON_OCR_API_KEY"):
        raise RuntimeError("TYPHOON_OCR_API_KEY is not set in .env — required for OCR")
    from typhoon_ocr import ocr_document  # lazy import so non-OCR runs need no install

    doc = fitz.open(path)
    pages = len(doc)
    doc.close()

    chunks: list[str] = []
    for page_num in range(1, pages + 1):
        print(f"    OCR page {page_num}/{pages} ...")
        page_md = ocr_document(pdf_or_image_path=str(path), page_num=page_num)
        chunks.append(f"<!-- page: {page_num} -->\n{page_md}")
        if page_num < pages:
            time.sleep(OCR_DELAY_SEC)
    return clean_thai_text("\n\n".join(chunks))


def convert(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return clean_thai_text(path.read_text(encoding="utf-8", errors="replace"))
    if ext == ".pdf":
        return ocr_pdf_typhoon(path) if is_scanned_pdf(path) else extract_text_pdf(path)
    raise ValueError(f"unsupported file type: {path}")


def main() -> None:
    sources = [
        p
        for p in sorted(INPUT_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED and p.name != ".gitkeep"
    ]
    if not sources:
        print(f"No source files found under {INPUT_DIR}/")
        return

    ok = 0
    for path in sources:
        out_path = OUTPUT_DIR / path.relative_to(INPUT_DIR).with_suffix(".md")
        print(f"• {path}  ->  {out_path}")
        try:
            md = convert(path)
        except Exception as exc:  # keep going; report at the end
            print(f"    skip: {exc}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        ok += 1

    print(f"\nDone: {ok}/{len(sources)} files -> {OUTPUT_DIR}/")
    print("เตือน: ตรวจของเสี่ยง 🔴 (เงิน/วันที่/ทุน) กับต้นฉบับก่อนนำไปใช้")


if __name__ == "__main__":
    main()
