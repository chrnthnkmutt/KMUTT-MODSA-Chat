"""Chunk processed Markdown into JSON records with metadata.

Reads:  data/processed/<category>/<name>.md   (output of data.pipeline.normalize)
Writes: chunks/<category>/<name>.json

Each JSON file follows the MOD-SA schema:

    {
      "doc_id": "...",
      "metadata": { category, source_name, language, last_updated, ... },
      "chunks": [ { chunk_id, content, page, section } ]
    }

- `page`     is pulled from the `<!-- page: N -->` markers (then stripped from text)
- `section`  is the most recent Markdown heading (# / ## / ###)
- `category` comes from the folder name

Run from the repo root:

    python -m data.pipeline.chunk
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "chunks"

CHUNK_SIZE = 1000      # target characters per chunk (matches the app default)
CHUNK_OVERLAP = 150    # characters carried over between chunks

PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")

SOURCES_FILE = ROOT / "data" / "sources.json"


def load_sources() -> dict:
    """Human-edited metadata sidecar (title/department/source_url/contact)."""
    if not SOURCES_FILE.exists():
        return {}
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8")).get("docs", {})


SOURCES = load_sources()


def detect_language(text: str) -> str:
    return "th" if any("฀" <= c <= "๿" for c in text) else "en"


def parse_lines(md: str) -> list[tuple[str, int | None, str]]:
    """Turn Markdown into (line, page, section) triples; drop page markers."""
    out: list[tuple[str, int | None, str]] = []
    page: int | None = None
    section = ""
    for line in md.split("\n"):
        marker = PAGE_RE.search(line.strip())
        if marker:
            page = int(marker.group(1))
            continue
        if line.lstrip().startswith("#"):
            section = line.lstrip("#").strip()
        out.append((line, page, section))
    return out


def pack(lines: list[tuple[str, int | None, str]]) -> list[list[tuple[str, int | None, str]]]:
    """Greedily group whole lines into chunks of ~CHUNK_SIZE with overlap."""
    groups: list[list[tuple[str, int | None, str]]] = []
    buf: list[tuple[str, int | None, str]] = []
    size = 0
    for item in lines:
        line_len = len(item[0]) + 1
        if size + line_len > CHUNK_SIZE and buf:
            groups.append(buf)
            # seed the next chunk with the tail of this one (overlap)
            seed: list[tuple[str, int | None, str]] = []
            acc = 0
            for prev in reversed(buf):
                seed.insert(0, prev)
                acc += len(prev[0]) + 1
                if acc >= CHUNK_OVERLAP:
                    break
            buf = list(seed)
            size = sum(len(i[0]) + 1 for i in buf)
        buf.append(item)
        size += line_len
    if buf:
        groups.append(buf)
    return groups


def build_document(md_path: Path) -> dict:
    rel = md_path.relative_to(PROCESSED_DIR)
    category = rel.parts[0] if len(rel.parts) > 1 else "others"
    doc_id = md_path.stem.replace(" ", "_")

    text = md_path.read_text(encoding="utf-8")
    lines = parse_lines(text)

    chunks: list[dict] = []
    for i, group in enumerate(pack(lines), start=1):
        content = "\n".join(l for l, _, _ in group).strip()
        if not content:
            continue
        page = next((p for _, p, _ in group if p is not None), None)
        section = next((s for _, _, s in group if s), "")
        item: dict = {"chunk_id": f"{doc_id}#{i:04d}", "content": content}
        if section:
            item["section"] = section
        if page is not None:
            item["page"] = page
        chunks.append(item)

    overrides = SOURCES.get(doc_id, {})
    metadata = {
        "title": overrides.get("title") or md_path.stem.replace("_", " "),
        "category": overrides.get("category") or category,
        "department": overrides.get("department", ""),
        "source_url": overrides.get("source_url", ""),
        "contact": overrides.get("contact", ""),
        "source_name": md_path.stem,
        "language": detect_language(text),
        "last_updated": date.today().isoformat(),
        "verified": overrides.get("verified", False),
    }
    return {"doc_id": doc_id, "metadata": metadata, "chunks": chunks}


def main() -> None:
    md_files = sorted(PROCESSED_DIR.rglob("*.md"))
    if not md_files:
        print(f"No Markdown files under {PROCESSED_DIR}/ — run pipeline.normalize first")
        return

    total = 0
    for md_path in md_files:
        doc = build_document(md_path)
        out_path = OUTPUT_DIR / md_path.relative_to(PROCESSED_DIR).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(doc["chunks"])
        total += n
        print(f"• {md_path}  ->  {out_path}  ({n} chunks)")

    print(f"\nDone: {len(md_files)} files, {total} chunks -> {OUTPUT_DIR}/")
    print("ต่อไป: เติม metadata มือ (title/department/source_url/contact) + ตรวจของ 🔴")


if __name__ == "__main__":
    main()
