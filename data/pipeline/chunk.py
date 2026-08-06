"""Structure-aware chunking: processed Markdown -> JSON records with metadata.

Reads:  data/processed/<category>/<name>.md   (output of data.pipeline.normalize)
Writes: chunks/<category>/<name>.json         (schema เดิม — app ฝั่ง RAG ใช้ต่อได้เลย)

วิธีตัด (ตัดตาม "โครงสร้างเอกสาร" ไม่ตัดกลางข้อความ):

    ตาราง          -> ตัดเป็นกลุ่มแถว ~CHUNK_SIZE โดย "หัวตาราง + บริบทหน้า
                      ตาราง" (เช่น ชื่อคณะ) ติดไปซ้ำทุก chunk
    ระเบียบ        -> 1 ข้อ/มาตรา = 1 block (ไม่ตัดกลางข้อ) แล้วรวมข้อสั้นๆ
                      ในหมวดเดียวกันให้เต็ม chunk; ข้อยาวมากตัดที่ข้อย่อย (21.1, 21.2)
    คู่มือ/ร้อยแก้ว -> ตัดตามหัวข้อ (#/##) + ย่อหน้า แล้วรวมย่อหน้าใน section เดียวกัน
    block ยักษ์     -> fallback: ตัดตามบรรทัด แล้วค่อยตัดที่ช่องว่าง
                      (ช่องว่างในภาษาไทย = ขอบเขตวลี จึงไม่ตัดกลางคำ;
                       ถ้าติดตั้ง pythainlp ไว้จะใช้ตัดประโยคแทน)

ทุก chunk ถูกเติม breadcrumb `[หมวด | ชื่อเอกสาร | section]` เป็นบรรทัดแรก
(contextual retrieval แบบเบา) ช่วยคำถามกว้างๆ/ข้ามหมวด เพราะตัว chunk
บอกเองว่ามาจากเอกสารอะไร

    {
      "doc_id": "...",
      "metadata": { category, source_name, language, last_updated, ... },
      "chunks": [ { chunk_id, content, page, section } ]
    }

- `page`     is pulled from the `<!-- page: N -->` markers (then stripped from text)
- `section`  is the nearest Markdown heading / หมวดที่ N
- `category` comes from the folder name

ไฟล์ใหม่ใน data/processed/ ที่ยังไม่มี entry ใน data/sources.json จะได้ template
เปล่าๆ (department/source_url/contact ว่าง) auto-generate ใส่ให้ทุกครั้งที่รัน
— ไปเติมข้อมูลจริงที่นั่น ไม่ต้องแก้ในไฟล์นี้หรือใน chunks/*.json

Run from the repo root:

    python -m data.pipeline.chunk
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "chunks"
SOURCES_FILE = ROOT / "data" / "sources.json"

CHUNK_SIZE = 1000   # เป้าหมายต่อ chunk (ตัวอักษร ~ 350-500 token ของ bge-m3)
CHUNK_MIN = 200     # chunk เล็กกว่านี้จะถูกรวมเข้ากับ chunk ก่อนหน้า
CHUNK_MAX = 1600    # เพดาน: block เดี่ยวที่ใหญ่กว่านี้ถึงจะโดน fallback split

PAGE_RE = re.compile(r"<!--\sPage\s(\d+)")  # รวม variant (OCR FAILED)
PAGE_TAG_RE = re.compile(r"<page_number>[^<]*</page_number>")
FOOTER_RE = re.compile(r"^\s*(หน้า\s*\d+|\d+\s*/\s*\d+)\s*$")  # เลขหน้าที่ OCR ติดมา
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
CHAPTER_RE = re.compile(r"^\*{0,2}หมวด(?:ที่)?\s*\d+")
CLAUSE_RE = re.compile(r"^\*{0,2}(ข้อ|มาตรา)\s*(\d+)")
SUBITEM_RE = re.compile(r"^\*{0,2}\d+\.\d+")
TABLE_ROW_RE = re.compile(r"^\|.*\|$")
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")

# ป้ายหมวดภาษาไทย ใส่ใน breadcrumb ให้ embedding จับคำถามไทยได้ตรงขึ้น
CATEGORY_TH = {
    "fees": "ค่าเทอม/ค่าใช้จ่าย",
    "scholarship": "ทุนการศึกษา",
    "registration": "การลงทะเบียนเรียน",
    "academic_calendar": "ปฏิทินการศึกษา",
    "academic_rules": "ระเบียบ/ข้อบังคับ",
    "dormitory": "หอพัก",
    "others": "บริการนักศึกษา",
}


DEFAULT_SOURCES_README = (
    "Human-edited metadata sidecar for nicer citations. pipeline/chunk.py reads the "
    "'docs' map (keyed by doc_id = file stem) and merges these fields into each "
    "chunk's metadata. ฝ่าย data เติม title/department/source_url/contact ที่นี่ "
    "(ไม่ต้องแก้ใน chunks/*.json เพราะถูก generate ทับทุกครั้ง). "
    "ค่า department บางตัวเป็นการคาดเดา — ตรวจสอบแล้วตั้ง verified=true. "
    "แถวที่ department/source_url/contact ยังว่างคือของที่ chunk.py เพิ่ง auto-generate "
    "ให้ (ไฟล์ใหม่ที่ยังไม่มีใครกรอก) — ไปเติมแล้วรัน chunk ใหม่"
)


def load_sources_raw() -> dict:
    """Human-edited metadata sidecar (title/department/source_url/contact)."""
    if not SOURCES_FILE.exists():
        return {"_README": DEFAULT_SOURCES_README, "docs": {}}
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


SOURCES_RAW = load_sources_raw()
SOURCES: dict = SOURCES_RAW.setdefault("docs", {})


def sync_sources(md_files: list[Path]) -> None:
    """เติม template ใน data/sources.json ให้ทุกไฟล์ใหม่ใน data/processed/ ที่ยังไม่มี entry

    ไม่ทับ/ไม่ลบ entry เดิม — แค่เพิ่มของที่ขาด แล้วเตือนถ้ามี entry ที่ไฟล์หายไปแล้ว
    (เช่น ไฟล์ถูกลบ/เปลี่ยนชื่อ) ให้คนไปเช็กเอง
    """
    current_ids: set[str] = set()
    added: list[str] = []
    for md_path in md_files:
        rel = md_path.relative_to(PROCESSED_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "others"
        doc_id = md_path.stem.replace(" ", "_")
        current_ids.add(doc_id)
        if doc_id not in SOURCES:
            SOURCES[doc_id] = {
                "title": md_path.stem.replace("_", " "),
                "category": category,
                "department": "",
                "source_url": "",
                "contact": "",
                "verified": False,
            }
            added.append(doc_id)

    if added:
        SOURCES_RAW["docs"] = dict(sorted(SOURCES.items()))
        SOURCES_FILE.write_text(
            json.dumps(SOURCES_RAW, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n📝 เพิ่ม template ใน data/sources.json ให้ {len(added)} ไฟล์ใหม่ (ไปเติม department/source_url/contact):")
        for doc_id in added:
            print(f"   + {doc_id}")

    stale = sorted(set(SOURCES) - current_ids)
    if stale:
        print(f"\n⚠️  data/sources.json มี {len(stale)} entry ที่ไม่มีไฟล์ processed แล้ว (ไฟล์ถูกลบ/เปลี่ยนชื่อ?):")
        for doc_id in stale:
            print(f"   - {doc_id}")


def detect_language(text: str) -> str:
    return "th" if any("฀" <= c <= "๿" for c in text) else "en"


# ── 1) parse: Markdown -> list of structural blocks ──────────────────
@dataclass
class Block:
    kind: str                 # "table" | "clause" | "heading" | "para"
    lines: list[str] = field(default_factory=list)
    page: int | None = None
    major: str = ""           # section ระดับบน (h1/h2/หมวด) ใช้ตัดสินขอบเขต chunk
    section: str = ""         # section ที่ละเอียดสุด ใส่ใน metadata

    def text(self) -> str:
        return "\n".join(self.lines).strip()

    def size(self) -> int:
        return len(self.text())


def _shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def parse_blocks(md: str) -> list[Block]:
    blocks: list[Block] = []
    page: int | None = None
    stack: dict[int, str] = {}   # heading level -> text
    chapter = ""                 # "หมวดที่ N ..." (ระเบียบไม่มี markdown heading)
    pending_chapter_name = False
    cur: Block | None = None

    def major() -> str:
        parts = [stack[k] for k in sorted(stack) if k <= 2]
        if chapter:
            parts.append(chapter)
        return " > ".join(parts)

    def display() -> str:
        parts = [stack[k] for k in sorted(stack)]
        if chapter:
            parts.append(chapter)
        return " > ".join(_shorten(p) for p in parts[-2:])

    def flush() -> None:
        nonlocal cur
        if cur and cur.text():
            blocks.append(cur)
        cur = None

    for raw in md.split("\n"):
        line = PAGE_TAG_RE.sub("", raw).rstrip()
        stripped = line.strip()

        marker = PAGE_RE.search(stripped)
        if marker:
            # ข้อระเบียบมักคร่อมหน้า — ปล่อยให้ block เดิมกินต่อ (page = หน้าที่เริ่ม)
            if not (cur and cur.kind == "clause"):
                flush()
            page = int(marker.group(1))
            continue
        if FOOTER_RE.match(stripped):
            continue

        hm = HEADING_RE.match(stripped)
        if hm:
            flush()
            level, text = len(hm.group(1)), hm.group(2).strip()
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = text
            chapter = ""
            pending_chapter_name = False
            blocks.append(Block("heading", [stripped], page, major(), display()))
            continue

        if CHAPTER_RE.match(stripped):
            flush()
            chapter = stripped.strip("*").strip()
            pending_chapter_name = True
            blocks.append(Block("heading", [stripped], page, major(), display()))
            continue

        # ชื่อหมวดมักอยู่บรรทัดถัดไป เช่น "หมวดที่ 1\nบททั่วไป"
        if (
            pending_chapter_name
            and stripped
            and len(stripped) <= 60
            and not (CLAUSE_RE.match(stripped) or TABLE_ROW_RE.match(stripped))
        ):
            chapter += " " + stripped.strip("*").strip()
            blocks[-1].lines.append(stripped)
            blocks[-1].major, blocks[-1].section = major(), display()
            pending_chapter_name = False
            continue
        pending_chapter_name = False

        if CLAUSE_RE.match(stripped):
            flush()
            cur = Block("clause", [stripped], page, major(), display())
            continue

        if TABLE_ROW_RE.match(stripped):
            if not (cur and cur.kind == "table"):
                flush()
                cur = Block("table", [], page, major(), display())
            cur.lines.append(stripped)
            continue

        if not stripped:
            # ทุกอย่างระหว่าง "ข้อ N" ถึง "ข้อ N+1" เป็นเนื้อหาของข้อนั้น
            if cur and cur.kind == "clause":
                cur.lines.append("")
            else:
                flush()
            continue

        if cur and cur.kind == "table":  # บรรทัดปกติ = จบตาราง
            flush()
        if cur is None:
            cur = Block("para", [], page, major(), display())
        cur.lines.append(stripped)

    flush()
    return blocks


# ── 2) fallback: ตัด block ยักษ์แบบไม่ตัดกลางคำไทย ────────────────────
def _split_long_line(line: str) -> list[str]:
    """ตัดบรรทัดยาวเกิน CHUNK_MAX: pythainlp (ถ้ามี) > ช่องว่าง > ตัดดิบ"""
    try:  # optional dependency
        from pythainlp.tokenize import sent_tokenize

        pieces = [s for s in sent_tokenize(line) if s.strip()]
        if len(pieces) > 1:
            return pieces
    except Exception:
        pass
    pieces = line.split(" ")
    if len(pieces) == 1:  # ไม่มีช่องว่างเลย — ตัดดิบเป็นท่อนๆ
        return [line[i : i + CHUNK_SIZE] for i in range(0, len(line), CHUNK_SIZE)]
    return pieces


def split_oversize(text: str) -> list[str]:
    """แตกข้อความใหญ่เป็นท่อน ≤ CHUNK_SIZE โดยเลือกจุดตัดที่ปลอดภัยกับภาษาไทย"""
    units: list[str] = []
    for line in text.split("\n"):
        if len(line) > CHUNK_MAX:
            units.extend(_split_long_line(line))
        else:
            units.append(line)

    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for unit in units:
        if buf and size + len(unit) + 1 > CHUNK_SIZE:
            parts.append("\n".join(buf).strip())
            buf, size = [], 0
        buf.append(unit)
        size += len(unit) + 1
    if buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


# ── 3) ตัดตาราง: หัวตาราง + บริบท ติดไปทุก chunk ──────────────────────
def table_chunks(block: Block, context: str) -> list[dict]:
    lines = block.lines
    header: list[str] = []
    if len(lines) >= 2 and TABLE_SEP_RE.match(lines[1]):
        header, lines = lines[:2], lines[2:]

    prefix = (context.strip() + "\n\n" if context.strip() else "") + "\n".join(header)
    out: list[dict] = []
    rows: list[str] = []
    size = len(prefix)
    for row in lines:
        if rows and size + len(row) + 1 > CHUNK_SIZE:
            out.append({"content": f"{prefix}\n" + "\n".join(rows), "block": block})
            rows, size = [], len(prefix)
        rows.append(row)
        size += len(row) + 1
    if rows:
        out.append({"content": f"{prefix}\n" + "\n".join(rows), "block": block})
    return out


# ── 4) ตัดข้อระเบียบยาวเกินเพดาน: แบ่งที่ข้อย่อย (21.1, 21.2, …) ────────
def clause_chunks(block: Block) -> list[dict]:
    m = CLAUSE_RE.match(block.lines[0].strip())
    label = f"{m.group(1)} {m.group(2)}" if m else "ข้อ (ต่อ)"

    segments: list[list[str]] = [[]]
    for line in block.lines:
        if SUBITEM_RE.match(line.strip()) and segments[-1]:
            segments.append([])
        segments[-1].append(line)

    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for seg in segments:
        seg_text = "\n".join(seg).strip()
        if buf and size + len(seg_text) > CHUNK_SIZE:
            pieces.append("\n".join(buf).strip())
            buf, size = [], 0
        buf.append(seg_text)
        size += len(seg_text) + 1
    if buf:
        pieces.append("\n".join(buf).strip())

    if len(pieces) == 1 and len(pieces[0]) > CHUNK_MAX:  # ไม่มีข้อย่อยให้ตัด
        pieces = split_oversize(pieces[0])
    return [
        {"content": p if i == 0 else f"{label} (ต่อ)\n{p}", "block": block}
        for i, p in enumerate(pieces)
    ]


# ── 5) pack: รวม block ให้เต็ม chunk โดยไม่ข้าม section ใหญ่ ────────────
def pack_blocks(blocks: list[Block]) -> list[dict]:
    chunks: list[dict] = []
    buf: list[Block] = []
    buf_size = 0

    def emit() -> None:
        nonlocal buf, buf_size
        if buf:
            chunks.append(
                {"content": "\n\n".join(b.text() for b in buf), "block": buf[0]}
            )
            buf, buf_size = [], 0

    for block in blocks:
        if block.kind == "table":
            # buffer สั้นๆ หน้าตาราง (หัวข้อ/ชื่อคณะ/วันที่) = บริบทของตาราง
            context = ""
            if buf and buf_size <= CHUNK_MIN * 2:
                context = "\n".join(b.text() for b in buf)
                buf, buf_size = [], 0
            emit()
            chunks.extend(table_chunks(block, context or block.section))
            continue

        if block.size() > CHUNK_MAX:
            emit()
            if block.kind == "clause":
                chunks.extend(clause_chunks(block))
            else:
                chunks.extend(
                    {"content": p, "block": block} for p in split_oversize(block.text())
                )
            continue

        # หัวข้อลอยๆ ต้องเกาะเนื้อหา "ข้างหน้า" — ห้าม emit buffer ที่มีแต่ heading
        if (
            buf
            and not all(b.kind == "heading" for b in buf)
            and (block.major != buf[0].major or buf_size + block.size() > CHUNK_SIZE)
        ):
            emit()
        buf.append(block)
        buf_size += block.size() + 2
    emit()

    # chunk จิ๋ว (หัวข้อลอย/ท้าย section) รวมเข้ากับ chunk ก่อนหน้า
    merged: list[dict] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk["content"]) < CHUNK_MIN
            and merged[-1]["block"].kind != "table"
            and len(merged[-1]["content"]) + len(chunk["content"]) <= CHUNK_MAX
        ):
            merged[-1]["content"] += "\n\n" + chunk["content"]
            continue
        merged.append(chunk)
    return merged


# ── 6) ประกอบเป็นเอกสาร JSON ตาม schema เดิม ─────────────────────────
def build_document(md_path: Path) -> dict:
    rel = md_path.relative_to(PROCESSED_DIR)
    category = rel.parts[0] if len(rel.parts) > 1 else "others"
    doc_id = md_path.stem.replace(" ", "_")

    text = md_path.read_text(encoding="utf-8")
    overrides = SOURCES.get(doc_id, {})
    title = overrides.get("title") or md_path.stem.replace("_", " ")
    category = overrides.get("category") or category
    cat_th = CATEGORY_TH.get(category, category)

    chunks: list[dict] = []
    for i, piece in enumerate(pack_blocks(parse_blocks(text)), start=1):
        block: Block = piece["block"]
        crumb_parts = [cat_th, _shorten(title, 80)]
        if block.section and block.section not in title:
            crumb_parts.append(block.section)
        content = f"[{' | '.join(crumb_parts)}]\n{piece['content']}".strip()

        item: dict = {"chunk_id": f"{doc_id}#{i:04d}", "content": content}
        if block.section:
            item["section"] = block.section
        if block.page is not None:
            item["page"] = block.page
        chunks.append(item)

    metadata = {
        "title": title,
        "category": category,
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

    sync_sources(md_files)

    total = 0
    for md_path in md_files:
        doc = build_document(md_path)
        out_path = OUTPUT_DIR / md_path.relative_to(PROCESSED_DIR).with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(doc["chunks"])
        sizes = [len(c["content"]) for c in doc["chunks"]] or [0]
        total += n
        print(
            f"• {md_path.relative_to(ROOT)}  ->  {n} chunks "
            f"(avg {sum(sizes)//max(len(sizes),1)} chars, max {max(sizes)})"
        )

    print(f"\nDone: {len(md_files)} files, {total} chunks -> {OUTPUT_DIR}/")
    print("ต่อไป: python -m data.pipeline.chunk เสร็จแล้ว rebuild index (rm -rf chroma_db)")


if __name__ == "__main__":
    main()
