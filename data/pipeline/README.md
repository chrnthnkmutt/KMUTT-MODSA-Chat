# Data Preparation Pipeline (ฝั่ง DATA)

แปลงเอกสารต้นฉบับ → **chunks JSON พร้อม metadata** ส่งให้ฝั่ง RAG ผ่านโฟลเดอร์ `chunks/`

```
data/raw/  ──normalize──►  data/processed/  ──chunk──►  chunks/   ──► (ฝั่ง RAG อ่าน)
 (ต้นฉบับ)                  (markdown สะอาด)            (handoff)
```

| ชนิดไฟล์ | วิธีแปลง |
|---|---|
| PDF มี text | ดึงข้อความ (PyMuPDF) + clean |
| PDF สแกน | **Typhoon OCR** + clean |
| `.txt` / `.md` | clean |

## ติดตั้งครั้งแรก

```bash
brew install poppler                       # Typhoon OCR ต้องใช้ (macOS)
pip install -r data/requirements.txt
```

ใส่ API key ใน `.env` (ที่ root) — แทนค่า `your-typhoon-ocr-key`:

```env
TYPHOON_OCR_API_KEY="sk-..."
```

> ขอ key ได้ที่ https://opentyphoon.ai · rate limit 2 req/s, 20 req/min

## วิธีใช้ (รันจาก root ของโปรเจค)

```bash
# 1) คัดว่าไฟล์ไหนต้องทำอะไร (ไม่เรียก OCR ไม่ต้องมี key)
python -m data.pipeline.triage

# 2) ต้นฉบับ -> markdown สะอาด (เรียก Typhoon เฉพาะไฟล์สแกน)
python -m data.pipeline.normalize

# 3) markdown -> chunks JSON + metadata (อ่าน data/sources.json)
python -m data.pipeline.chunk
```

ผลลัพธ์: `data/raw/fees/x.pdf` → `data/processed/fees/x.md` → `chunks/fees/x.json`

## เติม metadata ให้ citation สวย

แก้ **`data/sources.json`** (title/department/source_url/contact ต่อไฟล์) แล้วรัน `python -m data.pipeline.chunk` ใหม่
— อย่าแก้ใน `chunks/*.json` เพราะถูก generate ทับทุกครั้ง

## หลังรันเสร็จ

1. 🔴 **ตรวจของเสี่ยง** (เงิน/วันที่/เกณฑ์ทุน) ใน `data/processed/` กับต้นฉบับ — OCR/extract อาจผิดเงียบๆ
2. ⚠️ **วรรณยุกต์ที่หาย** (เช่น `คาใชจาย` → `ค่าใช้จ่าย`) `clean` ซ่อมไม่ได้ ต้องแก้มือใน `data/processed/`
3. ฝั่ง RAG อ่าน `chunks/` อยู่แล้ว (`.env`: `RAG_SOURCE_PATHS="chunks"`) → รีสตาร์ทแอปเพื่อ re-index

## หมายเหตุ
- จุดส่งต่อระหว่าง 2 ฝั่งคือ **`chunks/` เท่านั้น** (ฝั่ง RAG ไม่ยุ่งกับ `data/`)
- เปลี่ยน path/เกณฑ์ "สแกน" ได้ที่ค่าตัวแปรบนหัวไฟล์ `normalize.py`
- ไฟล์ที่ OCR ล้มเหลว (ยังไม่ใส่ key) จะถูกข้ามและรายงานท้ายรัน ไฟล์อื่นยังแปลงต่อ
