"""Shared text cleaning for the MOD-SA data-preparation pipeline."""
from __future__ import annotations

import re


def clean_thai_text(text: str) -> str:
    """Repair common Thai PDF-extraction artifacts and tidy whitespace.

    - Reassembles the SARA AM vowel (ำ) that extraction splits into either a
      decomposed ``ํ`` + ``า`` sequence or a stray space + ``า``.
    - Collapses trailing spaces and runs of blank lines.

    NOTE: this does NOT restore dropped tone marks (e.g. ``ค่าใช้จ่าย`` coming
    out as ``คาใชจาย``). Those need manual review — see pipeline/README.md.
    """
    text = text.replace("ํา", "ำ").replace(" า", "ำ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
