"""File helpers."""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePath


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded file name."""
    base = PurePath(name.replace("\\", "/")).name.strip()
    base = re.sub(r"[^\w.\- ()]+", "_", base).strip(" .")
    return base[:150] or "document"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"
