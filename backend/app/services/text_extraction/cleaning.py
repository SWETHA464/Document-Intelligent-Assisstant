"""Light text clean-up applied to extracted text. Keeps line breaks (tables rely on them)."""
from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()
