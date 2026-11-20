"""Text helpers shared by retrieval, grounding checks and comparison."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have how i if in into is it its
    me my of on or our please she so than that the their them then there these they this those to us
    was we were what when where which who whom why will with would you your tell show give list find
    document documents file files""".split()
)

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")

CURRENCY_WORDS = {"$": " usd ", "€": " eur ", "£": " gbp ", "₹": " inr ", "rs.": " inr ", "rs": " inr "}


def strip_thousands_separators(text: str) -> str:
    """'1,250,000' -> '1250000' (so numbers tokenise as one token)."""
    return _THOUSANDS.sub("", text)


def stem(token: str) -> str:
    """Very light stemmer: only drops a plural 's' (payments -> payment)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    text = strip_thousands_separators(text.lower())
    tokens = [stem(t) for t in _TOKEN.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Lowercase, drop thousands separators, collapse whitespace (for substring grounding checks)."""
    return normalize_whitespace(strip_thousands_separators(text.lower()))


def extract_numbers(text: str) -> list[str]:
    """Numbers in text as canonical strings ('10,000.50' -> '10000.5', '30' -> '30')."""
    out: list[str] = []
    for m in _NUMBER.findall(strip_thousands_separators(text)):
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        out.append(m.lstrip("0") or "0")
    return out


def normalize_value_for_comparison(value: str) -> str:
    """Normalise an extracted value so trivially different spellings compare equal.

    Lowercases, unifies currency symbols, removes thousands separators and punctuation.
    Deliberately conservative: only identical normalised text counts as a match.
    """
    text = value.lower()
    for symbol, word in CURRENCY_WORDS.items():
        text = text.replace(symbol, word)
    text = strip_thousands_separators(text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return normalize_whitespace(text)


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    text = normalize_whitespace(text)
    return text if len(text) <= limit else text[: max(0, limit - len(suffix))].rstrip() + suffix


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = "field"
    if not slug[0].isalpha():
        slug = "f_" + slug
    return slug[:max_len].rstrip("_") or "field"
