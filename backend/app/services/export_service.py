"""CSV / JSON export of extraction and comparison results."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from app.core.errors import InvalidRequestError
from app.schemas.analysis import ComparisonResult, ExtractionResult
from app.services.comparison_service import display_value

STATUS_LABEL = {
    "found": "Found",
    "unverified": "Unverified",
    "not_found": "Not specified",
    "error": "Unable to determine",
}


def _csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection for text that starts with = + - @."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _write(rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    for row in rows:
        writer.writerow([_csv_safe(c) for c in row])
    return "﻿" + buf.getvalue()  # BOM so Excel on Windows opens UTF-8 correctly


def extraction_to_csv(result: dict) -> str:
    data = ExtractionResult.model_validate(result)
    rows: list[list[Any]] = [["Document", "Field", "Value", "Status", "Page", "Source passage"]]
    for doc in data.documents:
        for f in doc.fields:
            rows.append(
                [
                    doc.document_name,
                    f.label,
                    f.value if f.value else STATUS_LABEL.get(f.status, ""),
                    STATUS_LABEL.get(f.status, f.status),
                    f.source.page_number if f.source and f.source.page_number else "",
                    f.source.snippet if f.source else "",
                ]
            )
    return _write(rows)


def comparison_to_csv(result: dict) -> str:
    data = ComparisonResult.model_validate(result)
    if data.mode == "versions" and data.version_diff:
        rows: list[list[Any]] = [["Change", "Page (A)", "Text (A)", "Page (B)", "Text (B)"]]
        for c in data.version_diff.changes:
            rows.append([c.kind, c.a_page or "", c.a_text or "", c.b_page or "", c.b_text or ""])
        return _write(rows)
    header = ["Field", *[f"{d.name}" for d in data.documents], *[f"{d.name} (page)" for d in data.documents], "Verdict"]
    rows = [header]
    for row in data.rows:
        rows.append(
            [
                row.label,
                *[display_value(c) for c in row.cells],
                *[c.source.page_number if c.source and c.source.page_number else "" for c in row.cells],
                row.verdict.replace("_", " "),
            ]
        )
    return _write(rows)


def export_analysis(analysis: dict, fmt: str) -> tuple[str, str, str]:
    """Return (content, media_type, file_extension) for a saved analysis."""
    op = analysis["operation"]
    if fmt == "json":
        payload = {
            "id": analysis.get("id"),
            "title": analysis.get("title"),
            "operation": op,
            "created_at": analysis.get("created_at"),
            "result": analysis["result"],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False), "application/json", "json"
    if fmt == "csv":
        if op == "extract":
            return extraction_to_csv(analysis["result"]), "text/csv", "csv"
        if op == "compare":
            return comparison_to_csv(analysis["result"]), "text/csv", "csv"
        raise InvalidRequestError("CSV export is available for extraction and comparison results only.")
    raise InvalidRequestError("Export format must be 'json' or 'csv'.")
