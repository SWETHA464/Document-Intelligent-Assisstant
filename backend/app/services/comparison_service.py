"""Document comparison.

Two modes:
* "fields"   - extract the same fields from every document (evidence-grounded, see
               extraction_service) and compare the values programmatically. The verdicts and findings
               are computed by code from the extracted evidence; the LLM is not asked to "summarise".
* "versions" - deterministic text diff between two versions of a document (no LLM).
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

from app.core.context import AppContext
from app.core.errors import InvalidRequestError
from app.models.domain import DocumentRecord
from app.schemas.analysis import (
    ComparisonCell,
    ComparisonResult,
    ComparisonRow,
    DocumentRef,
    ExtractionResult,
    FieldSpec,
    Finding,
    VersionChange,
    VersionDiff,
)
from app.services.extraction_service import extract_fields
from app.utils.text import normalize_value_for_comparison, truncate

NOT_SPECIFIED = "Not specified"
UNABLE = "Unable to determine"
MAX_VERSION_CHANGES = 200


def display_value(cell: ComparisonCell) -> str:
    if cell.status == "error":
        return UNABLE
    if cell.status == "not_found" or not cell.value:
        return NOT_SPECIFIED
    return cell.value


# ------------------------------------------------------------ field mode ----
def build_comparison(extraction: ExtractionResult) -> ComparisonResult:
    """Turn per-document extraction results into a comparison table, verdicts and findings."""
    docs = [DocumentRef(id=d.document_id, name=d.document_name) for d in extraction.documents]
    by_doc = {d.document_id: {f.key: f for f in d.fields} for d in extraction.documents}

    rows: list[ComparisonRow] = []
    findings: list[Finding] = []
    for spec in extraction.fields:
        cells = []
        for d in extraction.documents:
            f = by_doc[d.document_id].get(spec.key or "")
            cells.append(
                ComparisonCell(
                    document_id=d.document_id,
                    document_name=d.document_name,
                    value=f.value if f else None,
                    status=f.status if f else "not_found",
                    source=f.source if f else None,
                )
            )
        row = _judge_row(spec, cells)
        rows.append(row)
        finding = _finding_for(row)
        if finding:
            findings.append(finding)

    verdict_counts = Counter(r.verdict for r in rows)
    counts = {
        "fields": len(rows),
        "match": verdict_counts.get("match", 0),
        "differs": verdict_counts.get("differs", 0),
        "missing_in_some": verdict_counts.get("missing_in_some", 0),
        "not_found_in_any": verdict_counts.get("not_found_in_any", 0),
        "undetermined": verdict_counts.get("undetermined", 0),
    }
    warnings = [w for d in extraction.documents for w in d.warnings]
    if any(c.status == "unverified" for r in rows for c in r.cells):
        warnings.append(
            "Some values are marked 'unverified': the model's value could not be confirmed in the cited passage. "
            "Check them against the original document."
        )
    return ComparisonResult(
        mode="fields", documents=docs, rows=rows, findings=findings, counts=counts, warnings=warnings
    )


def _judge_row(spec: FieldSpec, cells: list[ComparisonCell]) -> ComparisonRow:
    key, label = spec.key or "", spec.label
    errors = [c for c in cells if c.status == "error"]
    present = [c for c in cells if c.status in ("found", "unverified") and c.value]
    if errors and not present:
        return ComparisonRow(key=key, label=label, cells=cells, verdict="undetermined", note=UNABLE)
    if not present:
        return ComparisonRow(key=key, label=label, cells=cells, verdict="not_found_in_any")
    if len(present) < len(cells):
        missing = [c.document_name for c in cells if c not in present]
        verdict = "undetermined" if errors else "missing_in_some"
        return ComparisonRow(
            key=key, label=label, cells=cells, verdict=verdict,
            note="Not specified in: " + ", ".join(missing),
        )
    normalised = {normalize_value_for_comparison(c.value or "") for c in present}
    if len(normalised) == 1:
        return ComparisonRow(key=key, label=label, cells=cells, verdict="match")
    return ComparisonRow(
        key=key, label=label, cells=cells, verdict="differs",
        note="Values are compared as text; check whether the wording differs but the meaning is the same.",
    )


def _cell_text(cell: ComparisonCell) -> str:
    value = truncate(display_value(cell), 160)
    return f"{cell.document_name}: {value}" + (f" (p. {cell.source.page_number})" if cell.source and cell.source.page_number else "")


def _finding_for(row: ComparisonRow) -> Finding | None:
    sources = [c.source for c in row.cells if c.source]
    if row.verdict == "differs":
        return Finding(kind="difference", text=f"{row.label} differs - " + "; ".join(_cell_text(c) for c in row.cells), sources=sources)
    if row.verdict == "missing_in_some":
        have = [c.document_name for c in row.cells if c.status in ("found", "unverified") and c.value]
        lack = [c.document_name for c in row.cells if c.document_name not in have]
        return Finding(
            kind="missing",
            text=f"{row.label} is specified in {', '.join(have)} but not specified in {', '.join(lack)}.",
            sources=sources,
        )
    if row.verdict == "not_found_in_any":
        return Finding(kind="not_found", text=f"{row.label} was not found in any of the selected documents.")
    if row.verdict == "undetermined":
        return Finding(kind="warning", text=f"{row.label}: unable to determine (the model's output could not be used).")
    if row.verdict == "match":
        return Finding(kind="match", text=f"{row.label} is the same in all documents: {truncate(display_value(row.cells[0]), 160)}", sources=sources)
    return None


def compare_fields(
    ctx: AppContext, documents: list[DocumentRecord], fields: list[FieldSpec], *, preset: str | None = None
) -> ComparisonResult:
    if len(documents) < 2:
        raise InvalidRequestError("Select at least two documents to compare.")
    extraction = extract_fields(ctx, documents, fields, preset=preset)
    return build_comparison(extraction)


# ---------------------------------------------------------- version mode ----
_SENTENCE_END = re.compile(r"[.:;!?]$")


def _segments(pages: list) -> list[tuple[str, int | None]]:
    """Page text -> [(segment, page)], re-joining lines that were only wrapped by the PDF layout."""
    segments: list[tuple[str, int | None]] = []
    for page in pages:
        current = ""
        for raw in page.text.split("\n"):
            line = raw.strip()
            if not line:
                if current:
                    segments.append((current, page.page_number))
                    current = ""
                continue
            if current and not _SENTENCE_END.search(current) and line[:1].islower():
                current = f"{current} {line}"
            else:
                if current:
                    segments.append((current, page.page_number))
                current = line
        if current:
            segments.append((current, page.page_number))
    return segments


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def compare_versions(ctx: AppContext, doc_a: DocumentRecord, doc_b: DocumentRecord) -> ComparisonResult:
    seg_a = _segments(ctx.content.get_pages(doc_a.id))
    seg_b = _segments(ctx.content.get_pages(doc_b.id))
    norm_a, norm_b = [_norm(s) for s, _ in seg_a], [_norm(s) for s, _ in seg_b]
    matcher = difflib.SequenceMatcher(a=norm_a, b=norm_b, autojunk=False)

    changes: list[VersionChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                changes.append(_change("changed", seg_a[i1 + k], seg_b[j1 + k]))
            for k in range(i1 + pairs, i2):
                changes.append(_change("removed", seg_a[k], None))
            for k in range(j1 + pairs, j2):
                changes.append(_change("added", None, seg_b[k]))
        elif tag == "delete":
            changes.extend(_change("removed", seg_a[k], None) for k in range(i1, i2))
        elif tag == "insert":
            changes.extend(_change("added", None, seg_b[k]) for k in range(j1, j2))

    counts = Counter(c.kind for c in changes)
    truncated = len(changes) > MAX_VERSION_CHANGES
    diff = VersionDiff(
        similarity=round(matcher.ratio(), 3),
        changes=changes[:MAX_VERSION_CHANGES],
        counts={"changed": counts.get("changed", 0), "removed": counts.get("removed", 0), "added": counts.get("added", 0)},
        truncated=truncated,
    )
    findings = [
        Finding(
            kind="difference",
            text=(
                f"{diff.counts['changed']} passage(s) changed, {diff.counts['removed']} only in "
                f"'{doc_a.filename}', {diff.counts['added']} only in '{doc_b.filename}' "
                f"(text similarity {diff.similarity:.0%})."
            ),
        )
    ]
    warnings = ["Showing the first %d of %d differences." % (MAX_VERSION_CHANGES, len(changes))] if truncated else []
    return ComparisonResult(
        mode="versions",
        documents=[DocumentRef(id=doc_a.id, name=doc_a.filename), DocumentRef(id=doc_b.id, name=doc_b.filename)],
        findings=findings,
        counts=diff.counts,
        version_diff=diff,
        warnings=warnings,
    )


def _change(kind: str, a: tuple[str, int | None] | None, b: tuple[str, int | None] | None) -> VersionChange:
    return VersionChange(
        kind=kind,  # type: ignore[arg-type]
        a_text=truncate(a[0], 400) if a else None,
        b_text=truncate(b[0], 400) if b else None,
        a_page=a[1] if a else None,
        b_page=b[1] if b else None,
    )


def compare_documents(
    ctx: AppContext,
    documents: list[DocumentRecord],
    *,
    mode: str,
    fields: list[FieldSpec],
    preset: str | None = None,
) -> ComparisonResult:
    if len(documents) < 2:
        raise InvalidRequestError("Select at least two documents to compare.")
    if mode == "versions":
        if len(documents) != 2:
            raise InvalidRequestError("Version comparison needs exactly two documents.")
        return compare_versions(ctx, documents[0], documents[1])
    return compare_fields(ctx, documents, fields, preset=preset)
