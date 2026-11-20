"""Structured information extraction: LLM JSON output -> Pydantic validation -> grounding check."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.context import AppContext
from app.core.errors import InvalidRequestError, LLMOutputError
from app.models.domain import ChunkRecord, DocumentRecord
from app.rag.citations import ContextBlock, block_by_id, build_blocks, format_context
from app.rag.retriever import RetrievalSession, open_session
from app.schemas.analysis import DocumentExtraction, ExtractedField, ExtractionResult, FieldSpec
from app.services.grounding import is_blank_value, is_grounded
from app.services.llm import generate_validated
from app.utils.text import truncate

logger = logging.getLogger(__name__)

MAX_VALUE_CHARS = 1000

SYSTEM_PROMPT = """You extract facts from excerpts of one business document.
Rules:
- Use ONLY the numbered excerpts. Never guess, infer or use outside knowledge.
- Return exactly one entry for every field key requested, in the same order.
- If an excerpt does not state the field, return "found": false and "value": null.
- If found, "value" is a concise string. Copy numbers, currencies, dates and names exactly as written.
- "source_id" is the id of the excerpt that states the value (for example "S2").
- "evidence" is a short verbatim quote (under 25 words) from that excerpt.
Reply with ONE JSON object only:
{"fields": [{"key": "<field key>", "found": true or false, "value": "<string or null>", "source_id": "S1 or null", "evidence": "<quote or null>"}]}"""


# ---------------------------------------------------------- LLM output ----
class _LLMField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    found: bool | None = None
    value: str | None = None
    source_id: str | None = None
    evidence: str | None = None

    @field_validator("value", "evidence", "source_id", mode="before")
    @classmethod
    def _to_text(cls, v: Any) -> Any:
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            return "; ".join(f"{k}: {x}" for k, x in v.items())
        return str(v)


class _LLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entries: list[_LLMField] = Field(default_factory=list, alias="fields")

    @model_validator(mode="before")
    @classmethod
    def _coerce_shape(cls, data: Any) -> Any:
        """Accept {"fields": [...]}, a bare list, or a {key: value-or-object} mapping."""
        if isinstance(data, list):
            return {"fields": data}
        if isinstance(data, dict):
            items = data.get("fields", data)
            if isinstance(items, dict):
                items = [
                    {"key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in items.items()
                ]
            return {"fields": items}
        return data


# ------------------------------------------------------------- context ----
def _select_chunks(
    ctx: AppContext, session: RetrievalSession, doc: DocumentRecord, fields: list[FieldSpec]
) -> tuple[list[ChunkRecord], list[str]]:
    """Choose the passages of one document to show the model.

    Short documents are shown in full; longer ones use hybrid retrieval per field (top 3 each)
    plus the first chunk (titles, dates and names usually sit at the start).
    """
    doc_chunks = [c for c in session.chunks if c.document_id == doc.id]
    budget = ctx.settings.max_context_chars
    warnings: list[str] = []
    if sum(len(c.text) for c in doc_chunks) <= budget:
        return doc_chunks, warnings

    scored: dict[int, tuple[float, ChunkRecord]] = {}
    if doc_chunks:
        scored[doc_chunks[0].id] = (10.0, doc_chunks[0])
    for spec in fields:
        res = session.search(f"{spec.label}. {spec.description}", top_k=3, only_document_ids=[doc.id])
        for w in res.warnings:
            if w not in warnings and "keyword" in w.lower():
                warnings.append(w)
        for rc in res.chunks:
            best = scored.get(rc.chunk.id, (0.0, rc.chunk))[0]
            scored[rc.chunk.id] = (max(best, rc.score), rc.chunk)

    chosen: list[ChunkRecord] = []
    used = 0
    for _, chunk in sorted(scored.values(), key=lambda t: -t[0]):
        if used + len(chunk.text) > budget and chosen:
            continue
        chosen.append(chunk)
        used += len(chunk.text)
    chosen.sort(key=lambda c: c.chunk_index)
    return chosen, warnings


# ------------------------------------------------------------ extraction ----
def _prompt(doc: DocumentRecord, fields: list[FieldSpec], blocks: list[ContextBlock]) -> str:
    field_lines = "\n".join(
        f'- key "{f.key}": {f.label}' + (f" - {f.description}" if f.description else "") for f in fields
    )
    return (
        f'Document: "{doc.filename}"\n\nFields to extract:\n{field_lines}\n\n'
        f"Excerpts:\n{format_context(blocks)}\n\nReply with the JSON object now."
    )


def _finalize(
    doc: DocumentRecord, fields: list[FieldSpec], out: _LLMOutput, blocks: list[ContextBlock]
) -> list[ExtractedField]:
    by_key = {e.key.strip().lower(): e for e in out.entries}
    by_label = {f.label.lower(): f.key for f in fields}
    for entry_key, entry in list(by_key.items()):  # tolerate the model answering with labels
        if entry_key in by_label and by_label[entry_key] not in by_key:
            by_key[by_label[entry_key]] = entry

    results: list[ExtractedField] = []
    for spec in fields:
        entry = by_key.get((spec.key or "").lower())
        if entry is None or entry.found is False or is_blank_value(entry.value):
            results.append(ExtractedField(key=spec.key or "", label=spec.label, status="not_found"))
            continue

        value = truncate(entry.value or "", MAX_VALUE_CHARS)
        block = block_by_id(blocks, entry.source_id)
        status, note = "found", None
        if block is None:
            # The model forgot or invented the source id: try to locate the value strictly.
            block = next(
                (b for b in blocks if is_grounded(value, entry.evidence, b.text, strict=True)), None
            )
            if block is None:
                status, note = "unverified", "The model did not point to a source passage that supports this value."
        if block is not None and status == "found" and not is_grounded(value, entry.evidence, block.text):
            status, note = "unverified", "This value could not be confirmed in the cited passage."
        results.append(
            ExtractedField(
                key=spec.key or "",
                label=spec.label,
                value=value,
                status=status,  # type: ignore[arg-type]
                source=block.source if block else None,
                evidence=truncate(entry.evidence, 300) if entry.evidence else None,
                note=note,
            )
        )
    return results


def extract_document(
    ctx: AppContext, session: RetrievalSession, doc: DocumentRecord, fields: list[FieldSpec], *, start_id: int = 1
) -> DocumentExtraction:
    chunks, warnings = _select_chunks(ctx, session, doc, fields)
    blocks = build_blocks(chunks, start=start_id)
    try:
        out = generate_validated(ctx.get_llm(), _LLMOutput, SYSTEM_PROMPT, _prompt(doc, fields, blocks))
    except LLMOutputError as exc:
        # One unreadable model reply should not discard the other documents' results.
        logger.warning("Extraction output invalid for %s: %s", doc.filename, exc.message)
        return DocumentExtraction(
            document_id=doc.id,
            document_name=doc.filename,
            fields=[
                ExtractedField(
                    key=f.key or "", label=f.label, status="error",
                    note="Unable to determine: the model's reply could not be understood.",
                )
                for f in fields
            ],
            warnings=[*warnings, exc.message],
        )
    return DocumentExtraction(
        document_id=doc.id,
        document_name=doc.filename,
        fields=_finalize(doc, fields, out, blocks),
        warnings=warnings,
    )


def extract_fields(
    ctx: AppContext, documents: list[DocumentRecord], fields: list[FieldSpec], *, preset: str | None = None
) -> ExtractionResult:
    if not fields:
        raise InvalidRequestError("No fields were requested for extraction.")
    session = open_session(ctx, [d.id for d in documents])
    results = [extract_document(ctx, session, doc, fields) for doc in documents]
    warnings: list[str] = []
    if len(results) > 0 and all(f.status == "error" for r in results for f in r.fields):
        warnings.append("The model returned unusable output for every document.")
    return ExtractionResult(preset=preset, fields=fields, documents=results, warnings=warnings)
