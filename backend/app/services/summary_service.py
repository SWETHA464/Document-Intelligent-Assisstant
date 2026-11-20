"""Grounded summarisation with an explicit split between stated facts and AI interpretation."""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.context import AppContext
from app.core.errors import InvalidRequestError, LLMOutputError
from app.models.domain import ChunkRecord, DocumentRecord
from app.rag.citations import build_blocks, format_context, resolve_citations
from app.rag.retriever import RetrievalSession, open_session
from app.schemas.analysis import DocumentSummary, SummaryItem, SummaryResult, SummarySection
from app.services.llm import generate_validated

logger = logging.getLogger(__name__)

MAX_ITEMS_PER_SECTION = 12
MAX_SECTIONS = 8

# summary_type -> (title, instruction, retrieval query used when the document is too long to send whole)
SUMMARY_TYPES: dict[str, tuple[str, str, str]] = {
    "general": (
        "General summary",
        "Write a general summary of the document: what it is, who it concerns, and its main content. "
        "Use 1-3 sections (for example 'Overview' and 'Main content').",
        "overview purpose scope introduction summary background main content",
    ),
    "executive": (
        "Executive summary",
        "Write a short executive summary for a busy decision-maker: purpose, headline numbers or terms, "
        "and what needs attention. Use sections 'Overview' and 'Decision-relevant points'. Keep it brief.",
        "executive summary overview total price cost timeline duration terms conclusion recommendation",
    ),
    "key_points": (
        "Key points and important findings",
        "List the key points and important findings of the document as short, separate items in a section "
        "'Key points'. Add a section 'Notable findings' only if the document contains clear findings.",
        "key points findings results conclusions highlights important",
    ),
    "risks": (
        "Risks and concerns",
        "Identify risks, concerns, penalties, liabilities, unusual or one-sided conditions, ambiguities and "
        "missing information. Use sections 'Risks stated in the document' (explicitly stated) and "
        "'Potential concerns' (your interpretation).",
        "risk liability penalty termination breach warranty limitation exclusion condition assumption dispute indemnity",
    ),
    "dates_values_obligations": (
        "Important dates, values and obligations",
        "Extract important dates and deadlines, monetary values and quantities, and obligations or commitments. "
        "Use sections 'Dates and deadlines', 'Values and amounts' and 'Obligations'.",
        "date deadline due within days weeks months price amount cost fee total payment shall must obligation responsible",
    ),
}

SYSTEM_PROMPT = """You summarise business documents using ONLY the numbered excerpts provided.
Rules:
- Every item has "kind": "stated" if the excerpts explicitly say it, or "interpretation" if it is your own
  inference, evaluation or judgement (for example that something is a risk or unusual). Never present an
  interpretation as a stated fact.
- Each item lists "source_ids": the excerpt ids that support it (for example ["S2"]). Items of kind "stated"
  MUST have at least one source id. Do not write page numbers yourself.
- Do not add facts that are not in the excerpts. Copy numbers, currencies and dates exactly.
- If the excerpts contain nothing relevant, return an empty "sections" list.
Reply with ONE JSON object only:
{"sections": [{"heading": "<text>", "items": [{"text": "<text>", "kind": "stated" or "interpretation", "source_ids": ["S1"]}]}]}"""


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    kind: Literal["stated", "interpretation"] = "interpretation"
    source_ids: list[str | int] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v):
        text = str(v or "").strip().lower()
        return "stated" if text.startswith("stat") or text in ("explicit", "fact") else "interpretation"

    @field_validator("source_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore")

    heading: str = ""
    items: list[_Item] = Field(default_factory=list)


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sections: list[_Section] = Field(default_factory=list)


def _select_chunks(
    ctx: AppContext, session: RetrievalSession, doc: DocumentRecord, query: str
) -> tuple[list[ChunkRecord], list[str], bool]:
    """Whole document if it fits the context budget, else the most relevant passages in order."""
    chunks = [c for c in session.chunks if c.document_id == doc.id]
    budget = ctx.settings.max_context_chars
    if sum(len(c.text) for c in chunks) <= budget:
        return chunks, [], False

    res = session.search(query, top_k=max(6, budget // 700), only_document_ids=[doc.id])
    scored: dict[int, tuple[float, ChunkRecord]] = {}
    if chunks:
        scored[chunks[0].id] = (10.0, chunks[0])  # the opening usually says what the document is
    for rc in res.chunks:
        scored[rc.chunk.id] = (rc.score, rc.chunk)
    chosen: list[ChunkRecord] = []
    used = 0
    for _, chunk in sorted(scored.values(), key=lambda t: -t[0]):
        if used + len(chunk.text) > budget and chosen:
            continue
        chosen.append(chunk)
        used += len(chunk.text)
    chosen.sort(key=lambda c: c.chunk_index)
    return chosen, [w for w in res.warnings if "keyword" in w.lower()], True


def summarize_document(
    ctx: AppContext, session: RetrievalSession, doc: DocumentRecord, summary_type: str
) -> DocumentSummary:
    title, instruction, query = SUMMARY_TYPES[summary_type]
    chunks, warnings, partial = _select_chunks(ctx, session, doc, query)
    blocks = build_blocks(chunks)
    if partial:
        warnings.append(
            "This document is long; the summary is based on the most relevant passages rather than every page."
        )
    prompt = (
        f'Document: "{doc.filename}"\nTask: {title}. {instruction}\n\n'
        f"Excerpts:\n{format_context(blocks)}\n\nReply with the JSON object now."
    )
    try:
        out = generate_validated(ctx.get_llm(), _Output, SYSTEM_PROMPT, prompt)
    except LLMOutputError as exc:
        return DocumentSummary(
            document_id=doc.id, document_name=doc.filename, sections=[],
            warnings=[*warnings, f"Unable to determine a summary: {exc.message}"],
        )

    sections: list[SummarySection] = []
    downgraded = 0
    for sec in out.sections[:MAX_SECTIONS]:
        items: list[SummaryItem] = []
        for raw in sec.items[:MAX_ITEMS_PER_SECTION]:
            text = raw.text.strip()
            if not text:
                continue
            sources = resolve_citations(raw.source_ids, blocks)
            kind, note = raw.kind, None
            if kind == "stated" and not sources:
                kind, note = "interpretation", "Shown as interpretation: no supporting source passage was cited."
                downgraded += 1
            items.append(SummaryItem(text=text, kind=kind, sources=sources, note=note))
        if items:
            sections.append(SummarySection(heading=sec.heading.strip() or "Summary", items=items))

    if downgraded:
        warnings.append(f"{downgraded} item(s) the model called 'stated' had no valid source and were relabelled as interpretation.")
    if not sections:
        warnings.append("No summary content could be produced from the document text.")
    return DocumentSummary(document_id=doc.id, document_name=doc.filename, sections=sections, warnings=warnings)


def summarize_documents(ctx: AppContext, documents: list[DocumentRecord], summary_type: str) -> SummaryResult:
    if summary_type not in SUMMARY_TYPES:
        raise InvalidRequestError(
            f"Unknown summary type '{summary_type}'. Available: {', '.join(SUMMARY_TYPES)}."
        )
    session = open_session(ctx, [d.id for d in documents])
    return SummaryResult(
        summary_type=summary_type,  # type: ignore[arg-type]
        documents=[summarize_document(ctx, session, d, summary_type) for d in documents],
    )
