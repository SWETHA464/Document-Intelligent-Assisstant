"""Multi-document question answering (RAG) with strict grounding."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.context import AppContext
from app.core.errors import InvalidRequestError
from app.models.domain import DocumentRecord
from app.rag.citations import build_blocks, format_context, resolve_citations
from app.rag.retriever import open_session
from app.schemas.analysis import QAResult
from app.services.llm import generate_validated

NO_INFO_ANSWER = (
    "The selected documents do not contain enough information to answer this question."
)

SYSTEM_PROMPT = """You answer questions about business documents using ONLY the numbered excerpts provided.
Rules:
- Use only facts stated in the excerpts. Never use outside knowledge and never guess.
- If the excerpts do not contain the answer, set "answerable" to false and leave "answer" empty.
- If the answer needs information from several documents, say which document each fact comes from.
- Cite the excerpts you used by id (for example "S2") in "citations". Do not write page numbers yourself.
- Be concise and precise. Copy numbers, currencies and dates exactly as written.
Reply with ONE JSON object only:
{"answerable": true or false, "answer": "<text>", "citations": ["S1", "S3"]}"""


class _QAOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answerable: bool = True
    answer: str = ""
    citations: list[str | int] = Field(default_factory=list)


def answer_question(
    ctx: AppContext, question: str, documents: list[DocumentRecord], *, top_k: int | None = None
) -> QAResult:
    question = (question or "").strip()
    if not question:
        raise InvalidRequestError("Please enter a question.")

    doc_ids = [d.id for d in documents]
    session = open_session(ctx, doc_ids)
    retrieval = session.search(
        question,
        top_k=top_k or ctx.settings.retrieval_top_k,
        per_doc_min=1 if 1 < len(doc_ids) <= 5 else 0,
    )
    warnings = list(retrieval.warnings)

    if not retrieval.chunks or not retrieval.sufficient:
        # No evidence: do not ask the model at all (it could only guess).
        return QAResult(
            question=question,
            answer=NO_INFO_ANSWER,
            answerable=False,
            sources=[],
            retrieval_mode=retrieval.mode,
            warnings=warnings,
        )

    blocks = build_blocks(retrieval.chunks)
    names = ", ".join(f'"{d.filename}"' for d in documents)
    prompt = (
        f"Documents selected: {names}\n\nExcerpts:\n{format_context(blocks)}\n\n"
        f"Question: {question}\n\nReply with the JSON object now."
    )
    out = generate_validated(ctx.get_llm(), _QAOutput, SYSTEM_PROMPT, prompt)

    answer = out.answer.strip()
    if not out.answerable or not answer:
        return QAResult(
            question=question,
            answer=NO_INFO_ANSWER,
            answerable=False,
            sources=[],
            retrieval_mode=retrieval.mode,
            warnings=warnings,
        )

    sources = resolve_citations(out.citations, blocks)
    dropped = len(out.citations) - len(sources)
    if out.citations and dropped > 0 and sources:
        warnings.append("Some citations returned by the model did not match any retrieved passage and were removed.")
    if not sources:
        warnings.append(
            "The model did not cite a valid source passage, so this answer could not be verified. "
            "Treat it with caution and check the documents."
        )
    return QAResult(
        question=question,
        answer=answer,
        answerable=True,
        grounded=bool(sources),
        sources=sources,
        retrieval_mode=retrieval.mode,
        warnings=warnings,
    )
