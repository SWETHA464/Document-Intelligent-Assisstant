"""Q&A: grounding, citations, insufficient evidence, LLM failures, malformed output."""
from __future__ import annotations

import pytest

from app.core.errors import InvalidRequestError, LLMOutputError, LLMUnavailableError
from app.services import qa_service
from app.services.qa_service import NO_INFO_ANSWER, answer_question
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, DownLLM, ScriptedLLM


def _setup(ctx, add_doc):
    a = add_doc("vendor_a.pdf", VENDOR_A_PAGES)
    b = add_doc("vendor_b.pdf", VENDOR_B_PAGES)
    return a, b


def _source_id_containing(llm_call_prompt: str, needle: str) -> str:
    """Find which [S#] excerpt in the prompt contains `needle`."""
    import re

    for m in re.finditer(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\n\[S\d+\]|\n\nQuestion:)", llm_call_prompt, re.S):
        if needle in m.group(2):
            return m.group(1)
    raise AssertionError(f"{needle!r} not in any excerpt")


def test_answer_with_citations_maps_to_document_and_page(ctx, add_doc):
    a, b = _setup(ctx, add_doc)

    def reply(system, prompt):
        s_a = _source_id_containing(prompt, "Net 30")
        s_b = _source_id_containing(prompt, "Net 45")
        return (
            '{"answerable": true, "answer": "Vendor A requires payment within Net 30 days; Vendor B within Net 45 days.", '
            f'"citations": ["{s_a}", "{s_b}"]}}'
        )

    ctx._llm_override = ScriptedLLM(reply)
    res = answer_question(ctx, "Compare the payment terms", [a, b])
    assert res.answerable and res.grounded
    labels = {(s.document_name, s.page_number) for s in res.sources}
    assert labels == {("vendor_a.pdf", 2), ("vendor_b.pdf", 2)}
    assert res.retrieval_mode == "hybrid"
    prompt = ctx._llm_override.calls[0]["prompt"]
    assert "vendor_a.pdf" in prompt and "vendor_b.pdf" in prompt
    assert ctx._llm_override.calls[0]["json_mode"] is True


def test_no_evidence_means_no_llm_call(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    llm = ScriptedLLM([])  # any call would raise AssertionError
    ctx._llm_override = llm
    res = answer_question(ctx, "What is the capital of France?", [a])
    assert res.answerable is False and res.answer == NO_INFO_ANSWER and res.sources == []
    assert llm.calls == []


def test_model_can_decline_even_with_retrieved_passages(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = ScriptedLLM(['{"answerable": false, "answer": "", "citations": []}'])
    res = answer_question(ctx, "What is the payment penalty for late delivery?", [a])
    assert res.answerable is False and res.answer == NO_INFO_ANSWER and res.sources == []


def test_invented_citation_ids_are_dropped(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = ScriptedLLM(
        ['{"answerable": true, "answer": "Net 30 days.", "citations": ["S1", "S42", "page 99"]}']
    )
    res = answer_question(ctx, "payment terms", [a])
    assert all(s.source_id == "S1" for s in res.sources)
    assert any("did not match" in w for w in res.warnings)


def test_answer_without_valid_citation_is_flagged_ungrounded(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "Net 30 days.", "citations": ["S77"]}'])
    res = answer_question(ctx, "payment terms", [a])
    assert res.answerable and res.grounded is False and res.sources == []
    assert any("could not be verified" in w for w in res.warnings)


def test_malformed_output_is_retried_once_then_succeeds(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    llm = ScriptedLLM(["Sure! Here you go: not json", '{"answerable": true, "answer": "Net 30.", "citations": ["S1"]}'])
    ctx._llm_override = llm
    res = answer_question(ctx, "payment terms", [a])
    assert res.answer == "Net 30." and len(llm.calls) == 2
    assert "not valid" in llm.calls[1]["prompt"]


def test_persistently_malformed_output_raises_clear_error(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(LLMOutputError):
        answer_question(ctx, "payment terms", [a])


def test_json_in_code_fence_with_prose_is_accepted(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = ScriptedLLM(
        ['Here is the answer:\n```json\n{"answerable": true, "answer": "Net 30.", "citations": [1]}\n```']
    )
    res = answer_question(ctx, "payment terms", [a])
    assert res.grounded and res.sources[0].source_id == "S1"


def test_llm_connection_failure_propagates_as_llm_unavailable(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    ctx._llm_override = DownLLM()
    with pytest.raises(LLMUnavailableError):
        answer_question(ctx, "payment terms", [a])


def test_empty_question_rejected(ctx, add_doc):
    a, _ = _setup(ctx, add_doc)
    with pytest.raises(InvalidRequestError):
        answer_question(ctx, "   ", [a])


def test_keyword_only_mode_is_reported(ctx, add_doc):
    from app.core.errors import EmbeddingError
    from app.rag.embeddings import HashingEmbedder

    class Broken(HashingEmbedder):
        name = "broken"

        def embed_documents(self, texts):
            raise EmbeddingError("embedding server unreachable")

    a, _ = _setup(ctx, add_doc)
    ctx._embedder_override = Broken()
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "Net 30.", "citations": ["S1"]}'])
    res = answer_question(ctx, "payment terms Net 30", [a])
    assert res.retrieval_mode == "keyword_only"
    assert any("embedding server unreachable" in w for w in res.warnings)
    assert res.answerable
