"""Routing, the workflow runner(s), and analysis persistence."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.errors import (
    DocumentNotReadyError,
    InvalidRequestError,
    LLMOutputError,
    LLMUnavailableError,
    NotFoundError,
)
from app.schemas.analysis import AnalysisRequest
from app.services.analysis_service import (
    delete_analysis,
    get_analysis,
    list_analyses,
    run_analysis,
)
from app.workflows import build_workflow, infer_summary_type, route_operation
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, DownLLM, ScriptedLLM

try:
    import langgraph  # noqa: F401

    HAVE_LANGGRAPH = True
except ImportError:
    HAVE_LANGGRAPH = False


def req(**kw):
    kw.setdefault("document_ids", [1])
    return AnalysisRequest(**kw)


# -------------------------------------------------------------- routing ----
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(operation="qa", question="anything"), "qa"),
        (dict(operation="summarize"), "summarize"),
        (dict(question="What are the payment terms?"), "qa"),
        (dict(question="Compare the pricing", document_ids=[1, 2]), "compare"),
        (dict(question="Which vendor is cheaper?", document_ids=[1, 2]), "compare"),
        (dict(question="Compare the pricing", document_ids=[1]), "qa"),  # nothing to compare with
        (dict(question="Extract the key fields"), "extract"),
        (dict(question="Summarize this document"), "summarize"),
        (dict(question="Give me an executive overview"), "summarize"),
        (dict(), "summarize"),
        (dict(preset="vendor_proposal"), "extract"),
        (dict(fields=[{"label": "Warranty"}]), "extract"),
    ],
)
def test_route_operation(kwargs, expected):
    assert route_operation(req(**kwargs)) == expected


def test_infer_summary_type():
    assert infer_summary_type("executive summary please") == "executive"
    assert infer_summary_type("what are the risks") == "risks"
    assert infer_summary_type("list deadlines") == "dates_values_obligations"
    assert infer_summary_type("key points") == "key_points"
    assert infer_summary_type(None) == "general"


def test_unsupported_operation_rejected_by_schema():
    with pytest.raises(ValidationError):
        AnalysisRequest(operation="translate", document_ids=[1])
    with pytest.raises(ValidationError):
        AnalysisRequest(operation="qa", document_ids=[])
    with pytest.raises(ValidationError):
        AnalysisRequest(operation="summarize", document_ids=[1], summary_type="haiku")


# ------------------------------------------------------------- workflow ----
RUNNERS = [False] + ([None] if HAVE_LANGGRAPH else [])


@pytest.fixture(params=RUNNERS, ids=lambda v: "simple" if v is False else "langgraph")
def use_langgraph(request, monkeypatch):
    # run_analysis builds the workflow itself; force the requested runner
    import app.services.analysis_service as svc

    original = svc.build_workflow
    monkeypatch.setattr(svc, "build_workflow", lambda c: original(c, use_langgraph=request.param))
    return request.param


def test_qa_flow_end_to_end(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "Net 30 days.", "citations": ["S1"]}'])
    rec = run_analysis(ctx, req(document_ids=[a.id], question="What are the payment terms?"))
    assert rec.operation == "qa" and rec.id and rec.result["answerable"] is True
    assert rec.title.startswith("Q&A:") and rec.document_names == {a.id: "a.pdf"}
    assert rec.result["sources"][0]["page_number"] is not None


def test_extract_compare_and_summarize_flows(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", VENDOR_B_PAGES)

    def llm(system, prompt):
        if "summarise" in system:
            return json.dumps({"sections": [{"heading": "Overview", "items": [{"text": "A proposal.", "kind": "stated", "source_ids": ["S1"]}]}]})
        return json.dumps({"fields": [{"key": "payment_terms", "found": True, "value": "Net 30 days" if '"a.pdf"' in prompt else "Net 45 days", "source_id": "S2"}]})

    ctx._llm_override = ScriptedLLM(llm)
    ext = run_analysis(ctx, req(document_ids=[a.id, b.id], operation="extract", question="payment terms"))
    assert ext.operation == "extract" and len(ext.result["documents"]) == 2
    assert [f["key"] for f in ext.result["documents"][0]["fields"]] == ["payment_terms"]

    cmpr = run_analysis(ctx, req(document_ids=[a.id, b.id], question="Compare the payment terms"))
    assert cmpr.operation == "compare" and cmpr.result["rows"][0]["verdict"] in ("differs", "match")
    assert cmpr.title.startswith("Comparison of 2 documents")

    summ = run_analysis(ctx, req(document_ids=[a.id], operation="summarize", summary_type="executive"))
    assert summ.operation == "summarize" and summ.title.startswith("Executive summary")


def test_workflow_trace_and_error_in_state(ctx, add_doc):
    wf = build_workflow(ctx, use_langgraph=False)
    state = wf.invoke({"request": req(document_ids=[999], question="hello"), "steps": []})
    assert state["error"].code == "not_found"
    assert state["steps"] == ["route", "validate:error"]
    assert "result" not in state


def test_compare_with_one_document_is_rejected(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    with pytest.raises(InvalidRequestError):
        run_analysis(ctx, req(document_ids=[a.id], operation="compare"))


def test_missing_and_unprocessed_documents(ctx, add_doc, use_langgraph):
    from app.services.document_service import ingest_upload

    with pytest.raises(NotFoundError):
        run_analysis(ctx, req(document_ids=[12345], operation="summarize"))
    bad = ingest_upload(ctx, "broken.pdf", b"%PDF-1.4 junk").document
    with pytest.raises(DocumentNotReadyError):
        run_analysis(ctx, req(document_ids=[bad.id], operation="summarize"))


def test_llm_unavailable_propagates_with_503(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = DownLLM()
    with pytest.raises(LLMUnavailableError) as exc:
        run_analysis(ctx, req(document_ids=[a.id], question="payment terms"))
    assert exc.value.status_code == 503


def test_malformed_llm_output_propagates_as_502_for_qa(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(["x", "y"])
    with pytest.raises(LLMOutputError) as exc:
        run_analysis(ctx, req(document_ids=[a.id], question="payment terms"))
    assert exc.value.status_code == 502


def test_unexpected_exception_becomes_generic_500(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(lambda s, p: (_ for _ in ()).throw(RuntimeError("secret internals")))
    with pytest.raises(Exception) as exc:
        run_analysis(ctx, req(document_ids=[a.id], question="payment terms"))
    assert exc.value.status_code == 500 and "secret internals" not in exc.value.message


def test_no_evidence_question_still_saved_as_unanswerable(ctx, add_doc, use_langgraph):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM([])
    rec = run_analysis(ctx, req(document_ids=[a.id], question="Who won the 1998 World Cup?"))
    assert rec.result["answerable"] is False and rec.id is not None


# ---------------------------------------------------------- persistence ----
def test_history_roundtrip_and_survives_document_deletion(ctx, add_doc):
    from app.services.document_service import delete_document

    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "Net 30.", "citations": ["S1"]}'])
    rec = run_analysis(ctx, req(document_ids=[a.id], question="payment terms"))
    delete_document(ctx, a.id)

    rows = list_analyses(ctx)
    assert len(rows) == 1 and rows[0]["document_names"] == {a.id: "a.pdf"} and "result" not in rows[0]
    reopened = get_analysis(ctx, rec.id)
    assert reopened.result["answer"] == "Net 30." and reopened.params["resolved_operation"] == "qa"
    assert list_analyses(ctx, operation="compare") == []
    delete_analysis(ctx, rec.id)
    with pytest.raises(NotFoundError):
        get_analysis(ctx, rec.id)
    with pytest.raises(NotFoundError):
        delete_analysis(ctx, rec.id)


def test_save_false_does_not_persist(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "x", "citations": ["S1"]}'])
    rec = run_analysis(ctx, req(document_ids=[a.id], question="payment terms", save=False))
    assert rec.id is None and list_analyses(ctx) == []


def test_custom_title_used(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "x", "citations": ["S1"]}'])
    rec = run_analysis(ctx, req(document_ids=[a.id], question="payment terms", title="  My analysis  "))
    assert rec.title == "My analysis"
