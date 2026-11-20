"""PDF report generation: content comes only from saved analyses; output is a readable PDF."""
from __future__ import annotations

import io
import json

import pytest
from pypdf import PdfReader

from app.core.errors import InvalidRequestError, NotFoundError
from app.schemas.analysis import AnalysisRequest
from app.services.analysis_service import run_analysis
from app.services.report_service import build_report_pdf, generate_full_report, generate_report
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, ScriptedLLM


def pdf_text(data: bytes) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def smart_llm(system, prompt):
    if "summarise" in system:
        return json.dumps(
            {
                "sections": [
                    {
                        "heading": "Overview",
                        "items": [
                            {"text": "Fixed price proposal.", "kind": "stated", "source_ids": ["S1", "S2"]},
                            {"text": "Timeline looks ambitious.", "kind": "interpretation", "source_ids": []},
                        ],
                    }
                ]
            }
        )
    is_a = 'Document: "a.pdf"' in prompt
    return json.dumps(
        {
            "fields": [
                {"key": "pricing", "found": True, "value": "USD 120,000" if is_a else "USD 98,500", "source_id": "S2", "evidence": "USD 120,000" if is_a else "USD 98,500"},
                {"key": "payment_terms", "found": True, "value": "Net 30 days" if is_a else "Net 45 days", "source_id": "S2"},
                {"key": "contract_duration", "found": is_a, "value": "24 months" if is_a else None, "source_id": "S3"},
            ]
        }
    )


@pytest.fixture
def two_docs(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", VENDOR_B_PAGES)
    ctx._llm_override = ScriptedLLM(smart_llm)
    return a, b


def _run(ctx, **kw):
    return run_analysis(ctx, AnalysisRequest(**kw))


def test_report_contains_required_sections_and_evidence(ctx, two_docs):
    a, b = two_docs
    ids = [
        _run(ctx, document_ids=[a.id, b.id], operation="summarize", summary_type="executive").id,
        _run(ctx, document_ids=[a.id, b.id], operation="extract", fields=[{"label": "Pricing"}, {"label": "Payment terms"}]).id,
        _run(ctx, document_ids=[a.id, b.id], operation="compare", fields=[{"label": "Pricing"}, {"label": "Payment terms"}, {"label": "Contract duration"}]).id,
    ]
    pdf, name = generate_report(ctx, analysis_ids=ids, title="Vendor Review Q4")
    assert pdf.startswith(b"%PDF") and name == "Vendor_Review_Q4.pdf"
    text = pdf_text(pdf)
    for heading in (
        "Vendor Review Q4", "Selected documents", "Executive summary", "Key extracted information",
        "Comparison results", "Important differences and missing information", "Key findings", "Source references",
    ):
        assert heading in text, heading
    assert "a.pdf" in text and "b.pdf" in text
    assert "USD 120,000" in text and "USD 98,500" in text
    assert "Not specified" in text  # b.pdf has no contract duration
    assert "AI interpretation" in text and "Stated" in text
    assert "p. 2" in text


def test_report_never_cites_documents_or_pages_that_do_not_exist(ctx, two_docs):
    a, b = two_docs
    ids = [_run(ctx, document_ids=[a.id, b.id], operation="compare", fields=[{"label": "Pricing"}]).id]
    text = pdf_text(generate_report(ctx, analysis_ids=ids)[0])
    import re

    for page in re.findall(r"p\. (\d+)", text):
        assert 1 <= int(page) <= 3  # fixture PDFs have 3 pages
    assert "c.pdf" not in text


def test_report_handles_missing_sections_gracefully(ctx, two_docs):
    a, _ = two_docs
    ids = [_run(ctx, document_ids=[a.id], question="payment terms", operation="qa").id]
    ctx._llm_override = ScriptedLLM([])
    text = pdf_text(generate_report(ctx, analysis_ids=ids)[0])
    assert "No executive summary was included" in text
    assert "No structured extraction was included" in text
    assert "No comparison was included" in text


def test_report_with_unanswerable_qa_and_no_sources(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM([])
    rec = _run(ctx, document_ids=[a.id], question="What is the capital of France?", operation="qa")
    text = pdf_text(generate_report(ctx, analysis_ids=[rec.id])[0])
    assert "do not contain enough information" in text
    assert "No supporting source passage" in text


def test_report_version_comparison(ctx, add_doc):
    v1 = add_doc("v1.pdf", ["Payment due in 30 days.\nTerm is 12 months."])
    v2 = add_doc("v2.pdf", ["Payment due in 45 days.\nTerm is 12 months."])
    rec = _run(ctx, document_ids=[v1.id, v2.id], operation="compare", compare_mode="versions")
    text = pdf_text(generate_report(ctx, analysis_ids=[rec.id])[0])
    assert "30 days" in text and "45 days" in text and "changed" in text


def test_unicode_currency_renders_or_degrades_safely(ctx, add_doc):
    a = add_doc("rupees.pdf", ["Total price: Rs 5,00,000 only"])
    ctx._llm_override = ScriptedLLM(
        ['{"fields":[{"key":"pricing","found":true,"value":"\\u20b95,00,000 \\u2013 fixed","source_id":"S1","evidence":"5,00,000"}]}']
    )
    rec = _run(ctx, document_ids=[a.id], operation="extract", fields=[{"label": "Pricing"}])
    pdf = generate_report(ctx, analysis_ids=[rec.id])[0]
    assert pdf.startswith(b"%PDF")
    assert "5,00,000" in pdf_text(pdf)


def test_report_text_is_escaped(ctx, add_doc):
    a = add_doc("x.pdf", ["Vendor: A & B <Ltd>"])
    ctx._llm_override = ScriptedLLM(
        ['{"fields":[{"key":"vendor_name","found":true,"value":"A & B <Ltd>","source_id":"S1","evidence":"A & B <Ltd>"}]}']
    )
    rec = _run(ctx, document_ids=[a.id], operation="extract", fields=[{"label": "Vendor name"}])
    assert "A & B <Ltd>" in pdf_text(generate_report(ctx, analysis_ids=[rec.id])[0])


def test_report_input_validation(ctx):
    with pytest.raises(InvalidRequestError):
        generate_report(ctx, analysis_ids=[])
    with pytest.raises(NotFoundError):
        generate_report(ctx, analysis_ids=[404])


def test_build_report_pdf_pure_function_with_no_analyses():
    pdf = build_report_pdf([], title="Empty", document_names=[])
    assert "Empty" in pdf_text(pdf)


def test_one_click_full_report_saves_its_analyses(ctx, two_docs):
    a, b = two_docs
    pdf, name, ids = generate_full_report(ctx, document_ids=[a.id, b.id], title="Full")
    assert len(ids) == 3 and name == "Full.pdf"
    ops = [ctx.analyses.get(i)["operation"] for i in ids]
    assert ops == ["summarize", "extract", "compare"]
    text = pdf_text(pdf)
    assert "Comparison results" in text and "USD 98,500" in text


def test_one_click_report_single_document_skips_comparison(ctx, two_docs):
    a, _ = two_docs
    _, _, ids = generate_full_report(ctx, document_ids=[a.id])
    assert [ctx.analyses.get(i)["operation"] for i in ids] == ["summarize", "extract"]
