"""Summaries (stated vs interpretation) and CSV/JSON export."""
from __future__ import annotations

import csv
import io
import json

import pytest

from app.core.errors import InvalidRequestError, LLMUnavailableError
from app.schemas.analysis import FieldSpec
from app.services import comparison_service as cmp
from app.services.export_service import export_analysis
from app.services.extraction_service import extract_fields
from app.services.summary_service import SUMMARY_TYPES, summarize_documents
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, DownLLM, ScriptedLLM


def test_summary_separates_stated_from_interpretation(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(
        [
            {
                "sections": [
                    {
                        "heading": "Overview",
                        "items": [
                            {"text": "Fixed fee of USD 120,000.", "kind": "stated", "source_ids": ["S1", "S2", "S3"]},
                            {"text": "The 16-week schedule may be tight.", "kind": "interpretation", "source_ids": []},
                            {"text": "Payment is Net 30.", "kind": "stated", "source_ids": ["S99"]},
                            {"text": "   ", "kind": "stated", "source_ids": ["S1"]},
                        ],
                    }
                ]
            }
        ]
    )
    res = summarize_documents(ctx, [a], "executive")
    doc = res.documents[0]
    items = doc.sections[0].items
    assert [i.kind for i in items] == ["stated", "interpretation", "interpretation"]
    assert items[0].sources and all(s.document_name == "a.pdf" for s in items[0].sources)
    assert "no supporting source" in items[2].note  # invented S99 -> downgraded
    assert any("relabelled" in w for w in doc.warnings)
    assert res.summary_type == "executive"


def test_every_summary_type_is_supported_and_unknown_rejected(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    for stype in SUMMARY_TYPES:
        ctx._llm_override = ScriptedLLM([{"sections": [{"heading": "H", "items": [{"text": "x", "kind": "stated", "source_ids": ["S1"]}]}]}])
        assert summarize_documents(ctx, [a], stype).documents[0].sections
        assert SUMMARY_TYPES[stype][0] in ctx._llm_override.calls[0]["prompt"]
    with pytest.raises(InvalidRequestError):
        summarize_documents(ctx, [a], "poem")


def test_summary_per_document_for_multiple_documents(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", VENDOR_B_PAGES)
    ctx._llm_override = ScriptedLLM(
        lambda s, p: json.dumps({"sections": [{"heading": "Overview", "items": [{"text": "ok", "kind": "stated", "source_ids": ["S1"]}]}]})
    )
    res = summarize_documents(ctx, [a, b], "general")
    assert [d.document_name for d in res.documents] == ["a.pdf", "b.pdf"]
    assert len(ctx._llm_override.calls) == 2


def test_summary_with_empty_model_output_warns(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(['{"sections": []}'])
    doc = summarize_documents(ctx, [a], "general").documents[0]
    assert doc.sections == [] and any("No summary content" in w for w in doc.warnings)


def test_summary_malformed_output_does_not_crash(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(["bad", "bad again"])
    doc = summarize_documents(ctx, [a], "general").documents[0]
    assert doc.sections == [] and any("Unable to determine" in w for w in doc.warnings)


def test_summary_llm_down_raises(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = DownLLM()
    with pytest.raises(LLMUnavailableError):
        summarize_documents(ctx, [a], "general")


def test_summary_tolerant_kind_labels(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(
        ['{"sections":[{"heading":"H","items":[{"text":"t","kind":"Stated explicitly","source_ids":"S1"},{"text":"u","kind":"AI opinion"}]}]}']
    )
    items = summarize_documents(ctx, [a], "general").documents[0].sections[0].items
    assert [i.kind for i in items] == ["stated", "interpretation"]


def test_long_document_summary_uses_subset_and_says_so(ctx, add_doc):
    from dataclasses import replace

    ctx.settings = replace(ctx.settings, max_context_chars=1500)
    pages = [f"Page {i} filler text about unrelated matters. " * 12 for i in range(12)]
    pages[6] = "Termination: either party may terminate with 60 days notice. Penalty applies for breach."
    a = add_doc("long.pdf", pages)
    ctx._llm_override = ScriptedLLM(['{"sections": []}'])
    doc = summarize_documents(ctx, [a], "risks").documents[0]
    assert any("most relevant passages" in w for w in doc.warnings)
    assert "60 days notice" in ctx._llm_override.calls[0]["prompt"]


# --------------------------------------------------------------- export ----
FIELDS = [FieldSpec(key="pricing", label="Pricing"), FieldSpec(key="payment_terms", label="Payment terms")]


def _extraction_analysis(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(
        [
            {
                "fields": [
                    {"key": "pricing", "found": True, "value": "=SUM(1,2) USD 120,000", "source_id": "S1", "evidence": "x"},
                    {"key": "payment_terms", "found": False},
                ]
            }
        ]
    )
    result = extract_fields(ctx, [a], FIELDS).model_dump()
    return {"id": 1, "title": "t", "operation": "extract", "created_at": "now", "result": result}


def test_extraction_csv_export(ctx, add_doc):
    content, media, ext = export_analysis(_extraction_analysis(ctx, add_doc), "csv")
    assert media == "text/csv" and ext == "csv" and content.startswith("﻿")
    rows = list(csv.reader(io.StringIO(content.lstrip("﻿"))))
    assert rows[0] == ["Document", "Field", "Value", "Status", "Page", "Source passage"]
    pricing, payment = rows[1], rows[2]
    assert pricing[0] == "a.pdf" and pricing[1] == "Pricing"
    assert pricing[2].startswith("'=")  # formula injection neutralised
    assert payment[2] == "Not specified" and payment[3] == "Not specified"


def test_json_export_roundtrips(ctx, add_doc):
    analysis = _extraction_analysis(ctx, add_doc)
    content, media, ext = export_analysis(analysis, "json")
    data = json.loads(content)
    assert media == "application/json" and data["operation"] == "extract"
    assert data["result"]["documents"][0]["document_name"] == "a.pdf"


def test_comparison_csv_export(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", VENDOR_B_PAGES)
    ctx._llm_override = ScriptedLLM(
        lambda s, p: json.dumps(
            {"fields": [{"key": "pricing", "found": True, "value": "USD 120,000" if '"a.pdf"' in p else "USD 98,500", "source_id": "S2"}, {"key": "payment_terms", "found": False}]}
        )
    )
    result = cmp.compare_fields(ctx, [a, b], FIELDS).model_dump()
    content, _, _ = export_analysis({"operation": "compare", "result": result}, "csv")
    rows = list(csv.reader(io.StringIO(content.lstrip("﻿"))))
    assert rows[0] == ["Field", "a.pdf", "b.pdf", "a.pdf (page)", "b.pdf (page)", "Verdict"]
    assert rows[1][0] == "Pricing" and rows[1][-1] == "differs"
    assert rows[2][1] == "Not specified"


def test_export_rejects_unsupported_requests(ctx, add_doc):
    with pytest.raises(InvalidRequestError):
        export_analysis({"operation": "qa", "result": {}}, "csv")
    with pytest.raises(InvalidRequestError):
        export_analysis({"operation": "extract", "result": {}}, "xml")
