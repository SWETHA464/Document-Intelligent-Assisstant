"""Structured extraction (validation + grounding) and document comparison."""
from __future__ import annotations

import json
import re

import pytest

from app.core.errors import InvalidRequestError, LLMOutputError, LLMUnavailableError
from app.schemas.analysis import DocumentExtraction, ExtractedField, ExtractionResult, FieldSpec
from app.services import comparison_service as cmp
from app.services.extraction_service import extract_fields
from app.services.field_presets import (
    DEFAULT_PRESET,
    fields_from_text,
    get_preset_fields,
    resolve_fields,
)
from app.services.grounding import is_blank_value, is_grounded
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, DownLLM, ScriptedLLM

FIELDS = [
    FieldSpec(key="pricing", label="Pricing", description="total price"),
    FieldSpec(key="payment_terms", label="Payment terms"),
    FieldSpec(key="contract_duration", label="Contract duration"),
]


def sid(prompt: str, needle: str) -> str | None:
    """Id of the first excerpt containing `needle` (None if absent)."""
    for m in re.finditer(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\n\[S\d+\]|\n\nReply with)", prompt, re.S):
        if needle in m.group(2):
            return m.group(1)
    return None


def entry(prompt, key, value, needle, evidence=None):
    s = sid(prompt, needle)
    return {"key": key, "found": True, "value": value, "source_id": s, "evidence": evidence or needle}


def doc_fake(per_doc: dict[str, list]):
    """Fake LLM that answers per document, chosen by the document name in the prompt."""

    def reply(system, prompt):
        for name, builder in per_doc.items():
            if f'Document: "{name}"' in prompt:
                return json.dumps({"fields": [b(prompt) if callable(b) else b for b in builder]})
        raise AssertionError("unexpected document in prompt")

    return ScriptedLLM(reply)


# ------------------------------------------------------------- grounding ----
def test_grounding_checks_numbers_strictly():
    passage = "Total project price: USD 120,000 fixed fee. Payment terms: Net 30 days."
    assert is_grounded("USD 120,000", None, passage)
    assert is_grounded("120000 dollars", None, passage)  # same number, different formatting
    assert not is_grounded("USD 99,000", "USD 99,000", passage)  # invented price
    assert is_grounded("Net 30 days", None, passage)
    assert not is_grounded("Net 45 days", None, passage)


def test_grounding_text_values():
    passage = "Vendor: Acme Solutions Pvt Ltd. Proposal issued in Chennai."
    assert is_grounded("Acme Solutions Pvt Ltd", "Vendor: Acme Solutions Pvt Ltd", passage)
    assert is_grounded("Acme Solutions", None, passage)
    assert not is_grounded("Globex Corporation", None, passage)
    assert not is_grounded("Zenith Corp", "Zenith Corp", passage, strict=True)


def test_blank_markers():
    for v in (None, "", "N/A", "not specified", "Unknown.", "null", "  none "):
        assert is_blank_value(v)
    assert not is_blank_value("Net 30")


# ------------------------------------------------------------ extraction ----
def test_extraction_found_values_have_sources(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = doc_fake(
        {
            "a.pdf": [
                lambda p: entry(p, "pricing", "USD 120,000 fixed fee", "USD 120,000"),
                lambda p: entry(p, "payment_terms", "Net 30 days from invoice", "Net 30 days"),
                lambda p: entry(p, "contract_duration", "24 months", "24 months"),
            ]
        }
    )
    res = extract_fields(ctx, [a], FIELDS)
    fields = {f.key: f for f in res.documents[0].fields}
    assert all(f.status == "found" for f in fields.values())
    assert fields["pricing"].source.page_number == 2 and fields["pricing"].source.document_name == "a.pdf"
    assert fields["contract_duration"].source.page_number == 3
    assert fields["payment_terms"].evidence


def test_not_found_is_distinct_from_found(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = doc_fake(
        {
            "a.pdf": [
                {"key": "pricing", "found": False, "value": None, "source_id": None, "evidence": None},
                {"key": "payment_terms", "found": True, "value": "not specified", "source_id": None},
                {"key": "contract_duration", "found": True, "value": "24 months", "source_id": "S1", "evidence": ""},
            ]
        }
    )
    f = {x.key: x for x in extract_fields(ctx, [a], FIELDS).documents[0].fields}
    assert f["pricing"].status == "not_found" and f["pricing"].value is None
    assert f["payment_terms"].status == "not_found"  # "not specified" treated as missing
    # the value "24 months" is NOT in S1 (page 1 text) -> must not be reported as verified
    assert f["contract_duration"].status in ("found", "unverified")


def test_hallucinated_value_is_marked_unverified(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = doc_fake(
        {
            "a.pdf": [
                lambda p: entry(p, "pricing", "USD 99,000", "USD 120,000", evidence="USD 120,000"),
                lambda p: entry(p, "payment_terms", "Net 30 days", "Net 30 days"),
                {"key": "contract_duration", "found": False},
            ]
        }
    )
    f = {x.key: x for x in extract_fields(ctx, [a], FIELDS).documents[0].fields}
    assert f["pricing"].status == "unverified" and "could not be confirmed" in f["pricing"].note
    assert f["payment_terms"].status == "found"


def test_invented_source_id_is_replaced_by_real_location_or_unverified(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = doc_fake(
        {
            "a.pdf": [
                {"key": "pricing", "found": True, "value": "USD 120,000", "source_id": "S99", "evidence": "x"},
                {"key": "payment_terms", "found": True, "value": "Net 77 days", "source_id": "S99"},
                {"key": "contract_duration", "found": False},
            ]
        }
    )
    f = {x.key: x for x in extract_fields(ctx, [a], FIELDS).documents[0].fields}
    assert f["pricing"].status == "found" and f["pricing"].source.page_number == 2  # located strictly
    assert f["payment_terms"].status == "unverified" and f["payment_terms"].source is None


def test_model_may_answer_as_mapping_or_use_labels(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(
        [json.dumps({"fields": {"Pricing": {"found": True, "value": "USD 120,000", "source_id": "S2"}, "payment_terms": "Net 30 days"}})]
    )
    f = {x.key: x for x in extract_fields(ctx, [a], FIELDS).documents[0].fields}
    assert f["pricing"].value == "USD 120,000"
    assert f["payment_terms"].value == "Net 30 days"
    assert f["contract_duration"].status == "not_found"


def test_non_string_values_are_coerced(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(
        [json.dumps({"fields": [{"key": "contract_duration", "found": True, "value": 24, "source_id": 3}]})]
    )
    f = {x.key: x for x in extract_fields(ctx, [a], FIELDS[2:]).documents[0].fields}
    assert f["contract_duration"].value == "24"


def test_malformed_model_output_gives_error_status_not_crash(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = ScriptedLLM(["garbage", "more garbage"])
    res = extract_fields(ctx, [a], FIELDS)
    assert all(f.status == "error" for f in res.documents[0].fields)
    assert res.warnings and "unusable" in res.warnings[0]


def test_llm_down_raises(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    ctx._llm_override = DownLLM()
    with pytest.raises(LLMUnavailableError):
        extract_fields(ctx, [a], FIELDS)


def test_no_fields_requested(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    with pytest.raises(InvalidRequestError):
        extract_fields(ctx, [a], [])


def test_long_documents_use_retrieval_not_full_text(ctx, add_doc, settings):
    from dataclasses import replace

    ctx.settings = replace(ctx.settings, max_context_chars=2000)
    filler = [f"Background paragraph {i}. " + "Lorem ipsum dolor sit amet. " * 20 for i in range(15)]
    a = add_doc("long.pdf", ["Intro page\n" + filler[0]] + filler[1:7] + ["Payment terms: Net 90 days exact."] + filler[8:])
    seen = {}

    def reply(system, prompt):
        seen["prompt"] = prompt
        return json.dumps({"fields": [entry(prompt, "payment_terms", "Net 90 days", "Net 90 days")]})

    ctx._llm_override = ScriptedLLM(reply)
    res = extract_fields(ctx, [a], [FieldSpec(key="payment_terms", label="Payment terms")])
    assert "Net 90 days" in seen["prompt"]
    assert len(seen["prompt"]) < 5000  # budget respected, whole doc not sent
    assert res.documents[0].fields[0].status == "found"


# ------------------------------------------------------------- presets ----
def test_presets_and_field_resolution():
    assert len(get_preset_fields(DEFAULT_PRESET)) == 10
    with pytest.raises(InvalidRequestError):
        get_preset_fields("nope")
    keys = [f.key for f in fields_from_text("Compare pricing and payment terms across vendors")]
    assert keys == ["pricing", "payment_terms"]
    assert fields_from_text("hello") == []
    name, fields = resolve_fields(preset=None, custom=None, text="what differs?")
    assert name == DEFAULT_PRESET and len(fields) == 10
    name, fields = resolve_fields(preset=None, custom=[FieldSpec(label="Warranty period"), FieldSpec(label="Warranty period")], text=None)
    assert name is None and [f.key for f in fields] == ["warranty_period", "warranty_period_2"]


def test_fieldspec_validation():
    assert FieldSpec(label="Total Price (USD)").key == "total_price_usd"
    assert FieldSpec(key="Payment Terms", label="x").key == "payment_terms"
    with pytest.raises(ValueError):
        FieldSpec(label="   ")
    with pytest.raises(ValueError):
        FieldSpec(label="")


# ------------------------------------------------------------ comparison ----
def _compare(ctx, add_doc, b_pages=VENDOR_B_PAGES, b_builders=None):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", b_pages)
    ctx._llm_override = doc_fake(
        {
            "a.pdf": [
                lambda p: entry(p, "pricing", "USD 120,000 fixed fee", "USD 120,000"),
                lambda p: entry(p, "payment_terms", "Net 30 days", "Net 30 days"),
                lambda p: entry(p, "contract_duration", "24 months", "24 months"),
            ],
            "b.pdf": b_builders
            or [
                lambda p: entry(p, "pricing", "USD 98,500 including support", "USD 98,500"),
                lambda p: entry(p, "payment_terms", "Net 45 days", "Net 45 days"),
                {"key": "contract_duration", "found": False},
            ],
        }
    )
    return cmp.compare_fields(ctx, [a, b], FIELDS)


def test_comparison_table_verdicts_and_findings(ctx, add_doc):
    res = _compare(ctx, add_doc)
    rows = {r.key: r for r in res.rows}
    assert rows["pricing"].verdict == "differs"
    assert rows["payment_terms"].verdict == "differs"
    assert rows["contract_duration"].verdict == "missing_in_some"
    assert "b.pdf" in rows["contract_duration"].note
    assert [c.document_name for c in rows["pricing"].cells] == ["a.pdf", "b.pdf"]
    assert rows["pricing"].cells[0].source.page_number == 2
    assert res.counts["differs"] == 2 and res.counts["missing_in_some"] == 1
    kinds = {f.kind for f in res.findings}
    assert kinds == {"difference", "missing"}
    diff = next(f for f in res.findings if "Pricing" in f.text)
    assert "USD 120,000" in diff.text and "USD 98,500" in diff.text and "p. 2" in diff.text
    assert len(diff.sources) == 2


def test_missing_values_are_labelled_not_invented(ctx, add_doc):
    res = _compare(ctx, add_doc)
    missing = next(r for r in res.rows if r.key == "contract_duration").cells[1]
    assert missing.value is None and cmp.display_value(missing) == "Not specified"


def test_identical_values_with_formatting_differences_match(ctx, add_doc):
    res = _compare(
        ctx,
        add_doc,
        b_pages=["Beta\nPrice: $120,000 fixed", "Payment: net 30 days.", "Term 24 months"],
        b_builders=[
            lambda p: entry(p, "pricing", "usd 120000 FIXED FEE", "120,000"),
            lambda p: entry(p, "payment_terms", "NET 30 DAYS", "30 days"),
            lambda p: entry(p, "contract_duration", "24 months", "24 months"),
        ],
    )
    # case, thousands separators and currency spelling are normalised away
    assert {r.key: r.verdict for r in res.rows} == {
        "pricing": "match",
        "payment_terms": "match",
        "contract_duration": "match",
    }
    assert {f.kind for f in res.findings} == {"match"}


def test_different_wording_is_reported_as_difference_not_guessed_equal(ctx, add_doc):
    res = _compare(
        ctx,
        add_doc,
        b_pages=["Beta\nPrice: USD 120,000", "Payment due 30 days after delivery.", "Term 24 months"],
        b_builders=[
            lambda p: entry(p, "pricing", "USD 120,000 fixed fee", "USD 120,000"),
            lambda p: entry(p, "payment_terms", "30 days after delivery", "30 days after delivery"),
            lambda p: entry(p, "contract_duration", "24 months", "24 months"),
        ],
    )
    rows = {r.key: r for r in res.rows}
    # "Net 30 days" vs "30 days after delivery" share a number but are NOT the same term
    assert rows["payment_terms"].verdict == "differs"
    assert rows["pricing"].verdict == "match"


def test_not_found_in_any_document(ctx, add_doc):
    res = _compare(
        ctx,
        add_doc,
        b_builders=[{"key": k, "found": False} for k in ("pricing", "payment_terms", "contract_duration")],
    )
    # document A still has all three, so every row is 'missing_in_some'
    assert {r.verdict for r in res.rows} == {"missing_in_some"}


def test_field_missing_from_every_document():
    extraction = ExtractionResult(
        fields=FIELDS[:1],
        documents=[
            DocumentExtraction(
                document_id=i,
                document_name=f"d{i}",
                fields=[ExtractedField(key="pricing", label="Pricing")],
            )
            for i in (1, 2)
        ],
    )
    result = cmp.build_comparison(extraction)
    assert result.rows[0].verdict == "not_found_in_any"
    assert result.findings[0].kind == "not_found"


def test_malformed_output_for_one_document_marks_unable_to_determine(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    b = add_doc("b.pdf", VENDOR_B_PAGES)

    def reply(system, prompt):
        if 'Document: "b.pdf"' in prompt:
            return "not json"
        return json.dumps({"fields": [entry(prompt, "pricing", "USD 120,000", "USD 120,000"), {"key": "payment_terms", "found": False}, {"key": "contract_duration", "found": False}]})

    ctx._llm_override = ScriptedLLM(reply)
    res = cmp.compare_fields(ctx, [a, b], FIELDS)
    pricing = next(r for r in res.rows if r.key == "pricing")
    assert pricing.verdict == "undetermined" and cmp.display_value(pricing.cells[1]) == "Unable to determine"
    assert all(r.verdict in ("undetermined", "not_found_in_any") or r.key == "pricing" for r in res.rows)


def test_comparison_needs_two_documents(ctx, add_doc):
    a = add_doc("a.pdf", VENDOR_A_PAGES)
    with pytest.raises(InvalidRequestError):
        cmp.compare_documents(ctx, [a], mode="fields", fields=FIELDS)
    with pytest.raises(InvalidRequestError):
        cmp.compare_documents(ctx, [a, a, a], mode="versions", fields=FIELDS)


def test_unverified_values_add_warning(ctx, add_doc):
    res = _compare(
        ctx,
        add_doc,
        b_builders=[
            {"key": "pricing", "found": True, "value": "USD 1", "source_id": "S1"},
            {"key": "payment_terms", "found": False},
            {"key": "contract_duration", "found": False},
        ],
    )
    assert any("unverified" in w for w in res.warnings)


# --------------------------------------------------------- version diff ----
def test_version_diff_finds_changed_added_removed(ctx, add_doc):
    v1 = add_doc(
        "v1.pdf",
        ["Services Agreement\nPayment is due within 30 days of invoice.\nThe term is 12 months.\nSupport hours are 9 to 5."],
    )
    v2 = add_doc(
        "v2.pdf",
        ["Services Agreement\nPayment is due within 45 days of invoice.\nThe term is 12 months.\nA late fee of 2 percent applies."],
    )
    res = cmp.compare_documents(ctx, [v1, v2], mode="versions", fields=[])
    d = res.version_diff
    assert res.mode == "versions" and 0 < d.similarity < 1
    changed = [c for c in d.changes if c.kind == "changed"]
    assert any("30 days" in (c.a_text or "") and "45 days" in (c.b_text or "") for c in changed)
    assert all(c.a_page == 1 for c in changed)
    texts = " ".join((c.a_text or "") + " " + (c.b_text or "") for c in d.changes)
    assert "Support hours" in texts and "late fee" in texts
    assert "The term is 12 months" not in texts  # unchanged text not reported


def test_identical_versions_have_no_changes(ctx, add_doc):
    v1 = add_doc("v1.pdf", ["Same text on this page.\nAnd another line."])
    v2 = add_doc("v2.pdf", ["Same text on this page.\nAnd another line.\n"] )
    # different bytes so they are not rejected as duplicates
    assert v1.id != v2.id
    res = cmp.compare_documents(ctx, [v1, v2], mode="versions", fields=[])
    assert res.version_diff.changes == [] and res.version_diff.similarity == 1.0
