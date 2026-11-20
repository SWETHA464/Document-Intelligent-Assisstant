"""HTTP-level tests. Skipped automatically when FastAPI/httpx are not installed
(run `pip install -r requirements-dev.txt` to enable them)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from tests.conftest import VENDOR_A_PAGES, VENDOR_B_PAGES, DownLLM, ScriptedLLM, make_pdf  # noqa: E402


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx), raise_server_exceptions=False)


@pytest.fixture
def pdf(tmp_path):
    def _make(pages):
        return make_pdf(tmp_path / "x.pdf", pages).read_bytes()

    return _make


def upload(client, name, data, ctype="application/pdf", **form):
    return client.post("/api/documents/upload", files=[("files", (name, data, ctype))], data=form)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_list_preview_delete(client, pdf):
    r = upload(client, "a.pdf", pdf(VENDOR_A_PAGES))
    assert r.status_code == 200
    doc = r.json()["items"][0]["document"]
    assert doc["status"] == "ready" and doc["page_count"] == 3
    assert len(client.get("/api/documents").json()) == 1
    assert "Acme" in client.get(f"/api/documents/{doc['id']}/preview").json()["preview"]
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_invalid_empty_corrupt_and_duplicate_uploads(client, pdf):
    r = upload(client, "notes.docx", b"hello", "application/octet-stream")
    assert r.json()["items"][0]["outcome"] == "rejected"
    assert upload(client, "empty.pdf", b"").json()["items"][0]["outcome"] == "rejected"
    bad = upload(client, "broken.pdf", b"%PDF-1.4 definitely not a pdf").json()["items"][0]
    assert bad["outcome"] == "failed" and bad["document"]["status"] == "failed"
    data = pdf(VENDOR_A_PAGES)
    assert upload(client, "a.pdf", data).json()["items"][0]["outcome"] == "ready"
    assert upload(client, "a_copy.pdf", data).json()["items"][0]["outcome"] == "duplicate"


def test_collections(client, pdf):
    c = client.post("/api/collections", json={"name": "Vendors"}).json()
    assert client.post("/api/collections", json={"name": "vendors"}).status_code == 400
    doc = upload(client, "a.pdf", pdf(VENDOR_A_PAGES), collection_id=str(c["id"])).json()["items"][0]["document"]
    assert doc["collection_id"] == c["id"]
    assert client.get("/api/collections").json()[0]["document_count"] == 1
    assert len(client.get(f"/api/documents?collection_id={c['id']}").json()) == 1


def test_qa_with_sources(client, ctx, pdf):
    doc = upload(client, "a.pdf", pdf(VENDOR_A_PAGES)).json()["items"][0]["document"]
    ctx._llm_override = ScriptedLLM(['{"answerable": true, "answer": "Net 30 days.", "citations": ["S1"]}'])
    r = client.post("/api/analyses/run", json={"document_ids": [doc["id"]], "question": "What are the payment terms?"})
    body = r.json()
    assert r.status_code == 200 and body["operation"] == "qa" and body["id"]
    assert body["result"]["sources"][0]["page_number"] == 2
    assert client.get(f"/api/analyses/{body['id']}").json()["result"]["answer"] == "Net 30 days."
    assert [a["id"] for a in client.get("/api/analyses").json()] == [body["id"]]


def test_error_cases(client, ctx, pdf):
    doc = upload(client, "a.pdf", pdf(VENDOR_A_PAGES)).json()["items"][0]["document"]
    assert client.post("/api/analyses/run", json={"document_ids": [999], "question": "x?"}).status_code == 404
    r = client.post("/api/analyses/run", json={"document_ids": [doc["id"]], "operation": "compare"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    assert client.post("/api/analyses/run", json={"document_ids": [doc["id"]], "operation": "translate"}).status_code == 422
    assert client.post("/api/analyses/run", json={"document_ids": []}).status_code == 422
    assert client.get("/api/analyses/12345").status_code == 404
    ctx._llm_override = DownLLM()
    r = client.post("/api/analyses/run", json={"document_ids": [doc["id"]], "question": "What are the payment terms?"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"


def test_compare_extract_summary_export_and_report(client, ctx, pdf):
    ids = [upload(client, n, pdf(p)).json()["items"][0]["document"]["id"] for n, p in (("a.pdf", VENDOR_A_PAGES), ("b.pdf", VENDOR_B_PAGES))]

    def llm(system, prompt):
        if '"fields"' in system:
            value = "Net 30 days" if '"a.pdf"' in prompt else "Net 45 days"
            return '{"fields": [{"key": "payment_terms", "found": true, "value": "%s", "source_id": "S1"}]}' % value
        return '{"sections": [{"heading": "Overview", "items": [{"text": "A proposal.", "kind": "interpretation", "source_ids": []}]}]}'

    ctx._llm_override = ScriptedLLM(llm)
    cmp_ = client.post("/api/analyses/run", json={"document_ids": ids, "operation": "compare", "question": "Compare payment terms"})
    assert cmp_.status_code == 200, cmp_.text
    aid = cmp_.json()["id"]
    csv = client.get(f"/api/analyses/{aid}/export?format=csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    assert client.get(f"/api/analyses/{aid}/export?format=json").json()["operation"] == "compare"
    summ = client.post("/api/analyses/run", json={"document_ids": ids, "operation": "summarize", "summary_type": "executive"})
    assert summ.status_code == 200
    assert client.get(f"/api/analyses/{summ.json()['id']}/export?format=csv").status_code == 400
    rep = client.post("/api/reports", json={"analysis_ids": [aid, summ.json()["id"]], "title": "Vendor review"})
    assert rep.status_code == 200 and rep.content.startswith(b"%PDF") and rep.headers["content-type"] == "application/pdf"
    assert client.post("/api/reports", json={}).status_code == 400
    assert client.post("/api/reports", json={"analysis_ids": [777]}).status_code == 404


def test_settings_roundtrip_and_llm_check(client, ctx):
    assert "llm_model" in client.get("/api/settings").json()["settings"]
    r = client.put("/api/settings", json={"llm_model": "qwen2.5"})
    assert r.json()["settings"]["llm_model"] == "qwen2.5"
    assert client.put("/api/settings", json={"llm_provider": "bogus"}).status_code == 422
    assert client.put("/api/settings", json={}).status_code == 400
    ctx._llm_override = DownLLM()
    check = client.post("/api/settings/test-llm").json()
    assert check["ok"] is False and "offline" in check["error"].lower()
    client.post("/api/settings/reset")
    assert client.get("/api/settings").json()["settings"]["llm_model"] == "llama3.2"


def test_dashboard(client, pdf):
    upload(client, "a.pdf", pdf(VENDOR_A_PAGES))
    d = client.get("/api/dashboard").json()
    assert d["documents"]["total"] == 1 and d["analyses"]["total"] == 0
