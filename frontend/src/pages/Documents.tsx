import { useRef, useState } from "react";
import { api } from "../services/api";
import { useLoad } from "../hooks/useAsync";
import { useToast } from "../hooks/useToast";
import { Badge, EmptyState, ErrorBox, PageHeader, Spinner } from "../components/Common";
import { formatBytes, formatDate } from "../utils";
import type { UploadItem } from "../types";

export default function Documents() {
  const toast = useToast();
  const docs = useLoad(() => api.listDocuments());
  const cols = useLoad(api.listCollections);
  const [target, setTarget] = useState<number | "">("");
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadItem[]>([]);
  const [dragging, setDragging] = useState(false);
  const [newCollection, setNewCollection] = useState("");
  const [preview, setPreview] = useState<{ name: string; text: string; truncated: boolean } | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const refresh = () => { void docs.reload(); void cols.reload(); };

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    try {
      const res = await api.uploadDocuments(files, target === "" ? null : target);
      setResults(res.items);
      const ok = res.counts.ready ?? 0;
      if (ok) toast.success(`${ok} document${ok > 1 ? "s" : ""} processed.`);
      const bad = res.items.length - ok;
      if (bad) toast.error(`${bad} file${bad > 1 ? "s were" : " was"} not added. See details below.`);
      refresh();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setUploading(false);
      if (input.current) input.current.value = "";
    }
  };

  const remove = async (id: number, name: string) => {
    if (!window.confirm(`Remove "${name}"? Saved analyses keep their results.`)) return;
    try { await api.deleteDocument(id); toast.success("Document removed."); refresh(); } catch (e) { toast.error((e as Error).message); }
  };
  const move = async (id: number, value: string) => {
    try { await api.moveDocument(id, value === "" ? null : Number(value)); refresh(); } catch (e) { toast.error((e as Error).message); }
  };
  const addCollection = async () => {
    if (!newCollection.trim()) return;
    try { await api.createCollection(newCollection.trim()); setNewCollection(""); toast.success("Collection created."); refresh(); } catch (e) { toast.error((e as Error).message); }
  };
  const removeCollection = async (id: number, name: string) => {
    if (!window.confirm(`Delete collection "${name}"? Its documents are kept.`)) return;
    try { await api.deleteCollection(id); refresh(); } catch (e) { toast.error((e as Error).message); }
  };
  const showPreview = async (id: number, name: string) => {
    try { const p = await api.previewDocument(id); setPreview({ name, text: p.preview, truncated: p.truncated }); } catch (e) { toast.error((e as Error).message); }
  };

  const list = docs.data ?? [];
  const collections = cols.data ?? [];

  return (
    <>
      <PageHeader title="Documents" subtitle="Upload PDF or TXT files and organise them into collections for analysis." />

      <section className="card">
        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); void upload(Array.from(e.dataTransfer.files)); }}
        >
          {uploading ? <Spinner label="Uploading and extracting text…" /> : (
            <>
              <strong>Drag & drop files here</strong>
              <span className="muted">PDF or TXT, up to the configured size limit. Scanned (image-only) PDFs are not supported yet.</span>
              <div className="row">
                <button className="btn btn-primary" onClick={() => input.current?.click()}>Choose files</button>
                <label className="inline">Add to collection
                  <select value={String(target)} onChange={(e) => setTarget(e.target.value === "" ? "" : Number(e.target.value))}>
                    <option value="">None</option>
                    {collections.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                </label>
              </div>
              <input ref={input} type="file" multiple accept=".pdf,.txt,application/pdf,text/plain" hidden onChange={(e) => void upload(Array.from(e.target.files ?? []))} />
            </>
          )}
        </div>
        {results.some((r) => r.outcome !== "ready" || r.warnings.length) && (
          <ul className="upload-results">
            {results.filter((r) => r.outcome !== "ready" || r.warnings.length).map((r, i) => (
              <li key={i} className={r.outcome === "ready" ? "warn" : "bad"}>
                <strong>{r.filename}</strong> — {r.outcome === "duplicate" ? "already uploaded. " : r.outcome === "rejected" ? "rejected. " : r.outcome === "failed" ? "processing failed. " : ""}
                {r.error ?? r.warnings.join(" ")}
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="two-col wide-left">
        <section className="card">
          <h2 className="section-title">Uploaded documents</h2>
          {docs.error && <ErrorBox message={docs.error} onRetry={docs.reload} />}
          {docs.loading && !docs.data ? <Spinner label="Loading…" /> : list.length === 0 ? (
            <EmptyState title="No documents yet">Upload your first document above.</EmptyState>
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Name</th><th>Type</th><th>Size</th><th>Pages</th><th>Status</th><th>Collection</th><th /></tr></thead>
                <tbody>
                  {list.map((d) => (
                    <tr key={d.id}>
                      <td><div className="ellipsis wide" title={d.filename}>{d.filename}</div><small className="muted">{formatDate(d.created_at)}</small></td>
                      <td>{d.file_type.toUpperCase()}</td>
                      <td>{formatBytes(d.size_bytes)}</td>
                      <td>{d.status === "ready" ? d.page_count || "—" : "—"}</td>
                      <td><Badge value={d.status} />{d.error && <div className="small bad-text">{d.error}</div>}</td>
                      <td>
                        <select aria-label={`Collection for ${d.filename}`} value={d.collection_id ?? ""} onChange={(e) => void move(d.id, e.target.value)}>
                          <option value="">None</option>
                          {collections.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                        </select>
                      </td>
                      <td className="nowrap">
                        {d.status === "ready" && <button className="btn btn-sm btn-ghost" onClick={() => void showPreview(d.id, d.filename)}>Preview</button>}
                        <button className="btn btn-sm btn-danger" onClick={() => void remove(d.id, d.filename)}>Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <h2 className="section-title">Collections</h2>
          <div className="row">
            <input type="text" placeholder="New collection name" value={newCollection} maxLength={80} onChange={(e) => setNewCollection(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void addCollection()} />
            <button className="btn" onClick={() => void addCollection()}>Add</button>
          </div>
          {collections.length === 0 ? <p className="muted">Group related documents (for example one collection per tender).</p> : (
            <ul className="plain-list">
              {collections.map((c) => (
                <li key={c.id}>
                  <span className="grow">{c.name} <small className="muted">({c.document_count})</small></span>
                  <button className="btn btn-sm btn-ghost" onClick={() => void removeCollection(c.id, c.name)}>Delete</button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <div className="modal" role="dialog" aria-label="Extracted text preview" onClick={(e) => e.stopPropagation()}>
            <header><h3>{preview.name}</h3><button className="btn btn-sm" onClick={() => setPreview(null)}>Close</button></header>
            <p className="muted small">Start of the extracted text{preview.truncated ? " (truncated)" : ""}. Use this to check extraction quality.</p>
            <pre>{preview.text || "(no text)"}</pre>
          </div>
        </div>
      )}
    </>
  );
}
