import { useEffect, useState } from "react";
import { api } from "../services/api";
import { useLoad } from "../hooks/useAsync";
import { useToast } from "../hooks/useToast";
import { ErrorBox, PageHeader, Spinner } from "../components/Common";
import type { LLMCheck } from "../types";

export default function Settings() {
  const toast = useToast();
  const { data, error, loading, reload, setData } = useLoad(api.getSettings);
  const [form, setForm] = useState({ llm_provider: "ollama", llm_model: "", llm_base_url: "", llm_temperature: 0, embedding_provider: "ollama", embedding_model: "" });
  const [saving, setSaving] = useState(false);
  const [checking, setChecking] = useState(false);
  const [check, setCheck] = useState<LLMCheck | null>(null);

  useEffect(() => {
    if (!data) return;
    const s = data.settings;
    setForm({ llm_provider: s.llm_provider, llm_model: s.llm_model, llm_base_url: s.llm_base_url, llm_temperature: s.llm_temperature, embedding_provider: s.embedding_provider, embedding_model: s.embedding_model });
  }, [data]);

  const set = (k: string, v: string | number) => setForm((f) => ({ ...f, [k]: v }));

  const save = async () => {
    setSaving(true);
    try { setData(await api.updateSettings(form)); setCheck(null); toast.success("Settings saved."); } catch (e) { toast.error((e as Error).message); } finally { setSaving(false); }
  };
  const reset = async () => {
    try { setData(await api.resetSettings()); setCheck(null); toast.success("Reverted to values from .env."); } catch (e) { toast.error((e as Error).message); }
  };
  const test = async () => {
    setChecking(true);
    try { setCheck(await api.testLlm()); } catch (e) { toast.error((e as Error).message); } finally { setChecking(false); }
  };

  if (loading && !data) return <Spinner label="Loading settings…" />;
  if (error && !data) return <ErrorBox message={error} onRetry={reload} />;
  const s = data?.settings;

  return (
    <>
      <PageHeader title="Settings" subtitle="Choose the language model and embedding model. Changes apply immediately and are stored locally." />
      <div className="two-col">
        <section className="card">
          <h2 className="section-title">Model configuration</h2>
          <div className="form-grid">
            <label>LLM provider
              <select value={form.llm_provider} onChange={(e) => set("llm_provider", e.target.value)}>
                <option value="ollama">Ollama (local, free)</option>
                <option value="openai">OpenAI-compatible API</option>
              </select>
            </label>
            <label>LLM model<input type="text" value={form.llm_model} onChange={(e) => set("llm_model", e.target.value)} /></label>
            <label>LLM base URL<input type="text" value={form.llm_base_url} onChange={(e) => set("llm_base_url", e.target.value)} /></label>
            <label>Temperature<input type="number" min={0} max={1.5} step={0.1} value={form.llm_temperature} onChange={(e) => set("llm_temperature", Number(e.target.value))} /></label>
            <label>Embedding provider
              <select value={form.embedding_provider} onChange={(e) => set("embedding_provider", e.target.value)}>
                <option value="ollama">Ollama</option>
                <option value="sentence_transformers">sentence-transformers</option>
                <option value="hashing">Hashing (offline, not semantic)</option>
              </select>
            </label>
            <label>Embedding model<input type="text" value={form.embedding_model} onChange={(e) => set("embedding_model", e.target.value)} /></label>
          </div>
          <p className="muted small">API keys are never entered here. For an OpenAI-compatible provider set <code>LLM_API_KEY</code> in <code>backend/.env</code>{s?.llm_has_api_key ? " (a key is configured)." : " (none configured)."}</p>
          <div className="row">
            <button className="btn btn-primary" onClick={() => void save()} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
            <button className="btn" onClick={() => void test()} disabled={checking}>{checking ? <Spinner label="Testing…" /> : "Test model connection"}</button>
            <button className="btn btn-ghost" onClick={() => void reset()}>Reset to .env</button>
          </div>
          {check && (check.ok
            ? <div className="alert alert-ok">Connected to <strong>{check.model}</strong>. Reply: “{check.reply}”</div>
            : <div className="alert alert-error">{check.error}</div>)}
        </section>

        <section className="card">
          <h2 className="section-title">Set up Ollama (default)</h2>
          <ol className="steps">
            <li>Install Ollama for Windows from <a href="https://ollama.com/download" target="_blank" rel="noreferrer">ollama.com/download</a>. It starts a local service on port 11434.</li>
            <li>Open PowerShell and download the models:<pre>ollama pull llama3.2{"\n"}ollama pull nomic-embed-text</pre></li>
            <li>Check they are installed: <code>ollama list</code>.</li>
            <li>Press <strong>Test model connection</strong> here. If it fails, make sure Ollama is running and the model name matches <code>ollama list</code>.</li>
          </ol>
          <p className="muted small">No GPU? A smaller model is faster; <code>llama3.2</code> (3B) runs on most laptops but may be slow and less accurate on long documents. Larger models give better extraction. Defaults can also be set in <code>backend/.env</code>.</p>
          {s && (
            <dl className="kv">
              <dt>Chunk size</dt><dd>{s.chunk_size} chars (overlap {s.chunk_overlap})</dd>
              <dt>Retrieved passages</dt><dd>{s.retrieval_top_k}</dd>
              <dt>Hybrid weight (semantic)</dt><dd>{s.hybrid_alpha}</dd>
              <dt>Max upload</dt><dd>{s.max_upload_mb} MB</dd>
            </dl>
          )}
          <p className="muted small">Retrieval values are changed in <code>backend/.env</code> and need a backend restart. Changing the embedding model re-embeds documents automatically on the next question.</p>
        </section>
      </div>
    </>
  );
}
