import { useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { useLoad } from "../hooks/useAsync";
import { useToast } from "../hooks/useToast";
import { EmptyState, ErrorBox, PageHeader, Spinner } from "../components/Common";
import DocumentPicker from "../components/DocumentPicker";
import ResultView from "../components/ResultView";
import ResultActions from "../components/ResultActions";
import type { AnalysisRecord, AnalysisRequest, FieldSpec, Operation, SummaryType } from "../types";

const TABS: { op: Operation; label: string; hint: string }[] = [
  { op: "qa", label: "Ask", hint: "Ask a question across the selected documents." },
  { op: "compare", label: "Compare", hint: "Compare 2 or more documents in a structured table." },
  { op: "extract", label: "Extract", hint: "Pull structured fields out of each document." },
  { op: "summarize", label: "Summarise", hint: "Grounded summaries; stated facts are separated from interpretation." },
];

/** "Label" or "Label: what to look for", one per line. */
function parseFields(text: string): FieldSpec[] {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((line) => {
    const [label, ...rest] = line.split(":");
    return { label: label.trim().slice(0, 80), description: rest.join(":").trim().slice(0, 300) };
  }).filter((f) => f.label);
}

export default function Analysis({ initialOp, initialDocs }: { initialOp: Operation | null; initialDocs: number[] }) {
  const toast = useToast();
  const docs = useLoad(() => api.listDocuments());
  const cols = useLoad(api.listCollections);
  const opts = useLoad(api.options);

  const [op, setOp] = useState<Operation>(initialOp ?? "qa");
  const [selected, setSelected] = useState<number[]>(initialDocs);
  const [collectionFilter, setCollectionFilter] = useState<number | "all">("all");
  const [question, setQuestion] = useState("");
  const [focus, setFocus] = useState("");
  const [preset, setPreset] = useState("");
  const [customFields, setCustomFields] = useState("");
  const [compareMode, setCompareMode] = useState<"fields" | "versions">("fields");
  const [summaryType, setSummaryType] = useState<SummaryType>("executive");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisRecord | null>(null);
  const [thread, setThread] = useState<AnalysisRecord[]>([]);
  const [reporting, setReporting] = useState(false);

  useEffect(() => { if (initialOp) setOp(initialOp); }, [initialOp]);
  // Drop selections for documents that no longer exist.
  useEffect(() => {
    if (docs.data) setSelected((s) => s.filter((id) => docs.data!.some((d) => d.id === id && d.status === "ready")));
  }, [docs.data]);

  const needed = op === "compare" ? 2 : 1;
  const tab = TABS.find((t) => t.op === op)!;
  const customParsed = useMemo(() => parseFields(customFields), [customFields]);
  const canRun = selected.length >= needed && !running && (op !== "qa" || question.trim().length > 0);

  const changeOp = (next: Operation) => { setOp(next); setResult(null); setError(null); };

  const build = (): AnalysisRequest => {
    const base: AnalysisRequest = { operation: op, document_ids: selected, save: true };
    if (op === "qa") return { ...base, question: question.trim() };
    if (op === "summarize") return { ...base, summary_type: summaryType };
    const withFields: AnalysisRequest = { ...base, question: focus.trim() || undefined };
    if (customParsed.length) withFields.fields = customParsed;
    else if (preset) withFields.preset = preset;
    if (op === "compare") withFields.compare_mode = compareMode;
    return withFields;
  };

  const run = async () => {
    setRunning(true); setError(null);
    try {
      const rec = await api.run(build());
      setResult(rec);
      if (op === "qa") { setThread((t) => [...t, rec]); setQuestion(""); }
      toast.success("Analysis complete and saved to history.");
    } catch (e) {
      setError((e as Error).message);
    } finally { setRunning(false); }
  };

  const fullReport = async () => {
    setReporting(true);
    try {
      toast.info("Building report: running summary, extraction and comparison…");
      const name = await api.report({ document_ids: selected });
      toast.success(`Report downloaded: ${name}`);
    } catch (e) { toast.error((e as Error).message); } finally { setReporting(false); }
  };

  if (docs.loading && !docs.data) return <Spinner label="Loading documents…" />;
  if (docs.error && !docs.data) return <ErrorBox message={docs.error} onRetry={docs.reload} />;
  const readyDocs = (docs.data ?? []).filter((d) => d.status === "ready");

  if (readyDocs.length === 0) {
    return (
      <>
        <PageHeader title="Document Analysis" />
        <section className="card"><EmptyState title="No processed documents" action={<a className="btn btn-primary" href="#/documents">Go to Documents</a>}>Upload at least one PDF or TXT file to start analysing.</EmptyState></section>
      </>
    );
  }

  const shown = op === "qa" ? thread : result ? [result] : [];

  return (
    <>
      <PageHeader title="Document Analysis" subtitle="Select documents, choose an operation and run it. Results are saved automatically."
        actions={<button className="btn" disabled={selected.length < 1 || reporting} onClick={() => void fullReport()} title="Runs an executive summary, extraction and (for 2+ documents) a comparison, then builds a PDF.">
          {reporting ? <Spinner label="Building report…" /> : "Full PDF report for selection"}
        </button>} />

      <div className="analysis-layout">
        <aside className="card">
          <h2 className="section-title">1. Documents</h2>
          <DocumentPicker documents={docs.data ?? []} collections={cols.data ?? []} selected={selected} onChange={setSelected} collectionFilter={collectionFilter} onCollectionFilter={setCollectionFilter} />
        </aside>

        <section className="card">
          <h2 className="section-title">2. Operation</h2>
          <div className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t.op} role="tab" aria-selected={op === t.op} className={op === t.op ? "tab active" : "tab"} onClick={() => changeOp(t.op)}>{t.label}</button>
            ))}
          </div>
          <p className="muted">{tab.hint}</p>

          {op === "qa" && (
            <div className="field">
              <label htmlFor="q">Your question</label>
              <textarea id="q" rows={3} value={question} maxLength={2000} placeholder="e.g. Which vendor offers the shortest delivery timeline?"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canRun) void run(); }} />
            </div>
          )}

          {(op === "compare" || op === "extract") && (
            <>
              {op === "compare" && (
                <div className="field">
                  <label>Comparison type</label>
                  <div className="row">
                    <label className="inline"><input type="radio" checked={compareMode === "fields"} onChange={() => setCompareMode("fields")} /> Compare fields (pricing, terms…)</label>
                    <label className="inline"><input type="radio" checked={compareMode === "versions"} onChange={() => setCompareMode("versions")} /> Differences between versions</label>
                  </div>
                </div>
              )}
              {(op === "extract" || compareMode === "fields") && (
                <>
                  <div className="field">
                    <label htmlFor="focus">{op === "compare" ? "What to compare (optional)" : "What to extract (optional)"}</label>
                    <input id="focus" type="text" value={focus} placeholder="e.g. compare pricing and payment terms" onChange={(e) => setFocus(e.target.value)} />
                    <small className="muted">Recognised topics (price, payment, delivery, duration…) pick matching fields.</small>
                  </div>
                  <div className="field">
                    <label htmlFor="preset">Field set</label>
                    <select id="preset" value={preset} onChange={(e) => setPreset(e.target.value)} disabled={customParsed.length > 0}>
                      <option value="">Automatic (from the text above, else vendor proposal)</option>
                      {opts.data?.presets.map((p) => <option key={p.name} value={p.name}>{p.label} ({p.fields.length} fields)</option>)}
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="custom">Custom fields (optional, one per line, “Label: what to look for”)</label>
                    <textarea id="custom" rows={3} value={customFields} placeholder={"Warranty period: length of any warranty\nPenalty clause"} onChange={(e) => setCustomFields(e.target.value)} />
                    {customParsed.length > 0 && <small className="muted">{customParsed.length} custom field{customParsed.length > 1 ? "s" : ""} will be used instead of the field set.</small>}
                  </div>
                </>
              )}
              {op === "compare" && compareMode === "versions" && <p className="muted small">Select exactly two documents: the older version first. Order follows your selection.</p>}
            </>
          )}

          {op === "summarize" && (
            <div className="field">
              <label htmlFor="stype">Summary type</label>
              <select id="stype" value={summaryType} onChange={(e) => setSummaryType(e.target.value as SummaryType)}>
                {(opts.data?.summary_types ?? [{ value: "executive", label: "Executive summary" }]).map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
          )}

          <div className="row">
            <button className="btn btn-primary" disabled={!canRun} onClick={() => void run()}>
              {running ? <Spinner label="Analysing… this can take a minute on a local model" /> : `Run ${tab.label.toLowerCase()}`}
            </button>
            {selected.length < needed && <small className="muted">Select at least {needed} document{needed > 1 ? "s" : ""}.</small>}
          </div>
          {error && <ErrorBox message={error} />}
        </section>
      </div>

      {shown.length > 0 && (
        <section className="card results">
          <h2 className="section-title">3. Result</h2>
          {shown.map((rec, i) => (
            <div key={rec.id ?? i} className="result-block">
              {op !== "qa" && <h3>{rec.title}</h3>}
              <ResultView record={rec} />
              {(op !== "qa" || i === shown.length - 1) && <ResultActions record={rec} />}
              {op !== "qa" && rec.id != null && <p className="muted small">Saved as analysis #{rec.id}. <a href={`#/history/${rec.id}`}>Open in history</a></p>}
            </div>
          ))}
        </section>
      )}
    </>
  );
}
