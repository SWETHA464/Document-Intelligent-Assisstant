import { useState } from "react";
import type { AnalysisRecord } from "../types";
import { api } from "../services/api";
import { useToast } from "../hooks/useToast";
import { Spinner } from "./Common";

/** Export (JSON/CSV) and PDF report buttons for a saved analysis. */
export default function ResultActions({ record }: { record: AnalysisRecord }) {
  const toast = useToast();
  const [busy, setBusy] = useState<string | null>(null);
  if (record.id == null) return null;
  const id = record.id;
  const csvOk = record.operation === "extract" || record.operation === "compare";

  const run = async (key: string, fn: () => Promise<string>, okText: (n: string) => string) => {
    setBusy(key);
    try { toast.success(okText(await fn())); } catch (e) { toast.error((e as Error).message); } finally { setBusy(null); }
  };

  return (
    <div className="actions">
      <button className="btn" disabled={!!busy} onClick={() => run("pdf", () => api.report({ analysis_ids: [id], title: record.title }), (n) => `Report downloaded: ${n}`)}>
        {busy === "pdf" ? <Spinner label="Building PDF…" /> : "Download PDF report"}
      </button>
      <button className="btn btn-ghost" disabled={!!busy} onClick={() => run("json", () => api.exportAnalysis(id, "json"), (n) => `Saved ${n}`)}>JSON</button>
      {csvOk && <button className="btn btn-ghost" disabled={!!busy} onClick={() => run("csv", () => api.exportAnalysis(id, "csv"), (n) => `Saved ${n}`)}>CSV</button>}
    </div>
  );
}
