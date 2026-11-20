import { useEffect, useState } from "react";
import { api } from "../services/api";
import { useLoad } from "../hooks/useAsync";
import { useToast } from "../hooks/useToast";
import { navigate } from "../hooks/useHashRoute";
import { EmptyState, ErrorBox, PageHeader, Spinner } from "../components/Common";
import ResultView from "../components/ResultView";
import ResultActions from "../components/ResultActions";
import { OPERATION_LABEL, formatDate } from "../utils";
import type { AnalysisRecord } from "../types";

export default function History({ openId }: { openId: number | null }) {
  const toast = useToast();
  const [filter, setFilter] = useState("");
  const list = useLoad(() => api.listAnalyses(filter || undefined), [filter]);
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [loadingRecord, setLoadingRecord] = useState(false);
  const [recordError, setRecordError] = useState<string | null>(null);

  useEffect(() => {
    setRecord(null); setRecordError(null);
    if (openId == null) return;
    setLoadingRecord(true);
    api.getAnalysis(openId).then(setRecord).catch((e) => setRecordError((e as Error).message)).finally(() => setLoadingRecord(false));
  }, [openId]);

  const remove = async (id: number) => {
    if (!window.confirm("Delete this saved analysis?")) return;
    try { await api.deleteAnalysis(id); toast.success("Analysis deleted."); if (openId === id) navigate("/history"); void list.reload(); } catch (e) { toast.error((e as Error).message); }
  };

  if (openId != null) {
    return (
      <>
        <PageHeader title={record?.title ?? "Saved analysis"} subtitle={record ? `${OPERATION_LABEL[record.operation]} · ${formatDate(record.created_at)}` : undefined}
          actions={<><a className="btn" href="#/history">← All analyses</a>{record?.id != null && <button className="btn btn-danger" onClick={() => void remove(record.id!)}>Delete</button>}</>} />
        {loadingRecord && <Spinner label="Loading…" />}
        {recordError && <ErrorBox message={recordError} />}
        {record && (
          <section className="card">
            <p className="muted small">Documents: {Object.values(record.document_names).join(", ") || "—"}</p>
            <ResultActions record={record} />
            <ResultView record={record} />
          </section>
        )}
      </>
    );
  }

  const rows = list.data ?? [];
  return (
    <>
      <PageHeader title="Analysis History" subtitle="Every analysis is saved automatically. Reopen one to review it or download a report."
        actions={
          <select aria-label="Filter by type" value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">All types</option>
            {Object.entries(OPERATION_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        } />
      {list.error && <ErrorBox message={list.error} onRetry={list.reload} />}
      {list.loading && !list.data ? <Spinner label="Loading…" /> : rows.length === 0 ? (
        <section className="card"><EmptyState title="No saved analyses" action={<a className="btn btn-primary" href="#/analysis">Run an analysis</a>}>Run a question, comparison, extraction or summary and it will appear here.</EmptyState></section>
      ) : (
        <section className="card">
          <ul className="history-list">
            {rows.map((a) => (
              <li key={a.id}>
                <a className="history-main" href={`#/history/${a.id}`}>
                  <span className={`op-pill op-${a.operation}`}>{OPERATION_LABEL[a.operation]}</span>
                  <strong className="ellipsis">{a.title}</strong>
                  <small className="muted ellipsis">{Object.values(a.document_names).join(", ")}</small>
                </a>
                <small className="muted nowrap">{formatDate(a.created_at)}</small>
                <button className="btn btn-sm btn-ghost" onClick={() => void remove(a.id)}>Delete</button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
