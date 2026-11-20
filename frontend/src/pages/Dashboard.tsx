import { api } from "../services/api";
import { useLoad } from "../hooks/useAsync";
import { Badge, EmptyState, ErrorBox, PageHeader, Spinner } from "../components/Common";
import { OPERATION_LABEL, formatDate } from "../utils";

const QUICK = [
  { op: "qa", title: "Ask a question", text: "Answers with document and page references." },
  { op: "compare", title: "Compare documents", text: "Side-by-side table of prices, terms and dates." },
  { op: "extract", title: "Extract information", text: "Pull structured fields into a table." },
  { op: "summarize", title: "Summarise", text: "Executive summary, key points, risks, dates." },
];

export default function Dashboard() {
  const { data, error, loading, reload } = useLoad(api.dashboard);
  if (loading && !data) return <Spinner label="Loading dashboard…" />;
  if (error && !data) return <ErrorBox message={error} onRetry={reload} />;
  if (!data) return null;
  const ready = data.documents.by_status.ready ?? 0;
  const failed = data.documents.by_status.failed ?? 0;

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Overview of your documents and recent analysis activity." />
      <div className="stat-grid">
        <div className="card stat-card"><strong>{data.documents.total}</strong><span>Documents</span><small className="muted">{ready} ready{failed ? `, ${failed} failed` : ""}</small></div>
        <div className="card stat-card"><strong>{data.documents.total_pages}</strong><span>Pages indexed</span></div>
        <div className="card stat-card"><strong>{data.collections}</strong><span>Collections</span></div>
        <div className="card stat-card"><strong>{data.analyses.total}</strong><span>Saved analyses</span></div>
      </div>

      <h2 className="section-title">Quick actions</h2>
      <div className="quick-grid">
        {QUICK.map((q) => (
          <a key={q.op} className="card quick" href={`#/analysis?op=${q.op}`}>
            <strong>{q.title}</strong>
            <span className="muted">{q.text}</span>
          </a>
        ))}
      </div>

      <div className="two-col">
        <section className="card">
          <h2 className="section-title">Recent documents</h2>
          {data.documents.recent.length === 0 ? (
            <EmptyState title="No documents yet" action={<a className="btn btn-primary" href="#/documents">Upload documents</a>}>Upload PDF or TXT files to get started.</EmptyState>
          ) : (
            <ul className="plain-list">
              {data.documents.recent.map((d) => (
                <li key={d.id}><span className="grow ellipsis">{d.filename}</span><Badge value={d.status} /></li>
              ))}
            </ul>
          )}
        </section>
        <section className="card">
          <h2 className="section-title">Recent analyses</h2>
          {data.analyses.recent.length === 0 ? (
            <EmptyState title="No analyses yet">Results you run are saved here automatically.</EmptyState>
          ) : (
            <ul className="plain-list">
              {data.analyses.recent.map((a) => (
                <li key={a.id}>
                  <a className="grow ellipsis" href={`#/history/${a.id}`}>{a.title}</a>
                  <small className="muted">{OPERATION_LABEL[a.operation]} · {formatDate(a.created_at)}</small>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <p className="muted small">Model: {data.model.llm_model} ({data.model.llm_provider}) · Embeddings: {data.model.embedding_model} ({data.model.embedding_provider})</p>
    </>
  );
}
