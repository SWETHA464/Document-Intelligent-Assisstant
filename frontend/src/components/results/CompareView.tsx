import type { ComparisonResult } from "../../types";
import { Badge, SourceChip, Warnings } from "../Common";

const FINDING_LABEL: Record<string, string> = {
  difference: "Difference", missing: "Missing", not_found: "Not found", match: "Match", warning: "Warning",
};

export default function CompareView({ result }: { result: ComparisonResult }) {
  if (result.mode === "versions" && result.version_diff) return <VersionDiffView result={result} />;
  return (
    <div className="stack">
      <Warnings items={result.warnings} />
      <div className="stat-row">
        {Object.entries(result.counts).map(([k, v]) => (
          <div className="stat" key={k}><strong>{v}</strong><span>{k.replace(/_/g, " ")}</span></div>
        ))}
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Field</th>
              {result.documents.map((d) => <th key={d.id}>{d.name}</th>)}
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row) => (
              <tr key={row.key}>
                <th scope="row">{row.label}</th>
                {row.cells.map((c) => (
                  <td key={c.document_id}>
                    {c.value ? <div>{c.value}</div> : <span className="muted">Not specified</span>}
                    {c.status === "unverified" && <Badge value="unverified" />}
                    {c.source && <SourceChip source={c.source} />}
                  </td>
                ))}
                <td>
                  <Badge value={row.verdict} />
                  {row.note && <div className="muted small">{row.note}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.findings.length > 0 && (
        <section>
          <h3>Key findings</h3>
          <ul className="findings">
            {result.findings.map((f, i) => (
              <li key={i} className={`finding finding-${f.kind}`}>
                <span className="finding-kind">{FINDING_LABEL[f.kind] ?? f.kind}</span>
                <div>
                  {f.text}
                  {f.sources.map((s) => <SourceChip key={s.source_id + s.chunk_id} source={s} />)}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function VersionDiffView({ result }: { result: ComparisonResult }) {
  const diff = result.version_diff!;
  const [a, b] = result.documents;
  return (
    <div className="stack">
      <Warnings items={result.warnings} />
      <p>
        Text similarity between <strong>{a?.name}</strong> and <strong>{b?.name}</strong>: <strong>{Math.round(diff.similarity * 100)}%</strong>.{" "}
        {Object.entries(diff.counts).map(([k, v]) => `${v} ${k}`).join(", ")}.
      </p>
      {diff.truncated && <div className="alert alert-warn">Only the first changes are shown.</div>}
      {diff.changes.length === 0 && <p className="muted">No differences found.</p>}
      <div className="table-wrap">
        <table className="data-table diff">
          <thead><tr><th>Change</th><th>{a?.name}</th><th>{b?.name}</th></tr></thead>
          <tbody>
            {diff.changes.map((c, i) => (
              <tr key={i}>
                <td><span className={`badge badge-${c.kind === "changed" ? "warn" : c.kind === "added" ? "good" : "bad"}`}>{c.kind}</span></td>
                <td>{c.a_text ? <>{c.a_text}{c.a_page && <small className="muted"> (p. {c.a_page})</small>}</> : <span className="muted">—</span>}</td>
                <td>{c.b_text ? <>{c.b_text}{c.b_page && <small className="muted"> (p. {c.b_page})</small>}</> : <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
