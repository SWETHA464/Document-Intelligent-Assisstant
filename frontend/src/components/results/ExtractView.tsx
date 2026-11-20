import type { ExtractionResult } from "../../types";
import { Badge, SourceChip, Warnings } from "../Common";

/** One table per document: field, extracted value (or "Not specified"), status, source. */
export default function ExtractView({ result }: { result: ExtractionResult }) {
  return (
    <div className="stack">
      <Warnings items={result.warnings} />
      {result.documents.map((doc) => (
        <section key={doc.document_id} className="card inner">
          <h3>{doc.document_name}</h3>
          <Warnings items={doc.warnings} />
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Field</th><th>Value</th><th>Status</th><th>Source</th></tr></thead>
              <tbody>
                {doc.fields.map((f) => (
                  <tr key={f.key}>
                    <th scope="row">{f.label}</th>
                    <td>{f.value ? f.value : <span className="muted">Not specified</span>}{f.note && <div className="muted small">{f.note}</div>}</td>
                    <td><Badge value={f.status} /></td>
                    <td>{f.source ? <SourceChip source={f.source} /> : <span className="muted">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
