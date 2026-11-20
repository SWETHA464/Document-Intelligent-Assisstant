import type { SummaryResult } from "../../types";
import { SourceChip, Warnings } from "../Common";

/** Sections of items. "Stated" items come from the text; "interpretation" items are the model's own reading. */
export default function SummaryView({ result }: { result: SummaryResult }) {
  return (
    <div className="stack">
      <div className="legend">
        <span className="tag tag-stated">Stated in document</span>
        <span className="tag tag-interp">AI interpretation</span>
      </div>
      {result.documents.map((doc) => (
        <section key={doc.document_id} className="card inner">
          <h3>{doc.document_name}</h3>
          <Warnings items={doc.warnings} />
          {doc.sections.length === 0 && <p className="muted">No summary content could be produced from this document.</p>}
          {doc.sections.map((sec, i) => (
            <div key={i} className="summary-section">
              <h4>{sec.heading}</h4>
              <ul>
                {sec.items.map((item, j) => (
                  <li key={j}>
                    <span className={`tag ${item.kind === "stated" ? "tag-stated" : "tag-interp"}`}>{item.kind === "stated" ? "Stated" : "Interpretation"}</span>
                    <div>
                      {item.text}
                      {item.note && <div className="muted small">{item.note}</div>}
                      {item.sources.map((s) => <SourceChip key={s.source_id + s.chunk_id} source={s} />)}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}
