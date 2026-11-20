import type { QAResult } from "../../types";
import { SourceList, Warnings } from "../Common";

/** Chat-style answer: the question as a bubble, the answer (or "not enough information") below it. */
export default function QAView({ result }: { result: QAResult }) {
  return (
    <div className="chat">
      <div className="bubble bubble-user">{result.question}</div>
      <div className={`bubble bubble-ai ${result.answerable ? "" : "bubble-noinfo"}`}>
        {!result.answerable && <div className="bubble-tag">Not enough information</div>}
        {result.answerable && !result.grounded && <div className="bubble-tag bubble-tag-warn">Unverified: no valid source cited</div>}
        <p>{result.answer}</p>
        <SourceList sources={result.sources} />
        <Warnings items={result.warnings} />
        <div className="muted small">Retrieval: {result.retrieval_mode}</div>
      </div>
    </div>
  );
}
