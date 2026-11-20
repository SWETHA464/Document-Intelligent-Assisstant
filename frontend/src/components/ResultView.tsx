import type { AnalysisRecord, ComparisonResult, ExtractionResult, QAResult, SummaryResult } from "../types";
import QAView from "./results/QAView";
import CompareView from "./results/CompareView";
import ExtractView from "./results/ExtractView";
import SummaryView from "./results/SummaryView";

/** Picks the right layout for the operation: chat, comparison table, fields, or sections. */
export default function ResultView({ record }: { record: AnalysisRecord }) {
  const r = record.result;
  switch (record.operation) {
    case "qa": return <QAView result={r as unknown as QAResult} />;
    case "compare": return <CompareView result={r as unknown as ComparisonResult} />;
    case "extract": return <ExtractView result={r as unknown as ExtractionResult} />;
    case "summarize": return <SummaryView result={r as unknown as SummaryResult} />;
    default: return <p className="muted">Unknown result type.</p>;
  }
}
