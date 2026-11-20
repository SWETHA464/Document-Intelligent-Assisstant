import type { ReactNode } from "react";
import type { Source } from "../types";
import { sourceLabel } from "../utils";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <span className="spinner" />
      {label && <span>{label}</span>}
    </span>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon" aria-hidden>▤</div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="alert alert-error" role="alert">
      <span>{message}</span>
      {onRetry && <button className="btn btn-sm" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Warnings({ items }: { items?: string[] }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="alert alert-warn">
      {items.map((w, i) => <div key={i}>{w}</div>)}
    </div>
  );
}

const STATUS_TEXT: Record<string, string> = {
  ready: "Ready", failed: "Failed", processing: "Processing",
  found: "Found", unverified: "Unverified", not_found: "Not specified", error: "Error",
  match: "Match", differs: "Differs", missing_in_some: "Missing in some", not_found_in_any: "Not found in any", undetermined: "Unable to determine",
};
const STATUS_TONE: Record<string, string> = {
  ready: "good", found: "good", match: "good",
  failed: "bad", error: "bad", differs: "bad",
  processing: "info", unverified: "warn", missing_in_some: "warn", undetermined: "warn",
  not_found: "muted", not_found_in_any: "muted",
};
export function Badge({ value }: { value: string }) {
  return <span className={`badge badge-${STATUS_TONE[value] ?? "muted"}`}>{STATUS_TEXT[value] ?? value}</span>;
}

/** One citation: label with document + page, snippet shown on expand. */
export function SourceChip({ source }: { source: Source }) {
  return (
    <details className="source">
      <summary>
        <span className="source-id">{source.source_id}</span> {sourceLabel(source)}
      </summary>
      {source.snippet && <blockquote>{source.snippet}</blockquote>}
    </details>
  );
}

export function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <div className="sources">
      <div className="sources-title">Sources</div>
      {sources.map((s) => <SourceChip key={s.source_id + s.chunk_id} source={s} />)}
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        {subtitle && <p className="muted">{subtitle}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}
