export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export const OPERATION_LABEL: Record<string, string> = {
  qa: "Question",
  compare: "Comparison",
  extract: "Extraction",
  summarize: "Summary",
};

export const sourceLabel = (s: { document_name: string; page_number: number | null }) =>
  s.page_number ? `${s.document_name}, p. ${s.page_number}` : s.document_name;
