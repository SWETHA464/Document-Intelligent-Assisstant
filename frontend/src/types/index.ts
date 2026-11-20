// Mirrors the backend Pydantic schemas (backend/app/schemas).

export type Operation = "qa" | "compare" | "extract" | "summarize";
export type SummaryType = "general" | "executive" | "key_points" | "risks" | "dates_values_obligations";

export interface DocumentItem {
  id: number;
  collection_id: number | null;
  filename: string;
  file_type: string;
  size_bytes: number;
  status: "processing" | "ready" | "failed" | string;
  error: string | null;
  page_count: number;
  char_count: number;
  chunk_count: number;
  created_at: string;
}

export interface Collection {
  id: number;
  name: string;
  description: string;
  created_at: string;
  document_count: number;
}

export interface UploadItem {
  filename: string;
  outcome: "ready" | "failed" | "rejected" | "duplicate";
  document: DocumentItem | null;
  error: string | null;
  warnings: string[];
}
export interface UploadResponse {
  items: UploadItem[];
  counts: Record<string, number>;
}

export interface Source {
  source_id: string;
  document_id: number;
  document_name: string;
  page_number: number | null;
  chunk_id: number;
  snippet: string;
}

export interface FieldSpec {
  key?: string | null;
  label: string;
  description?: string;
}

export interface QAResult {
  question: string;
  answer: string;
  answerable: boolean;
  grounded: boolean;
  sources: Source[];
  retrieval_mode: string;
  warnings: string[];
}

export type FieldStatus = "found" | "unverified" | "not_found" | "error";

export interface ExtractedField {
  key: string;
  label: string;
  value: string | null;
  status: FieldStatus;
  source: Source | null;
  evidence: string | null;
  note: string | null;
}
export interface DocumentExtraction {
  document_id: number;
  document_name: string;
  fields: ExtractedField[];
  warnings: string[];
}
export interface ExtractionResult {
  preset: string | null;
  fields: FieldSpec[];
  documents: DocumentExtraction[];
  warnings: string[];
}

export type Verdict = "match" | "differs" | "missing_in_some" | "not_found_in_any" | "undetermined";
export interface ComparisonCell {
  document_id: number;
  document_name: string;
  value: string | null;
  status: FieldStatus;
  source: Source | null;
}
export interface ComparisonRow {
  key: string;
  label: string;
  cells: ComparisonCell[];
  verdict: Verdict;
  note: string | null;
}
export interface Finding {
  kind: "difference" | "missing" | "not_found" | "match" | "warning";
  text: string;
  sources: Source[];
}
export interface VersionChange {
  kind: "changed" | "removed" | "added";
  a_text: string | null;
  b_text: string | null;
  a_page: number | null;
  b_page: number | null;
}
export interface ComparisonResult {
  mode: "fields" | "versions";
  documents: { id: number; name: string }[];
  rows: ComparisonRow[];
  findings: Finding[];
  counts: Record<string, number>;
  version_diff: { similarity: number; changes: VersionChange[]; counts: Record<string, number>; truncated: boolean } | null;
  warnings: string[];
}

export interface SummaryItem {
  text: string;
  kind: "stated" | "interpretation";
  sources: Source[];
  note: string | null;
}
export interface SummaryResult {
  summary_type: SummaryType;
  documents: {
    document_id: number;
    document_name: string;
    sections: { heading: string; items: SummaryItem[] }[];
    warnings: string[];
  }[];
}

export interface AnalysisRecord {
  id: number | null;
  operation: Operation;
  title: string;
  document_ids: number[];
  document_names: Record<string, string>;
  params: Record<string, unknown>;
  result: Record<string, unknown>;
  created_at: string | null;
}
export interface AnalysisListItem {
  id: number;
  operation: Operation;
  title: string;
  document_ids: number[];
  document_names: Record<string, string>;
  created_at: string;
}

export interface AnalysisRequest {
  operation: "auto" | Operation;
  document_ids: number[];
  question?: string;
  preset?: string;
  fields?: FieldSpec[];
  compare_mode?: "fields" | "versions";
  summary_type?: SummaryType;
  save?: boolean;
  title?: string;
}

export interface AnalysisOptions {
  presets: { name: string; label: string; fields: FieldSpec[] }[];
  summary_types: { value: SummaryType; label: string }[];
}

export interface Dashboard {
  documents: {
    total: number;
    by_status: Record<string, number>;
    total_pages: number;
    recent: { id: number; filename: string; status: string; created_at: string }[];
  };
  collections: number;
  analyses: { total: number; by_operation: Record<string, number>; recent: AnalysisListItem[] };
  model: { llm_provider: string; llm_model: string; embedding_provider: string; embedding_model: string };
}

export interface AppSettings {
  llm_provider: string;
  llm_model: string;
  llm_base_url: string;
  llm_temperature: number;
  llm_num_ctx: number;
  llm_has_api_key: boolean;
  embedding_provider: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  retrieval_top_k: number;
  hybrid_alpha: number;
  max_upload_mb: number;
}
export interface SettingsResponse {
  settings: AppSettings;
  editable: string[];
}
export interface LLMCheck {
  ok: boolean;
  model: string;
  reply?: string;
  error?: string;
  code?: string;
}
