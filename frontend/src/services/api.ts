import type {
  AnalysisListItem, AnalysisOptions, AnalysisRecord, AnalysisRequest, Collection, Dashboard,
  DocumentItem, LLMCheck, SettingsResponse, UploadResponse,
} from "../types";

const BASE = ((import.meta as any).env?.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

/** Error with a user-presentable message (taken from the backend's {error:{message}} body). */
export class ApiError extends Error {
  constructor(message: string, public status = 0, public code = "") {
    super(message);
  }
}

async function toError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.message) return new ApiError(body.error.message, res.status, body.error.code);
    if (typeof body?.detail === "string") return new ApiError(body.detail, res.status);
  } catch {
    /* not JSON */
  }
  return new ApiError(`Request failed (HTTP ${res.status}).`, res.status);
}

async function raw(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "Cannot reach the backend. Make sure it is running (uvicorn app.main:app --port 8000) and try again."
    );
  }
  if (!res.ok) throw await toError(res);
  return res;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await raw(path, init);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

/** Save a Response body as a file download in the browser. */
async function download(res: Response, fallbackName: string): Promise<string> {
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const name = /filename="?([^";]+)"?/i.exec(disposition)?.[1] ?? fallbackName;
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return name;
}

export const api = {
  dashboard: () => json<Dashboard>("/api/dashboard"),

  listDocuments: (collectionId?: number) =>
    json<DocumentItem[]>(`/api/documents${collectionId ? `?collection_id=${collectionId}` : ""}`),
  uploadDocuments: (files: File[], collectionId?: number | null) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    if (collectionId) form.append("collection_id", String(collectionId));
    return json<UploadResponse>("/api/documents/upload", { method: "POST", body: form });
  },
  deleteDocument: (id: number) => json<void>(`/api/documents/${id}`, { method: "DELETE" }),
  moveDocument: (id: number, collectionId: number | null) =>
    json<DocumentItem>(`/api/documents/${id}`, send("PATCH", { collection_id: collectionId })),
  previewDocument: (id: number) =>
    json<{ document: DocumentItem; preview: string; truncated: boolean }>(`/api/documents/${id}/preview`),

  listCollections: () => json<Collection[]>("/api/collections"),
  createCollection: (name: string, description = "") => json<Collection>("/api/collections", send("POST", { name, description })),
  deleteCollection: (id: number) => json<void>(`/api/collections/${id}`, { method: "DELETE" }),

  options: () => json<AnalysisOptions>("/api/analysis/options"),
  run: (req: AnalysisRequest) => json<AnalysisRecord>("/api/analyses/run", send("POST", req)),
  listAnalyses: (operation?: string) => json<AnalysisListItem[]>(`/api/analyses${operation ? `?operation=${operation}` : ""}`),
  getAnalysis: (id: number) => json<AnalysisRecord>(`/api/analyses/${id}`),
  deleteAnalysis: (id: number) => json<void>(`/api/analyses/${id}`, { method: "DELETE" }),
  exportAnalysis: async (id: number, format: "json" | "csv") =>
    download(await raw(`/api/analyses/${id}/export?format=${format}`), `analysis_${id}.${format}`),
  report: async (body: { analysis_ids?: number[]; document_ids?: number[]; title?: string }) =>
    download(await raw("/api/reports", send("POST", body)), "report.pdf"),

  getSettings: () => json<SettingsResponse>("/api/settings"),
  updateSettings: (values: Record<string, unknown>) => json<SettingsResponse>("/api/settings", send("PUT", values)),
  resetSettings: () => json<SettingsResponse>("/api/settings/reset", send("POST")),
  testLlm: () => json<LLMCheck>("/api/settings/test-llm", send("POST")),
};
