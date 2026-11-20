import type { Collection, DocumentItem } from "../types";
import { Badge } from "./Common";
import { formatBytes } from "../utils";

interface Props {
  documents: DocumentItem[];
  collections: Collection[];
  selected: number[];
  onChange: (ids: number[]) => void;
  collectionFilter: number | "all";
  onCollectionFilter: (v: number | "all") => void;
}

/** Checkbox list of documents. Only processed ("ready") documents can be selected. */
export default function DocumentPicker({ documents, collections, selected, onChange, collectionFilter, onCollectionFilter }: Props) {
  const visible = documents.filter((d) => collectionFilter === "all" || d.collection_id === collectionFilter);
  const selectable = visible.filter((d) => d.status === "ready");
  const toggle = (id: number) => onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  const allSelected = selectable.length > 0 && selectable.every((d) => selected.includes(d.id));

  return (
    <div className="picker">
      <div className="picker-head">
        <select aria-label="Filter by collection" value={String(collectionFilter)} onChange={(e) => onCollectionFilter(e.target.value === "all" ? "all" : Number(e.target.value))}>
          <option value="all">All documents</option>
          {collections.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.document_count})</option>)}
        </select>
        <button
          className="btn btn-sm"
          disabled={selectable.length === 0}
          onClick={() => onChange(allSelected ? selected.filter((id) => !selectable.some((d) => d.id === id)) : Array.from(new Set([...selected, ...selectable.map((d) => d.id)])))}
        >
          {allSelected ? "Clear" : "Select all"}
        </button>
      </div>
      <ul className="picker-list">
        {visible.map((d) => (
          <li key={d.id}>
            <label className={d.status !== "ready" ? "disabled" : ""}>
              <input type="checkbox" disabled={d.status !== "ready"} checked={selected.includes(d.id)} onChange={() => toggle(d.id)} />
              <span className="picker-name" title={d.filename}>{d.filename}</span>
              {d.status !== "ready" ? <Badge value={d.status} /> : <small className="muted">{d.page_count ? `${d.page_count} p` : formatBytes(d.size_bytes)}</small>}
            </label>
          </li>
        ))}
        {visible.length === 0 && <li className="muted pad">No documents here yet.</li>}
      </ul>
      <div className="picker-foot muted">{selected.length} selected</div>
    </div>
  );
}
