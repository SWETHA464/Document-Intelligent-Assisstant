"""Request/response models for document analysis operations."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import Source
from app.utils.text import slugify

Operation = Literal["qa", "compare", "extract", "summarize"]
SummaryType = Literal["general", "executive", "key_points", "risks", "dates_values_obligations"]

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


# ------------------------------------------------------------- request ----
class FieldSpec(BaseModel):
    """A piece of information to look for. Reusable for any document type."""

    key: str | None = None
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _default_key(self) -> "FieldSpec":
        self.label = self.label.strip()
        if not self.label:
            raise ValueError("label cannot be blank")
        self.key = slugify(self.key or self.label)
        if not _KEY_RE.match(self.key):
            raise ValueError(f"invalid field key '{self.key}'")
        return self


def unique_fields(fields: list[FieldSpec]) -> list[FieldSpec]:
    """Make sure keys are unique within a list (appends _2, _3 ... when needed)."""
    seen: dict[str, int] = {}
    out: list[FieldSpec] = []
    for f in fields:
        key = f.key or slugify(f.label)
        n = seen.get(key, 0) + 1
        seen[key] = n
        out.append(f.model_copy(update={"key": key if n == 1 else f"{key}_{n}"}))
    return out


class AnalysisRequest(BaseModel):
    operation: Literal["auto", "qa", "compare", "extract", "summarize"] = "auto"
    document_ids: list[int] = Field(min_length=1, max_length=20)
    question: str | None = Field(default=None, max_length=2000)
    preset: str | None = Field(default=None, max_length=40)
    fields: list[FieldSpec] | None = Field(default=None, max_length=25)
    compare_mode: Literal["fields", "versions"] = "fields"
    summary_type: SummaryType | None = None
    save: bool = True
    title: str | None = Field(default=None, max_length=200)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


# -------------------------------------------------------------- results ----
class DocumentRef(BaseModel):
    id: int
    name: str


class QAResult(BaseModel):
    question: str
    answer: str
    answerable: bool
    grounded: bool = False  # True when at least one valid source is cited
    sources: list[Source] = []
    retrieval_mode: str = "hybrid"
    warnings: list[str] = []


FieldStatus = Literal["found", "unverified", "not_found", "error"]


class ExtractedField(BaseModel):
    key: str
    label: str
    value: str | None = None
    status: FieldStatus = "not_found"
    source: Source | None = None
    evidence: str | None = None
    note: str | None = None


class DocumentExtraction(BaseModel):
    document_id: int
    document_name: str
    fields: list[ExtractedField]
    warnings: list[str] = []


class ExtractionResult(BaseModel):
    preset: str | None = None
    fields: list[FieldSpec]
    documents: list[DocumentExtraction]
    warnings: list[str] = []


Verdict = Literal["match", "differs", "missing_in_some", "not_found_in_any", "undetermined"]


class ComparisonCell(BaseModel):
    document_id: int
    document_name: str
    value: str | None = None
    status: FieldStatus = "not_found"
    source: Source | None = None


class ComparisonRow(BaseModel):
    key: str
    label: str
    cells: list[ComparisonCell]
    verdict: Verdict
    note: str | None = None


class Finding(BaseModel):
    kind: Literal["difference", "missing", "not_found", "match", "warning"]
    text: str
    sources: list[Source] = []


class VersionChange(BaseModel):
    kind: Literal["changed", "removed", "added"]
    a_text: str | None = None
    b_text: str | None = None
    a_page: int | None = None
    b_page: int | None = None


class VersionDiff(BaseModel):
    similarity: float
    changes: list[VersionChange]
    counts: dict[str, int]
    truncated: bool = False


class ComparisonResult(BaseModel):
    mode: Literal["fields", "versions"]
    documents: list[DocumentRef]
    rows: list[ComparisonRow] = []
    findings: list[Finding] = []
    counts: dict[str, int] = {}
    version_diff: VersionDiff | None = None
    warnings: list[str] = []


class SummaryItem(BaseModel):
    text: str
    kind: Literal["stated", "interpretation"]
    sources: list[Source] = []
    note: str | None = None


class SummarySection(BaseModel):
    heading: str
    items: list[SummaryItem]


class DocumentSummary(BaseModel):
    document_id: int
    document_name: str
    sections: list[SummarySection]
    warnings: list[str] = []


class SummaryResult(BaseModel):
    summary_type: SummaryType
    documents: list[DocumentSummary]


# --------------------------------------------------------------- record ----
class AnalysisRecordOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None  # None when the request set save=false
    operation: Operation
    title: str
    document_ids: list[int]
    document_names: dict[int, str] = {}
    params: dict[str, Any] = {}
    result: dict[str, Any]
    created_at: str | None = None
