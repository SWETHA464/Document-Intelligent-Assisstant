"""PDF report generation (reportlab).

The report is assembled ONLY from saved analysis results (summaries, extractions, comparisons,
Q&A). It adds no new model output, so it cannot contain claims or citations that were not
already present - and verified - in those results.
"""
from __future__ import annotations

import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.context import AppContext
from app.core.errors import InvalidRequestError, NotFoundError
from app.schemas.analysis import AnalysisRequest, ComparisonResult, ExtractionResult, SummaryResult
from app.services.analysis_service import run_analysis, to_record
from app.services.comparison_service import display_value
from app.services.export_service import STATUS_LABEL
from app.utils.text import truncate

ACCENT = colors.HexColor("#2f4a8a")
MUTED = colors.HexColor("#5b6472")
GRID = colors.HexColor("#c9ced8")
HEAD_BG = colors.HexColor("#e8ecf5")

_FONT_CANDIDATES = [
    # (regular, bold)
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]
_font_cache: dict[str, tuple[str, str, bool]] = {}


def _register_fonts(custom_path: str = "") -> tuple[str, str, bool]:
    """Return (regular, bold, unicode_ok). Falls back to built-in Helvetica (Latin-1 only)."""
    cache_key = custom_path or "-"
    if cache_key in _font_cache:
        return _font_cache[cache_key]
    candidates = ([(custom_path, custom_path)] if custom_path else []) + _FONT_CANDIDATES
    for regular, bold in candidates:
        if regular and os.path.exists(regular):
            try:
                name = "ReportFont" + str(len(_font_cache))
                pdfmetrics.registerFont(TTFont(name, regular))
                bold_name = name
                if bold and os.path.exists(bold):
                    bold_name = name + "-Bold"
                    pdfmetrics.registerFont(TTFont(bold_name, bold))
                pdfmetrics.registerFontFamily(name, normal=name, bold=bold_name, italic=name, boldItalic=bold_name)
                _font_cache[cache_key] = (name, bold_name, True)
                return _font_cache[cache_key]
            except Exception:  # unreadable font file -> try the next candidate
                continue
    _font_cache[cache_key] = ("Helvetica", "Helvetica-Bold", False)
    return _font_cache[cache_key]


def _latin1(text: str) -> str:
    """Replace characters the built-in font cannot draw (only used when no TTF font was found)."""
    swaps = {"\u20b9": "INR ", "\u20ac": "EUR ", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2022": "-"}
    for k, v in swaps.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Styles:
    def __init__(self, font: str, bold: str, unicode_ok: bool) -> None:
        self.unicode_ok = unicode_ok
        base = getSampleStyleSheet()
        self.body = ParagraphStyle("body", parent=base["BodyText"], fontName=font, fontSize=9.5, leading=13.5, alignment=TA_LEFT)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=8, leading=10.5, textColor=MUTED)
        self.cell = ParagraphStyle("cell", parent=self.body, fontSize=8.5, leading=11)
        self.cell_head = ParagraphStyle("cellh", parent=self.cell, fontName=bold)
        self.title = ParagraphStyle("title", parent=self.body, fontName=bold, fontSize=22, leading=27, textColor=ACCENT, spaceAfter=4)
        self.subtitle = ParagraphStyle("sub", parent=self.body, fontSize=10, textColor=MUTED, spaceAfter=10)
        self.h1 = ParagraphStyle("h1", parent=self.body, fontName=bold, fontSize=14, leading=18, textColor=ACCENT, spaceBefore=14, spaceAfter=6)
        self.h2 = ParagraphStyle("h2", parent=self.body, fontName=bold, fontSize=11, leading=14, spaceBefore=8, spaceAfter=3)
        self.bullet = ParagraphStyle("bullet", parent=self.body, leftIndent=12, bulletIndent=2, spaceAfter=2)

    def p(self, text: str, style: ParagraphStyle | None = None) -> Paragraph:
        return Paragraph(self.esc(text), style or self.body)

    def esc(self, text: Any) -> str:
        s = str(text if text is not None else "")
        if not self.unicode_ok:
            s = _latin1(s)
        return escape(s).replace("\n", "<br/>")


def _src_label(source: dict | None) -> str:
    if not source:
        return ""
    page = f", p. {source['page_number']}" if source.get("page_number") else ""
    return f"{source['document_name']}{page}"


def _refs(sources: list[dict]) -> str:
    seen, out = set(), []
    for s in sources:
        label = _src_label(s)
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return "; ".join(out)


def _table(st: _Styles, rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    data = []
    for r_i, row in enumerate(rows):
        style = st.cell_head if header and r_i == 0 else st.cell
        data.append([c if not isinstance(c, str) else Paragraph(st.esc(c), style) for c in row])
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    table.setStyle(TableStyle(cmds))
    return table


# ------------------------------------------------------------- sections ----
def _section_summary(st: _Styles, summaries: list[dict], story: list) -> None:
    story.append(st.p("Executive summary", st.h1))
    if not summaries:
        story.append(st.p("No executive summary was included in this report.", st.small))
        return
    for analysis in summaries:
        data = SummaryResult.model_validate(analysis["result"])
        for doc in data.documents:
            story.append(st.p(doc.document_name, st.h2))
            if not doc.sections:
                story.append(st.p("Unable to determine a summary for this document.", st.small))
            for sec in doc.sections:
                story.append(st.p(sec.heading, st.h2))
                for item in sec.items:
                    tag = "Stated" if item.kind == "stated" else "AI interpretation"
                    refs = _refs([s.model_dump() for s in item.sources])
                    line = f"[{tag}] {item.text}" + (f"  (Source: {refs})" if refs else "")
                    story.append(Paragraph(st.esc(line), st.bullet, bulletText="-"))
    story.append(
        st.p("Items marked 'AI interpretation' are the model's own judgement, not statements from the documents.", st.small)
    )


def _section_extraction(st: _Styles, analyses: list[dict], width: float, story: list) -> None:
    story.append(st.p("Key extracted information", st.h1))
    if not analyses:
        story.append(st.p("No structured extraction was included in this report.", st.small))
        return
    for analysis in analyses:
        data = ExtractionResult.model_validate(analysis["result"])
        for doc in data.documents:
            story.append(st.p(doc.document_name, st.h2))
            rows: list[list[Any]] = [["Field", "Value", "Source"]]
            for f in doc.fields:
                value = f.value if f.value else STATUS_LABEL.get(f.status, "")
                if f.status == "unverified":
                    value += " (unverified)"
                rows.append([f.label, value, _src_label(f.source.model_dump()) if f.source else ""])
            story.append(_table(st, rows, [width * 0.25, width * 0.5, width * 0.25]))
            story.append(Spacer(1, 4))


def _section_comparison(st: _Styles, analyses: list[dict], width: float, story: list) -> list[dict]:
    """Returns all comparison findings' sources for the reference list."""
    story.append(st.p("Comparison results", st.h1))
    collected: list[dict] = []
    if not analyses:
        story.append(st.p("No comparison was included in this report.", st.small))
        return collected
    for analysis in analyses:
        data = ComparisonResult.model_validate(analysis["result"])
        names = [d.name for d in data.documents]
        story.append(st.p("Documents compared: " + ", ".join(names), st.body))
        story.append(Spacer(1, 4))
        if data.mode == "fields":
            col = (width * 0.76) / max(len(data.documents), 1)
            rows: list[list[Any]] = [["Field", *names]]
            for row in data.rows:
                cells = []
                for c in row.cells:
                    text = display_value(c)
                    if c.status == "unverified":
                        text += " (unverified)"
                    if c.source:
                        text += f"\n[{_src_label(c.source.model_dump())}]"
                        collected.append(c.source.model_dump())
                    cells.append(text)
                rows.append([row.label, *cells])
            story.append(_table(st, rows, [width * 0.24, *[col] * len(data.documents)]))
            story.append(Spacer(1, 6))
            story.append(st.p("Important differences and missing information", st.h2))
            notable = [f for f in data.findings if f.kind in ("difference", "missing", "not_found", "warning")]
            if not notable:
                story.append(st.p("No differences or missing information were found for the compared fields.", st.body))
            for f in notable:
                story.append(Paragraph(st.esc(f.text), st.bullet, bulletText="-"))
                collected.extend(s.model_dump() for s in f.sources)
        elif data.version_diff:
            d = data.version_diff
            story.append(st.p(data.findings[0].text if data.findings else "", st.body))
            rows = [["Change", "Page (A)", "Version A", "Page (B)", "Version B"]]
            for ch in d.changes[:60]:
                rows.append([ch.kind, str(ch.a_page or ""), ch.a_text or "", str(ch.b_page or ""), ch.b_text or ""])
            story.append(_table(st, rows, [width * 0.1, width * 0.08, width * 0.37, width * 0.08, width * 0.37]))
            if len(d.changes) > 60:
                story.append(st.p(f"Showing 60 of {len(d.changes)} differences.", st.small))
        for w in data.warnings:
            story.append(st.p("Note: " + w, st.small))
    return collected


def _section_findings(st: _Styles, qa: list[dict], comparisons: list[dict], summaries: list[dict], story: list) -> list[dict]:
    story.append(st.p("Key findings", st.h1))
    collected: list[dict] = []
    any_item = False
    for analysis in comparisons:
        data = ComparisonResult.model_validate(analysis["result"])
        if data.mode == "fields":
            for f in data.findings:
                if f.kind in ("difference", "missing", "match"):
                    story.append(Paragraph(st.esc(f.text), st.bullet, bulletText="-"))
                    collected.extend(s.model_dump() for s in f.sources)
                    any_item = True
    for analysis in summaries:
        data = SummaryResult.model_validate(analysis["result"])
        for doc in data.documents:
            for sec in doc.sections:
                for item in sec.items:
                    if item.kind == "stated" and item.sources:
                        story.append(Paragraph(st.esc(f"{item.text} ({doc.document_name})"), st.bullet, bulletText="-"))
                        collected.extend(s.model_dump() for s in item.sources)
                        any_item = True
                        break  # one headline item per section keeps this list short
    for analysis in qa:
        r = analysis["result"]
        story.append(st.p("Question: " + r.get("question", ""), st.h2))
        story.append(st.p(r.get("answer", ""), st.body))
        refs = _refs(r.get("sources", []))
        story.append(st.p("Sources: " + refs if refs else "No supporting source passage.", st.small))
        collected.extend(r.get("sources", []))
        any_item = True
    if not any_item:
        story.append(st.p("No key findings were available from the included analyses.", st.small))
    return collected


def _section_references(st: _Styles, sources: list[dict], story: list) -> None:
    story.append(st.p("Source references", st.h1))
    seen: dict[tuple, dict] = {}
    for s in sources:
        seen.setdefault((s["document_name"], s.get("page_number")), s)
    if not seen:
        story.append(st.p("No source passages were cited by the included analyses.", st.small))
        return
    story.append(st.p("Passages from the uploaded documents that support the statements above.", st.small))
    for (name, page), s in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        label = f"{name}" + (f", page {page}" if page else "")
        story.append(KeepTogether([st.p(label, st.h2), st.p('"' + truncate(s.get("snippet", ""), 260) + '"', st.small)]))


# ------------------------------------------------------------- builder ----
def build_report_pdf(
    analyses: list[dict[str, Any]], *, title: str, document_names: list[str], font_path: str = ""
) -> bytes:
    """Create the PDF from saved analysis dicts ({'operation','result','title',...})."""
    font, bold, unicode_ok = _register_fonts(font_path)
    st = _Styles(font, bold, unicode_ok)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=title, author="Document Intelligence Assistant",
    )
    width = A4[0] - 36 * mm
    by_op: dict[str, list[dict]] = {"summarize": [], "extract": [], "compare": [], "qa": []}
    for a in analyses:
        by_op.setdefault(a["operation"], []).append(a)
    exec_summaries = [a for a in by_op["summarize"] if a["result"].get("summary_type") == "executive"] or by_op["summarize"][:1]

    story: list = [st.p(title, st.title)]
    story.append(st.p(f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}", st.subtitle))
    story.append(st.p("Selected documents", st.h1))
    for name in document_names:
        story.append(Paragraph(st.esc(name), st.bullet, bulletText="-"))

    _section_summary(st, exec_summaries, story)
    _section_extraction(st, by_op["extract"], width, story)
    comp_sources = _section_comparison(st, by_op["compare"], width, story)
    finding_sources = _section_findings(st, by_op["qa"], by_op["compare"], exec_summaries, story)

    all_sources = list(comp_sources) + list(finding_sources)
    for a in by_op["extract"]:
        for d in a["result"].get("documents", []):
            all_sources.extend(f["source"] for f in d.get("fields", []) if f.get("source"))
    for a in exec_summaries:
        for d in a["result"].get("documents", []):
            for sec in d.get("sections", []):
                for item in sec.get("items", []):
                    all_sources.extend(item.get("sources", []))
    _section_references(st, all_sources, story)

    story.append(Spacer(1, 10))
    story.append(
        st.p(
            "This report was generated automatically from the uploaded documents. Values marked 'Not specified' were not "
            "found; 'unverified' values could not be confirmed in the cited passage. Verify important figures against the originals.",
            st.small,
        )
    )

    def footer(canvas, d):  # page numbers
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        canvas.drawString(18 * mm, 10 * mm, "Document Intelligence Assistant")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _safe_pdf_name(title: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:60] or "report"
    return f"{stem}.pdf"


def generate_report(
    ctx: AppContext, *, analysis_ids: list[int], title: str | None = None
) -> tuple[bytes, str]:
    """Build a report from saved analyses. Returns (pdf_bytes, filename)."""
    if not analysis_ids:
        raise InvalidRequestError("Select at least one saved analysis to include in the report.")
    rows = ctx.analyses.get_many(list(dict.fromkeys(analysis_ids)))
    missing = set(analysis_ids) - {r["id"] for r in rows}
    if missing:
        raise NotFoundError("Analysis not found: " + ", ".join(map(str, sorted(missing))))
    names: dict[int, str] = {}
    for row in rows:
        for k, v in (to_record(row).document_names or {}).items():
            names[k] = v
    title = (title or "").strip() or "Document Analysis Report"
    pdf = build_report_pdf(rows, title=title, document_names=list(names.values()), font_path=ctx.settings.report_font_path)
    return pdf, _safe_pdf_name(title)


def generate_full_report(
    ctx: AppContext, *, document_ids: list[int], title: str | None = None
) -> tuple[bytes, str, list[int]]:
    """One-click report: run executive summary + extraction (+ comparison if 2+ documents), then build the PDF.

    Returns (pdf_bytes, filename, ids_of_saved_analyses).
    """
    ids: list[int] = []
    base = dict(document_ids=document_ids, save=True)
    steps = [
        AnalysisRequest(operation="summarize", summary_type="executive", **base),
        AnalysisRequest(operation="extract", preset="vendor_proposal", **base),
    ]
    if len(set(document_ids)) >= 2:
        steps.append(AnalysisRequest(operation="compare", preset="vendor_proposal", **base))
    for step in steps:
        ids.append(run_analysis(ctx, step).id)  # type: ignore[arg-type]
    pdf, name = generate_report(ctx, analysis_ids=ids, title=title)
    return pdf, name, ids
