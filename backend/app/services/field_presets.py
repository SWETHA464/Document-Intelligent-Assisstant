"""Reusable field sets for structured extraction/comparison, plus free-text aspect matching."""
from __future__ import annotations

from app.core.errors import InvalidRequestError
from app.schemas.analysis import FieldSpec, unique_fields

DEFAULT_PRESET = "vendor_proposal"


def _f(key: str, label: str, description: str) -> FieldSpec:
    return FieldSpec(key=key, label=label, description=description)


PRESETS: dict[str, dict] = {
    "vendor_proposal": {
        "label": "Vendor proposal",
        "fields": [
            _f("company_name", "Company name", "Name of the client/customer company the proposal is addressed to, if stated"),
            _f("vendor_name", "Vendor name", "Name of the company submitting the proposal"),
            _f("proposal_date", "Proposal date", "Date the proposal was issued"),
            _f("contract_duration", "Contract duration", "Length or term of the contract or engagement"),
            _f("pricing", "Pricing", "Total price, fees or cost with currency, and the main price components"),
            _f("payment_terms", "Payment terms", "When and how payment is due (net days, milestones, advance payment)"),
            _f("delivery_timeline", "Delivery timeline", "Delivery or completion schedule, durations, milestones, deadlines"),
            _f("contact_information", "Contact information", "Email, phone or address of a contact person"),
            _f("key_obligations", "Key obligations", "Main obligations the vendor commits to"),
            _f("key_requirements", "Important requirements", "Important requirements, conditions or assumptions placed on the client"),
        ],
    },
    "contract": {
        "label": "Contract / agreement",
        "fields": [
            _f("parties", "Parties", "Names of the parties to the agreement"),
            _f("effective_date", "Effective date", "Date the agreement starts or is signed"),
            _f("contract_term", "Contract term", "Duration of the agreement"),
            _f("renewal_terms", "Renewal terms", "How and when the agreement renews"),
            _f("payment_terms", "Payment terms", "Amounts and when payment is due"),
            _f("termination", "Termination", "Conditions and notice period for ending the agreement"),
            _f("liability", "Liability", "Liability limits or indemnities"),
            _f("confidentiality", "Confidentiality", "Confidentiality obligations"),
            _f("governing_law", "Governing law", "Jurisdiction or governing law"),
            _f("key_obligations", "Key obligations", "Main obligations of each party"),
        ],
    },
    "report": {
        "label": "Business report / policy",
        "fields": [
            _f("title", "Title", "Title of the document"),
            _f("author_organization", "Author / organisation", "Who authored or issued the document"),
            _f("document_date", "Document date", "Date of publication or approval"),
            _f("period_covered", "Period covered", "Time period the document covers or applies to"),
            _f("key_figures", "Key figures", "Most important numbers, amounts or metrics"),
            _f("main_conclusions", "Main conclusions", "Main conclusions or decisions"),
            _f("recommendations", "Recommendations", "Recommendations or required actions"),
        ],
    },
}

# keyword -> vendor_proposal field key, used to turn "compare pricing and payment terms" into fields
_ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pricing": ("price", "pricing", "cost", "fee", "budget", "quote", "amount"),
    "payment_terms": ("payment", "invoice", "billing"),
    "delivery_timeline": ("delivery", "timeline", "schedule", "deadline", "lead time", "milestone"),
    "contract_duration": ("duration", "contract term", "contract length", "period", "term of"),
    "contact_information": ("contact", "email", "phone"),
    "key_obligations": ("obligation", "responsibilit"),
    "key_requirements": ("requirement", "condition", "assumption"),
    "proposal_date": ("proposal date", "dated"),
    # Entity-name fields need explicit phrases: words like "vendors" usually name the
    # things being compared ("compare pricing across vendors"), not an aspect to extract.
    "vendor_name": ("vendor name", "supplier name", "name of the vendor", "name of the supplier"),
    "company_name": ("company name", "client name", "customer name", "name of the company"),
}


def list_presets() -> list[dict]:
    return [
        {"name": name, "label": p["label"], "fields": [f.model_dump() for f in p["fields"]]}
        for name, p in PRESETS.items()
    ]


def get_preset_fields(name: str) -> list[FieldSpec]:
    preset = PRESETS.get(name)
    if preset is None:
        raise InvalidRequestError(f"Unknown preset '{name}'. Available: {', '.join(PRESETS)}.")
    return list(preset["fields"])


def fields_from_text(text: str | None) -> list[FieldSpec]:
    """Pick vendor-proposal fields whose keywords appear in free text (may return [])."""
    if not text:
        return []
    lowered = text.lower()
    wanted = {key for key, words in _ASPECT_KEYWORDS.items() if any(w in lowered for w in words)}
    return [f for f in PRESETS[DEFAULT_PRESET]["fields"] if f.key in wanted]


def resolve_fields(
    *, preset: str | None, custom: list[FieldSpec] | None, text: str | None
) -> tuple[str | None, list[FieldSpec]]:
    """Priority: custom fields > named preset > aspects mentioned in text > default preset."""
    if custom:
        return None, unique_fields(custom)
    if preset:
        return preset, get_preset_fields(preset)
    matched = fields_from_text(text)
    if matched:
        return None, matched
    return DEFAULT_PRESET, get_preset_fields(DEFAULT_PRESET)
