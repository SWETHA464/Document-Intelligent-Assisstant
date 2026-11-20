"""Parse and validate structured (JSON) model output with one automatic repair retry."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.errors import LLMOutputError
from app.services.llm.base import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull a JSON object out of model output (tolerates code fences and surrounding prose)."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc


def generate_validated(
    llm: LLMClient, model_cls: type[T], system: str, prompt: str, *, retries: int = 1
) -> T:
    """Call the LLM in JSON mode and validate against `model_cls`.

    On malformed output, asks the model once more (with the error) before raising
    `LLMOutputError`. `LLMUnavailableError` is never swallowed.
    """
    reply = llm.generate(system, prompt, json_mode=True)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            return model_cls.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as exc:
            last_error = _short(exc)
            logger.warning("Malformed model output (attempt %s): %s", attempt + 1, last_error)
            if attempt == retries:
                break
            reply = llm.generate(
                system,
                f"{prompt}\n\nYour previous reply was not valid ({last_error}). "
                "Reply again with ONLY one valid JSON object that follows the requested schema.",
                json_mode=True,
            )
    raise LLMOutputError(
        "The model returned output that could not be understood. Try again, or use a larger/better "
        f"model in Settings. ({last_error})"
    )


def _short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    return text[:200]
