"""LangChain-backed chat model (Ollama by default, OpenAI-compatible optionally).

LangChain imports are lazy so the rest of the app (and the test-suite) works without them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.errors import LLMUnavailableError
from app.services.llm.base import LLMClient

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """LangChain message content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class LangChainLLM(LLMClient):
    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._models: dict[bool, Any] = {}

    def describe(self) -> dict:
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url if self.settings.llm_provider == "ollama" else None,
        }

    def _build(self, json_mode: bool) -> Any:
        s = self.settings
        if s.llm_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise LLMUnavailableError(
                    "langchain-ollama is not installed. Run: pip install langchain-ollama"
                ) from exc
            kwargs: dict[str, Any] = dict(
                model=s.llm_model,
                base_url=s.llm_base_url,
                temperature=s.llm_temperature,
                num_ctx=s.llm_num_ctx,
                client_kwargs={"timeout": s.llm_timeout_seconds},
            )
            if json_mode:
                kwargs["format"] = "json"
            return ChatOllama(**kwargs)
        if s.llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise LLMUnavailableError(
                    "langchain-openai is not installed. Run: pip install langchain-openai"
                ) from exc
            if not s.llm_api_key:
                raise LLMUnavailableError("LLM_API_KEY is not set (required for LLM_PROVIDER=openai).")
            kwargs = dict(
                model=s.llm_model,
                api_key=s.llm_api_key,
                temperature=s.llm_temperature,
                timeout=s.llm_timeout_seconds,
            )
            if s.llm_base_url and "11434" not in s.llm_base_url:
                kwargs["base_url"] = s.llm_base_url
            model = ChatOpenAI(**kwargs)
            return model.bind(response_format={"type": "json_object"}) if json_mode else model
        raise LLMUnavailableError(f"Unknown LLM_PROVIDER '{s.llm_provider}'. Use 'ollama' or 'openai'.")

    def generate(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise LLMUnavailableError("langchain-core is not installed. Run: pip install langchain-core") from exc

        if json_mode not in self._models:
            self._models[json_mode] = self._build(json_mode)
        model = self._models[json_mode]
        try:
            reply = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            hint = (
                f"Is Ollama running at {self.settings.llm_base_url} and is the model pulled "
                f"(ollama pull {self.settings.llm_model})?"
                if self.settings.llm_provider == "ollama"
                else "Check LLM_API_KEY, LLM_MODEL and LLM_BASE_URL."
            )
            raise LLMUnavailableError(
                f"The language model '{self.settings.llm_model}' could not be reached. {hint} Details: {exc}"
            ) from exc
        return _content_to_text(reply.content)
