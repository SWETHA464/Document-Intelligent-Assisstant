"""Model-integration boundary. Application code only ever sees `LLMClient`."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        """Return the model's text reply.

        Raises `LLMUnavailableError` if the model cannot be reached or is misconfigured.
        When `json_mode` is True the provider is asked to emit a JSON object.
        """

    def describe(self) -> dict:
        return {"provider": "unknown", "model": "unknown"}
