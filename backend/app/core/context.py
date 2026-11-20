"""AppContext: wires settings, database, repositories and model clients together.

Services receive the context explicitly, which keeps them easy to test (swap in a fake LLM
or a hashing embedder) without global state.
"""
from __future__ import annotations

import logging
import threading

from app.core.config import Settings
from app.database.db import Database
from app.database.repositories import (
    AnalysisRepository,
    CollectionRepository,
    ContentRepository,
    DocumentRepository,
    SettingsRepository,
)
from app.rag.embeddings import Embedder, create_embedder
from app.services.llm import LangChainLLM, LLMClient

logger = logging.getLogger(__name__)


class AppContext:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.base_settings = settings
        self.settings = settings
        settings.ensure_dirs()
        self.db = Database(settings.database_path)
        self.db.init_schema()

        self.collections = CollectionRepository(self.db)
        self.documents = DocumentRepository(self.db)
        self.content = ContentRepository(self.db)
        self.analyses = AnalysisRepository(self.db)
        self.app_settings = SettingsRepository(self.db)

        # Explicit overrides (tests) win over anything built from settings.
        self._llm_override = llm
        self._embedder_override = embedder
        self._llm: LLMClient | None = None
        self._embedder: Embedder | None = None
        self._lock = threading.Lock()
        self.reload_settings()

    # ---------------------------------------------------------- settings --
    def reload_settings(self) -> None:
        """Apply runtime overrides stored in SQLite on top of the environment settings."""
        with self._lock:
            self.settings = self.base_settings.with_overrides(self.app_settings.get_all())
            self._llm = None
            self._embedder = None

    def update_runtime_settings(self, values: dict) -> None:
        self.app_settings.set_many(values)
        self.reload_settings()

    def reset_runtime_settings(self) -> None:
        self.app_settings.clear()
        self.reload_settings()

    # ------------------------------------------------------------ models --
    def get_llm(self) -> LLMClient:
        if self._llm_override is not None:
            return self._llm_override
        with self._lock:
            if self._llm is None:
                self._llm = LangChainLLM(self.settings)
            return self._llm

    def get_embedder(self) -> Embedder:
        if self._embedder_override is not None:
            return self._embedder_override
        with self._lock:
            if self._embedder is None:
                self._embedder = create_embedder(self.settings)
            return self._embedder
