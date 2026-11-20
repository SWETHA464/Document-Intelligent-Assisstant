"""Application settings, loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[2]

LLM_PROVIDERS = ("ollama", "openai")
EMBEDDING_PROVIDERS = ("ollama", "sentence_transformers", "hashing")

# Settings that may be changed at runtime from the Settings page (persisted in SQLite).
RUNTIME_EDITABLE = (
    "llm_provider",
    "llm_model",
    "llm_base_url",
    "llm_temperature",
    "embedding_provider",
    "embedding_model",
)


def _load_dotenv() -> None:
    """Load backend/.env if python-dotenv is installed. Real env vars take precedence."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements
        return
    load_dotenv(BACKEND_DIR / ".env", override=False)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    data_dir: Path = BACKEND_DIR / "data"
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    max_upload_mb: int = 25

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = field(default="", repr=False)
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 180
    llm_num_ctx: int = 8192

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = ""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    hybrid_alpha: float = 0.6
    min_term_coverage: float = 0.3
    min_semantic_score: float = 0.55
    max_context_chars: int = 12000

    report_font_path: str = ""

    # ------------------------------------------------------------ paths --
    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.embedding_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- helpers --
    def with_overrides(self, overrides: dict[str, Any]) -> "Settings":
        """Return a copy with runtime-editable values replaced (unknown keys ignored)."""
        allowed = {k: v for k, v in overrides.items() if k in RUNTIME_EDITABLE}
        return replace(self, **allowed)

    def public_dict(self) -> dict[str, Any]:
        """Settings that are safe to show in the UI (never includes secrets)."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_temperature": self.llm_temperature,
            "llm_num_ctx": self.llm_num_ctx,
            "llm_has_api_key": bool(self.llm_api_key),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
            "hybrid_alpha": self.hybrid_alpha,
            "max_upload_mb": self.max_upload_mb,
        }

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        data_dir_raw = os.environ.get("DATA_DIR", "").strip()
        origins = _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        chunk_size = max(200, _env_int("CHUNK_SIZE", 1000))
        overlap = max(0, min(_env_int("CHUNK_OVERLAP", 150), chunk_size // 2))
        return cls(
            data_dir=Path(data_dir_raw) if data_dir_raw else BACKEND_DIR / "data",
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            max_upload_mb=max(1, _env_int("MAX_UPLOAD_MB", 25)),
            llm_provider=_env("LLM_PROVIDER", "ollama").lower(),
            llm_model=_env("LLM_MODEL", "llama3.2"),
            llm_base_url=_env("LLM_BASE_URL", "http://localhost:11434"),
            llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 180),
            llm_num_ctx=_env_int("LLM_NUM_CTX", 8192),
            embedding_provider=_env("EMBEDDING_PROVIDER", "ollama").lower(),
            embedding_model=_env("EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", "").strip(),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            retrieval_top_k=max(1, _env_int("RETRIEVAL_TOP_K", 8)),
            hybrid_alpha=min(1.0, max(0.0, _env_float("HYBRID_ALPHA", 0.6))),
            min_term_coverage=_env_float("MIN_TERM_COVERAGE", 0.3),
            min_semantic_score=_env_float("MIN_SEMANTIC_SCORE", 0.55),
            max_context_chars=max(2000, _env_int("MAX_CONTEXT_CHARS", 12000)),
            report_font_path=os.environ.get("REPORT_FONT_PATH", "").strip(),
        )
