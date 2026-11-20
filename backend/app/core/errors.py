"""Application exceptions. Each carries an HTTP status and a stable machine-readable code."""
from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code = 400
    code = "app_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnsupportedFileError(AppError):
    status_code = 415
    code = "unsupported_file"


class EmptyFileError(AppError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


class DocumentNotReadyError(AppError):
    status_code = 409
    code = "document_not_ready"


class ExtractionError(AppError):
    """Text could not be extracted from a file (corrupted, empty, protected...)."""

    status_code = 422
    code = "extraction_failed"


class LLMUnavailableError(AppError):
    """The language model could not be reached or is not installed/configured."""

    status_code = 503
    code = "llm_unavailable"


class LLMOutputError(AppError):
    """The language model replied with output that could not be parsed/validated."""

    status_code = 502
    code = "llm_bad_output"


class EmbeddingError(AppError):
    """The embedding model could not be reached or failed."""

    status_code = 503
    code = "embedding_unavailable"
